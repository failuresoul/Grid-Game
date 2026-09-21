"""
game/game_engine.py — Central Game Engine (State Machine)

Manages:
  - State machine: WAITING → RUNNING → WIN / TIMEOUT / PAUSED
  - Delegates movement + collision to game.player.Player
  - Delegates metric recording to metrics.performance.MetricsCollector
  - Exposes read-only properties used by the Renderer / HUD

Dependency chain (no circular imports):
    game.game_engine → game.player, game.levels, config
    game.game_engine accepts a MetricsCollector instance at construction
    (injected by main.py) — it does NOT import metrics directly.
"""

from __future__ import annotations
import math
import time
import logging
from enum import Enum, auto
from typing import Optional, Tuple, TYPE_CHECKING

import config
from game.levels import Level
from game.player import Player

if TYPE_CHECKING:
    # Import only for type-checking tools; avoids a runtime circular import
    # if metrics ever imports from game in the future.
    from metrics.performance import MetricsCollector

log = logging.getLogger(__name__)


class GameState(Enum):
    """
    Game session state machine:
        MENU -> LEVEL_SELECT -> READY -> PLAYING -> COMPLETED -> RESULTS
    """
    MENU         = auto()   # Title screen / difficulty selection
    LEVEL_SELECT = auto()   # Maze level browser and preview
    READY        = auto()   # Level loaded, cursor at START, waiting for movement
    PLAYING      = auto()   # Active movement: timer active, CCD active, metrics recorded
    PAUSED       = auto()   # Session paused
    COMPLETED    = auto()   # Reached END: timer stopped immediately, metrics finalized, session saved
    RESULTS      = auto()   # Results screen displaying full clinical breakdown
    TIMEOUT      = auto()   # Time limit exceeded
    HISTORY      = auto()   # Progress / historical session performance & trends


# Backward compatibility aliases for existing regression tests and modules:
GameState.WAITING  = GameState.READY
GameState.RUNNING  = GameState.PLAYING
GameState.WIN      = GameState.RESULTS
GameState.PROGRESS = GameState.HISTORY


class GameEngine:
    """
    Orchestrates one maze session.

    The hand-cursor position is supplied each frame by the main loop.
    GameEngine moves the Player, checks win/timeout conditions, and
    transitions the state machine accordingly.

    Usage:
        engine = GameEngine(level, difficulty_cfg, metrics_collector)
        while True:
            cursor = tracker.process(frame)    # (cx, cy) or None
            engine.update(cursor, dt)
            canvas = renderer.draw(engine, dt)
    """

    def __init__(
        self,
        level:         Level,
        difficulty_cfg,                   # config.DifficultyConfig
        metrics_collector = None,          # metrics.performance.MetricsCollector
    ) -> None:
        self.level          = level
        self.difficulty_cfg = difficulty_cfg
        self.metrics        = metrics_collector   # type: Optional[MetricsCollector]

        self.player = Player(
            start_x=float(level.start[0]),
            start_y=float(level.start[1]),
            radius=difficulty_cfg.player_radius,
        )

        self.state: GameState = GameState.READY
        self._session_start: Optional[float] = None
        self._session_end:   Optional[float] = None

        # Final computed metrics (set when session ends)
        self.final_metrics: Optional[dict] = None
        self._metrics_saved: bool = False
        self.last_saved_session_file: Optional[str] = None
        self.adaptive_recommendation = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Main update — called every frame by main.py
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[float, float]], dt: float) -> None:
        """
        Advance the engine by one frame.

        Args:
            cursor: Continuous smoothed (cx, cy) in float coords, or None if no input.
            dt:     Seconds elapsed since the previous frame.
        """
        if self.state in (GameState.PAUSED, GameState.COMPLETED, GameState.RESULTS, GameState.TIMEOUT, GameState.MENU, GameState.LEVEL_SELECT, GameState.HISTORY):
            return

        self.player.update_timers(dt)

        if cursor is None:
            return   # freeze player when hand/mouse input is not active

        tx, ty = float(cursor[0]), float(cursor[1])

        # ── READY → PLAYING (start timer when movement actually begins) ──────
        if self.state in (GameState.READY, GameState.WAITING):
            dist = math.dist((tx, ty), self.level.start)
            if dist > self.level.start_r + self.player.radius:
                self.start_playing()

        # ── PLAYING: move, record, check win/timeout ──────────────────────────
        if self.state in (GameState.PLAYING, GameState.RUNNING):
            moved = self.player.move_to_cursor(tx, ty, self.level.walls)
            if moved:
                self.player.record_trail()

            if self.metrics:
                self.metrics.record(self.player.px, self.player.py, dt)

            # Win check: player reaches END target
            if math.dist(self.player.position_f, self.level.end) <= (
                self.level.end_r + self.player.radius
            ):
                self.complete_session(won=True)
                return

            # Time-limit check
            limit = self.difficulty_cfg.time_limit_sec
            if limit > 0 and self.elapsed_time > limit:
                self.complete_session(won=False)

    # ─────────────────────────────────────────────────────────────────────────
    #  State transitions
    # ─────────────────────────────────────────────────────────────────────────

    def start_playing(self) -> None:
        """Begin active gameplay session and start timer."""
        self.state          = GameState.PLAYING
        self._session_start = time.perf_counter()
        self._session_end   = None
        if self.metrics:
            self.metrics.start_recording()
        log.info("Session PLAYING — Timer started")

    def _begin_session(self) -> None:
        """Backward-compatible alias for start_playing."""
        self.start_playing()

    def complete_session(self, won: bool = True) -> None:
        """
        Player reached END (or timed out).
        Immediately freezes the timer, finalizes clinical metrics, saves session,
        and transitions to RESULTS.
        """
        self._session_end = time.perf_counter()
        self.state        = GameState.COMPLETED if won else GameState.TIMEOUT

        if self.metrics:
            self.metrics.stop_recording()
            self.final_metrics = self.metrics.compute()

        # Compute clinical adaptive difficulty recommendation
        try:
            from metrics.adaptive import evaluate_adaptive_difficulty
            from metrics.history_reader import load_all_sessions
            history = load_all_sessions()
            diff_tag = (getattr(self.level, "difficulty_tag", None) or self.difficulty_cfg.name).upper()
            self.adaptive_recommendation = evaluate_adaptive_difficulty(
                current_metrics=self.final_metrics or {},
                current_difficulty=diff_tag,
                wall_hits=self.player.wall_hit_count,
                completion_status="COMPLETED" if won else "TIMEOUT",
                session_history=history,
            )
            if self.adaptive_recommendation and self.final_metrics is not None:
                self.final_metrics["adaptive_recommendation"] = self.adaptive_recommendation.to_dict()
        except Exception as e:
            log.warning(f"Could not compute adaptive difficulty recommendation: {e}")

        # Save session CSV immediately upon completion
        self.save_session_metrics()

        # Transition to RESULTS display
        if won:
            self.state = GameState.RESULTS

        log.info(
            f"Session {'COMPLETED/RESULTS' if won else 'TIMEOUT'} | "
            f"time={self.elapsed_time:.2f}s | "
            f"wall_hits={self.player.wall_hit_count}"
        )

    def _end_session(self, won: bool) -> None:
        """Backward-compatible alias for complete_session."""
        self.complete_session(won=won)

    def restart(self) -> None:
        """
        Restart the current level back to READY state:
        Resets player position, clears trajectory, zeroes timers, and resets metrics.
        """
        self.player.reset(self.level.start[0], self.level.start[1])
        self._session_start = None
        self._session_end   = None
        self.final_metrics  = None
        self._metrics_saved = False
        self.last_saved_session_file = None
        self.adaptive_recommendation = None
        if self.metrics:
            self.metrics.start_recording()
            self.metrics.stop_recording()
        self.state = GameState.READY
        log.info("Level RESTARTED -> State: READY (Timer: 0.0s)")

    def save_session_metrics(self) -> Optional[str]:
        """Save session recording (JSON + daily CSV) exactly once upon completion."""
        if self._metrics_saved:
            return self.last_saved_session_file

        self._metrics_saved = True
        diff_name = self.difficulty_cfg.name if hasattr(self.difficulty_cfg, "name") else str(getattr(self.level, "difficulty", "EASY"))
        lvl_name = getattr(self.level, "name", "Level")
        maze_seed = getattr(self.level, "seed", None)
        status = "COMPLETED" if self.state in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN) else "TIMEOUT"

        # Trajectory points
        traj_samples = []
        if self.metrics and hasattr(self.metrics, "get_trajectory_samples"):
            traj_samples = self.metrics.get_trajectory_samples()
        if not traj_samples and hasattr(self.player, "trajectory"):
            traj_samples = [{"x": p[0], "y": p[1], "t": 0.0} for p in self.player.trajectory]

        # Metrics values
        m = self.final_metrics or {}
        completion_t = float(m.get("completion_time_s", self.elapsed_time))
        act_dist = float(m.get("actual_distance", self.actual_distance))
        min_dist = float(m.get("minimum_distance", self.minimum_distance))
        eff = float(m.get("path_efficiency", self.path_efficiency))
        acc = float(m.get("trajectory_accuracy", self.trajectory_accuracy))
        smooth = float(m.get("smoothness_score", self.smoothness_score))
        wall_hits = int(self.player.wall_hit_count)
        dev_cnt = int(m.get("deviation_events", 0))

        from metrics.session_recorder import save_session
        saved_file = save_session(
            difficulty=diff_name,
            maze_seed=maze_seed,
            completion_status=status,
            completion_time=completion_t,
            actual_distance=act_dist,
            minimum_distance=min_dist,
            path_efficiency=eff,
            accuracy=acc,
            smoothness=smooth,
            collision_count=wall_hits,
            deviation_count=dev_cnt,
            trajectory=traj_samples,
            optimal_path=getattr(self.level, "optimal_waypoints", []),
            level_name=lvl_name,
            game_performance_metrics=m,
            save_csv_also=True,
        )
        self.last_saved_session_file = saved_file
        return saved_file

    def toggle_pause(self) -> None:
        """Toggle between PLAYING and PAUSED."""
        if self.state in (GameState.PLAYING, GameState.RUNNING):
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING

    # ─────────────────────────────────────────────────────────────────────────
    #  Read-only properties (used by Renderer and main.py)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def player_pos(self) -> Tuple[int, int]:
        return self.player.position

    @property
    def minimum_distance(self) -> float:
        """Minimum collision-free path distance from START to END in pixels."""
        if hasattr(self.level, "minimum_path_distance") and self.level.minimum_path_distance is not None:
            return float(self.level.minimum_path_distance)
        return float(math.dist(self.level.start, self.level.end))

    @property
    def actual_distance(self) -> float:
        """Total actual smoothed hand/player movement distance in pixels/game units."""
        return self.player.actual_distance

    @property
    def path_efficiency(self) -> float:
        """Path efficiency percentage (minimum_distance / actual_distance * 100)."""
        from metrics.distance import path_efficiency
        return path_efficiency(actual=self.actual_distance, minimum=self.minimum_distance)

    @property
    def trajectory_accuracy(self) -> float:
        """Trajectory accuracy percentage [0.0%, 100.0%] along intended route."""
        if self.metrics is not None:
            return self.metrics.trajectory_accuracy
        from metrics.accuracy import trajectory_accuracy_percentage
        traj = self.player.full_trajectory
        if len(traj) < 2:
            return 0.0
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        wps = getattr(self.level, "optimal_waypoints", [self.level.start, self.level.end])
        return trajectory_accuracy_percentage(xs, ys, wps)

    @property
    def mean_path_deviation(self) -> float:
        """Mean orthogonal path deviation from intended route in pixels."""
        if self.metrics is not None:
            return self.metrics.mean_path_deviation
        from metrics.accuracy import path_deviation_summary
        traj = self.player.full_trajectory
        if len(traj) < 1:
            return 0.0
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        wps = getattr(self.level, "optimal_waypoints", [self.level.start, self.level.end])
        return path_deviation_summary(xs, ys, wps)["mean_path_deviation"]

    @property
    def time_outside_route_s(self) -> float:
        """Cumulative seconds spent outside the intended route corridor."""
        if self.metrics is not None:
            return self.metrics.time_outside_route_s
        return 0.0

    @property
    def deviation_events(self) -> int:
        """Count of distinct wrong movement / excursion episodes outside intended corridor."""
        if self.metrics is not None:
            return self.metrics.deviation_events
        return 0

    @property
    def smoothness_score(self) -> float:
        """
        Game-derived movement smoothness score [0.0, 100.0] (NOT a clinical measure).
        Higher = smoother movement. Evaluates directional and speed variations.
        """
        if self.metrics is not None:
            return self.metrics.smoothness_score
        from metrics.smoothness import movement_smoothness_score
        traj = self.player.full_trajectory
        if len(traj) < 3:
            return 100.0
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        return movement_smoothness_score(xs, ys)

    @property
    def raw_trajectory(self) -> List[Tuple[float, float, float]]:
        """Raw trajectory coordinates with timestamps retained for future algorithm enhancements."""
        if self.metrics is not None:
            return self.metrics.raw_trajectory
        return [(p[0], p[1], 0.0) for p in self.player.full_trajectory]

    @property
    def trail(self):
        return self.player.trail

    @property
    def trajectory(self):
        return self.player.trajectory

    @property
    def full_trajectory(self):
        return self.player.full_trajectory

    @property
    def wall_hit_count(self) -> int:
        return self.player.wall_hit_count

    @property
    def radius(self) -> int:
        return self.player.radius

    @property
    def elapsed_time(self) -> float:
        """Session duration in seconds (0 while WAITING)."""
        if self._session_start is None:
            return 0.0
        if self._session_end is not None:
            return self._session_end - self._session_start
        return time.perf_counter() - self._session_start

    @property
    def distance_to_end(self) -> float:
        return math.dist(self.player.position_f, self.level.end)

    @property
    def is_completed(self) -> bool:
        """True if the player successfully reached the goal and completed the maze."""
        return self.state in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN)

    @property
    def is_finished(self) -> bool:
        """True if the session has concluded (completed or timed out)."""
        return self.state in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN, GameState.TIMEOUT)
