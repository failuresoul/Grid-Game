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
    """Game session state machine values."""
    WAITING = auto()   # Waiting for the player to leave the start zone
    RUNNING = auto()   # Active gameplay
    PAUSED  = auto()   # Player pressed P
    WIN     = auto()   # Player reached the end zone
    TIMEOUT = auto()   # Time limit exceeded


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

        self.state: GameState = GameState.WAITING
        self._session_start: Optional[float] = None
        self._session_end:   Optional[float] = None

        # Final computed metrics (set when session ends)
        self.final_metrics: Optional[dict] = None
        self._metrics_saved: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Main update — called every frame by main.py
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[int, int]], dt: float) -> None:
        """
        Advance the engine by one frame.

        Args:
            cursor: Smoothed (cx, cy) from HandTracker, or None if no hand.
            dt:     Seconds elapsed since the previous frame.
        """
        if self.state in (GameState.PAUSED, GameState.WIN, GameState.TIMEOUT):
            return

        if cursor is None:
            return   # freeze player when hand is not visible

        tx, ty = float(cursor[0]), float(cursor[1])

        # ── WAITING → RUNNING ─────────────────────────────────────────────────
        if self.state == GameState.WAITING:
            dist = math.dist((tx, ty), self.level.start)
            if dist > self.level.start_r + self.player.radius:
                self._begin_session()

        # ── RUNNING: move, record, check win/timeout ──────────────────────────
        if self.state == GameState.RUNNING:
            self.player.move_to_cursor(tx, ty, self.level.walls)
            self.player.record_trail()

            if self.metrics:
                self.metrics.record(self.player.px, self.player.py, dt)

            # Win check
            if math.dist(self.player.position_f, self.level.end) <= (
                self.level.end_r + self.player.radius
            ):
                self._end_session(won=True)
                return

            # Time-limit check
            limit = self.difficulty_cfg.time_limit_sec
            if limit > 0 and self.elapsed_time > limit:
                self._end_session(won=False)

    # ─────────────────────────────────────────────────────────────────────────
    #  State transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _begin_session(self) -> None:
        self.state          = GameState.RUNNING
        self._session_start = time.perf_counter()
        if self.metrics:
            self.metrics.start_recording()
        log.info("Session RUNNING")

    def _end_session(self, won: bool) -> None:
        self._session_end = time.perf_counter()
        self.state        = GameState.WIN if won else GameState.TIMEOUT
        if self.metrics:
            self.metrics.stop_recording()
            self.final_metrics = self.metrics.compute()
        log.info(
            f"Session {'WIN' if won else 'TIMEOUT'} | "
            f"time={self.elapsed_time:.1f}s | "
            f"wall_hits={self.player.wall_hit_count}"
        )

    def toggle_pause(self) -> None:
        """Toggle between RUNNING and PAUSED."""
        if self.state == GameState.RUNNING:
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.RUNNING

    # ─────────────────────────────────────────────────────────────────────────
    #  Read-only properties (used by Renderer and main.py)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def player_pos(self) -> Tuple[int, int]:
        return self.player.position

    @property
    def trail(self):
        return self.player.trail

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
    def is_finished(self) -> bool:
        return self.state in (GameState.WIN, GameState.TIMEOUT)
