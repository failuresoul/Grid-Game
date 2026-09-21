"""
game_state.py — Player State Machine and Collision System

Manages:
  - Player position (continuous pixel coordinates)
  - Collision detection with rectangular walls (push-back response)
  - Win detection (player circle overlaps the End zone)
  - Game state machine: WAITING → RUNNING → WIN / TIMEOUT / PAUSED
  - Tracking trail (list of recent positions for the renderer)
"""

from __future__ import annotations
import math
import time
import logging
from enum import Enum, auto
from typing import List, Optional, Tuple

import config
from maze_generator import Level, Rect

log = logging.getLogger(__name__)


class State(Enum):
    WAITING  = auto()   # Waiting for player to move into start zone / camera detected
    RUNNING  = auto()   # Active gameplay
    PAUSED   = auto()   # Player pressed P
    WIN      = auto()   # Player reached the End zone
    TIMEOUT  = auto()   # Time limit exceeded (difficulty with time limit)


# Maximum number of trail points stored (older ones discarded)
TRAIL_MAX_LENGTH = 400


class GameState:
    """
    Central game logic for one maze session.

    The cursor (cx, cy) is supplied each frame by HandTracker.
    GameState moves the player, checks collisions, updates the state machine,
    and exposes everything the Renderer needs.

    Usage:
        state = GameState(level, difficulty_cfg)
        while True:
            cursor = tracker.process(frame)
            state.update(cursor, dt)
            renderer.draw(canvas, state, metrics)
    """

    def __init__(
        self,
        level: Level,
        difficulty_cfg,                  # config.DifficultyConfig
        metrics_collector = None,         # metrics.MetricsCollector (optional)
    ) -> None:
        self.level          = level
        self.difficulty_cfg = difficulty_cfg
        self.metrics        = metrics_collector

        # ── Player ────────────────────────────────────────────────────────────
        # Start at the level's start position
        self.px: float = float(level.start[0])
        self.py: float = float(level.start[1])
        self.radius: int = difficulty_cfg.player_radius

        # ── State machine ─────────────────────────────────────────────────────
        self.state: State = State.WAITING
        self._session_start_time: Optional[float] = None
        self._session_end_time:   Optional[float] = None

        # ── Trail ─────────────────────────────────────────────────────────────
        self.trail: List[Tuple[int, int]] = []

        # ── Last valid cursor (for tracking "no hand detected") ───────────────
        self._last_cursor: Optional[Tuple[int, int]] = None

        # ── Collision counters ────────────────────────────────────────────────
        self.wall_hit_count: int = 0

        # ── Summary metrics (set when session ends) ───────────────────────────
        self.final_metrics: Optional[dict] = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Main update — called every frame
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[int, int]], dt: float) -> None:
        """
        Advance the game by one frame.

        Args:
            cursor: (cx, cy) from HandTracker, or None if no hand detected.
            dt:     seconds since last frame.
        """
        if self.state == State.PAUSED:
            return

        if self.state in (State.WIN, State.TIMEOUT):
            return   # session ended, freeze

        if cursor is None:
            # No hand detected — freeze player in place
            return

        target_x, target_y = float(cursor[0]), float(cursor[1])
        self._last_cursor = cursor

        # ── Transition: WAITING → RUNNING when player moves away from start ──
        if self.state == State.WAITING:
            dist_from_start = math.dist((target_x, target_y), self.level.start)
            if dist_from_start > self.level.start_r + self.radius:
                self._begin_running()

        # ── Move player toward cursor target ─────────────────────────────────
        if self.state == State.RUNNING:
            self._move_player(target_x, target_y)
            self._record_trail()

            # Record metrics sample
            if self.metrics:
                self.metrics.record(self.px, self.py, dt)

            # ── Check win ────────────────────────────────────────────────────
            dist_to_end = math.dist((self.px, self.py), self.level.end)
            if dist_to_end <= self.level.end_r + self.radius:
                self._end_session(won=True)
                return

            # ── Check time limit ─────────────────────────────────────────────
            limit = self.difficulty_cfg.time_limit_sec
            if limit > 0 and self.elapsed_time > limit:
                self._end_session(won=False)

    # ─────────────────────────────────────────────────────────────────────────
    #  Player movement & collision
    # ─────────────────────────────────────────────────────────────────────────

    def _move_player(self, target_x: float, target_y: float) -> None:
        """
        Snap player to the cursor target, then push back from any wall collisions.

        This approach (move-then-resolve) is simpler and more predictable for
        a rehabilitation game where the player needs clear, tactile feedback.
        """
        prev_x, prev_y = self.px, self.py

        # Move directly to cursor position
        self.px = target_x
        self.py = target_y

        # Resolve collisions — iterate a few times for stability at corners
        collision_occurred = False
        for _ in range(3):
            hit = self._resolve_collisions()
            if hit:
                collision_occurred = True

        if collision_occurred:
            self.wall_hit_count += 1

    def _resolve_collisions(self) -> bool:
        """
        Check player circle against all wall rectangles.
        Push the player out of any penetrated wall.

        Returns True if at least one collision was resolved.
        """
        hit = False
        r = self.radius

        for wall in self.level.walls:
            wx, wy, ww, wh = wall

            # Find nearest point on rect to player centre
            nearest_x = max(wx, min(self.px, wx + ww))
            nearest_y = max(wy, min(self.py, wy + wh))

            dist_x = self.px - nearest_x
            dist_y = self.py - nearest_y
            dist_sq = dist_x**2 + dist_y**2

            if dist_sq < r * r:
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0
                hit = True

                if dist < 1e-4:
                    # Player centre is exactly on wall edge — push upward as default
                    self.py -= r
                else:
                    # Push player out along the collision normal
                    penetration = r - dist
                    nx = dist_x / dist
                    ny = dist_y / dist
                    self.px += nx * (penetration + 0.5)
                    self.py += ny * (penetration + 0.5)

        # Clamp to canvas bounds
        r = self.radius
        self.px = max(r, min(self.px, config.CANVAS_WIDTH  - r))
        self.py = max(r, min(self.py, config.CANVAS_HEIGHT - r))

        return hit

    # ─────────────────────────────────────────────────────────────────────────
    #  State transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _begin_running(self) -> None:
        self.state = State.RUNNING
        self._session_start_time = time.perf_counter()
        if self.metrics:
            self.metrics.start_recording()
        log.info("Session RUNNING")

    def _end_session(self, won: bool) -> None:
        self._session_end_time = time.perf_counter()
        self.state = State.WIN if won else State.TIMEOUT
        if self.metrics:
            self.metrics.stop_recording()
            self.final_metrics = self.metrics.compute()
        status = "WIN" if won else "TIMEOUT"
        log.info(f"Session {status} | time={self.elapsed_time:.1f}s | "
                 f"wall_hits={self.wall_hit_count}")

    def toggle_pause(self) -> None:
        if self.state == State.RUNNING:
            self.state = State.PAUSED
        elif self.state == State.PAUSED:
            self.state = State.RUNNING

    # ─────────────────────────────────────────────────────────────────────────
    #  Trail management
    # ─────────────────────────────────────────────────────────────────────────

    def _record_trail(self) -> None:
        pos = (int(self.px), int(self.py))
        if not self.trail or self.trail[-1] != pos:
            self.trail.append(pos)
        if len(self.trail) > TRAIL_MAX_LENGTH:
            self.trail.pop(0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Convenience properties (used by Renderer / HUD)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def player_pos(self) -> Tuple[int, int]:
        return int(self.px), int(self.py)

    @property
    def elapsed_time(self) -> float:
        """Seconds since session started (0 while WAITING)."""
        if self._session_start_time is None:
            return 0.0
        if self._session_end_time is not None:
            return self._session_end_time - self._session_start_time
        return time.perf_counter() - self._session_start_time

    @property
    def is_active(self) -> bool:
        return self.state == State.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.state in (State.WIN, State.TIMEOUT)

    @property
    def distance_to_end(self) -> float:
        return math.dist(self.player_pos, self.level.end)
