"""
game/player.py — Player Entity

Manages:
  - Continuous pixel-space position (px, py)
  - Movement toward a cursor target
  - Wall collision resolution (via game.collision)
  - Path trail recording

Dependency chain:
    game.player → game.collision, config
"""

from __future__ import annotations
import math
import logging
from typing import List, Optional, Tuple

import config
from game.collision import resolve_against_walls

log = logging.getLogger(__name__)


class Player:
    """
    Represents the player cursor in the continuous 2D game world.

    Design constraints:
      - Continuous floating-point coordinates (px, py) — NO grid or tile snapping.
      - Direct mapping from hand/cursor position to game space.
      - Noise filtering via dead-zone threshold (MIN_MOVE_PX).
      - Continuous collision resolution sliding along walls without block snapping.
      - Continuous trajectory point storage for clinical kinematics analysis.
    """

    def __init__(
        self,
        start_x: float,
        start_y: float,
        radius:  int = config.PLAYER_RADIUS,
    ) -> None:
        self.px:     float = float(start_x)
        self.py:     float = float(start_y)
        self.radius: int   = int(radius)

        # Continuous trajectory points [(x0, y0), (x1, y1), ...]
        self.trail: List[Tuple[float, float]] = []
        self.wall_hit_count: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def move_to_cursor(
        self,
        target_x: float,
        target_y: float,
        walls:    list,
        min_move_threshold: float = config.MIN_MOVE_PX,
    ) -> bool:
        """
        Move player toward (target_x, target_y) in continuous 2D space.

        Filters out micro-movements smaller than min_move_threshold to suppress
        camera noise / tremor jitter when the hand is stationary.

        Args:
            target_x, target_y: Desired target position from hand tracker / mouse.
            walls:              List of (wx, wy, ww, wh) obstacle rectangles.
            min_move_threshold: Dead-zone distance in pixels (defaults to config.MIN_MOVE_PX).

        Returns:
            bool: True if movement occurred; False if movement was suppressed by dead-zone.
        """
        tx = float(target_x)
        ty = float(target_y)

        # 1. Noise dead-zone check
        displacement = math.hypot(tx - self.px, ty - self.py)
        if displacement < min_move_threshold:
            return False

        # 2. Continuous position update (pure 2D floats, no grid snapping)
        self.px = tx
        self.py = ty

        # 3. Continuous collision resolution with walls (slides smoothly)
        new_px, new_py, hit = resolve_against_walls(
            self.px, self.py, float(self.radius), walls, iterations=3
        )
        self.px = float(new_px)
        self.py = float(new_py)

        # 4. Canvas bounds clamp (continuous floating bounds)
        r = float(self.radius)
        self.px = max(r, min(self.px, float(config.CANVAS_WIDTH) - r))
        self.py = max(r, min(self.py, float(config.CANVAS_HEIGHT) - r))

        if hit:
            self.wall_hit_count += 1

        return True

    def record_trail(self, min_dist: float = config.MIN_TRAIL_DIST_PX) -> None:
        """
        Append current continuous position to the trajectory trail.
        Ignores points closer than min_dist to avoid storing static noise.
        """
        pt = (self.px, self.py)
        if not self.trail:
            self.trail.append(pt)
        else:
            last_pt = self.trail[-1]
            if math.hypot(self.px - last_pt[0], self.py - last_pt[1]) >= min_dist:
                self.trail.append(pt)

        if len(self.trail) > config.TRAIL_MAX_LENGTH:
            self.trail.pop(0)

    def set_radius(self, radius: int) -> None:
        """Update player cursor radius (e.g. for difficulty changes)."""
        self.radius = max(2, int(radius))

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset player position and clear trajectory for a new session."""
        self.px = float(start_x)
        self.py = float(start_y)
        self.trail.clear()
        self.wall_hit_count = 0

    def clear_trajectory(self) -> None:
        """Explicitly clear recorded trajectory points."""
        self.trail.clear()

    # ─────────────────────────────────────────────────────────────────────────
    #  Properties (read by GameEngine, MetricsCollector, and Renderer)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[int, int]:
        """Integer pixel position for rasterization operations."""
        return int(round(self.px)), int(round(self.py))

    @property
    def position_f(self) -> Tuple[float, float]:
        """Continuous sub-pixel floating-point position."""
        return self.px, self.py

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Return a copy of the stored continuous trajectory points."""
        return list(self.trail)

