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
from game.collision import resolve_against_walls, continuous_move_and_resolve

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
      - Anti-tunneling Continuous Collision Detection (CCD) for high-speed hand gestures.
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
        self.is_colliding: bool = False
        self.collision_flash_timer: float = 0.0

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
        Uses Continuous Collision Detection (CCD) to prevent tunneling through
        thin walls even when hand moves rapidly between frames.

        Args:
            target_x, target_y: Desired target position from hand tracker / mouse.
            walls:              List of obstacle rectangles and polygons.
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

        # 2. Continuous Collision Detection (CCD)
        new_px, new_py, hit = continuous_move_and_resolve(
            self.px, self.py, tx, ty, float(self.radius), walls, max_step_size=4.0
        )
        self.px = float(new_px)
        self.py = float(new_py)

        # 3. Canvas bounds clamp (continuous floating bounds)
        r = float(self.radius)
        self.px = max(r, min(self.px, float(config.CANVAS_WIDTH) - r))
        self.py = max(r, min(self.py, float(config.CANVAS_HEIGHT) - r))

        if hit:
            self.wall_hit_count += 1
            self.is_colliding = True
            self.collision_flash_timer = 0.25
        else:
            self.is_colliding = False

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

    def update_timers(self, dt: float) -> None:
        """Update animation and collision visual feedback timers."""
        if self.collision_flash_timer > 0.0:
            self.collision_flash_timer = max(0.0, self.collision_flash_timer - dt)

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset player position and clear trajectory for a new session."""
        self.px = float(start_x)
        self.py = float(start_y)
        self.trail.clear()
        self.wall_hit_count = 0
        self.is_colliding = False
        self.collision_flash_timer = 0.0

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

