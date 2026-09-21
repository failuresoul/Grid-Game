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
import logging
from typing import List, Optional, Tuple

import config
from game.collision import resolve_against_walls

log = logging.getLogger(__name__)

# Maximum number of trail points kept in memory
TRAIL_MAX_LENGTH = 400


class Player:
    """
    Represents the player cursor in the game world.

    The player position is a continuous floating-point pixel coordinate
    (NOT snapped to a grid).  Each frame the player teleports to the
    hand-cursor target and is then pushed out of any walls.
    """

    def __init__(
        self,
        start_x: float,
        start_y: float,
        radius:  int,
    ) -> None:
        self.px:     float = start_x
        self.py:     float = start_y
        self.radius: int   = radius

        self.trail: List[Tuple[int, int]] = []
        self.wall_hit_count: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def move_to_cursor(
        self,
        target_x: float,
        target_y: float,
        walls:    list,
    ) -> None:
        """
        Teleport the player to (target_x, target_y) then resolve wall collisions.

        Args:
            target_x, target_y: Desired position (from hand tracker).
            walls:              List of (wx, wy, ww, wh) obstacle rectangles.
        """
        self.px = target_x
        self.py = target_y

        new_px, new_py, hit = resolve_against_walls(
            self.px, self.py, float(self.radius), walls, iterations=3
        )
        self.px = new_px
        self.py = new_py

        # Canvas bounds clamp
        r = self.radius
        self.px = max(r, min(self.px, config.CANVAS_WIDTH  - r))
        self.py = max(r, min(self.py, config.CANVAS_HEIGHT - r))

        if hit:
            self.wall_hit_count += 1

    def record_trail(self) -> None:
        """Append current position to the trail (deduplicates adjacent points)."""
        pos = (int(self.px), int(self.py))
        if not self.trail or self.trail[-1] != pos:
            self.trail.append(pos)
        if len(self.trail) > TRAIL_MAX_LENGTH:
            self.trail.pop(0)

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset position and trail for a new session."""
        self.px = start_x
        self.py = start_y
        self.trail.clear()
        self.wall_hit_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Properties (read by GameEngine and Renderer)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[int, int]:
        return int(self.px), int(self.py)

    @property
    def position_f(self) -> Tuple[float, float]:
        return self.px, self.py
