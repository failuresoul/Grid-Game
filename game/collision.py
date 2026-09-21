"""
game/collision.py — Circle-vs-Rectangle Collision Resolution

Pure geometry, zero project imports.  Usable standalone or in tests.

Provides:
  resolve_circle_rect(px, py, r, wx, wy, ww, wh)
      Push a circle out of an axis-aligned rectangle if penetrating.
      Returns (new_px, new_py, hit: bool).
"""

from __future__ import annotations
import math
from typing import Tuple


def resolve_circle_rect(
    px: float,
    py: float,
    r: float,
    wx: float,
    wy: float,
    ww: float,
    wh: float,
) -> Tuple[float, float, bool]:
    """
    Detect and resolve a circle-vs-AABB (axis-aligned bounding box) collision.

    The circle centre is at (px, py) with radius r.
    The rectangle has its top-left corner at (wx, wy) with size (ww × wh).

    If the circle overlaps the rectangle, the circle is pushed out along
    the shortest penetration vector (collision normal).

    Args:
        px, py: Circle centre (float pixels).
        r:      Circle radius.
        wx, wy: Rectangle top-left corner.
        ww, wh: Rectangle width and height.

    Returns:
        (new_px, new_py, hit)
        new_px, new_py: Adjusted circle centre (unchanged if no collision).
        hit:            True if a collision was detected and resolved.
    """
    # Find the nearest point on the rectangle surface to the circle centre
    nearest_x = max(wx, min(px, wx + ww))
    nearest_y = max(wy, min(py, wy + wh))

    dist_x = px - nearest_x
    dist_y = py - nearest_y
    dist_sq = dist_x * dist_x + dist_y * dist_y

    if dist_sq >= r * r:
        return px, py, False   # no overlap

    dist = math.sqrt(dist_sq)

    if dist < 1e-6:
        # Circle centre is exactly on (or inside) the wall surface.
        # Degenerate case: push straight up as a safe default.
        return px, py - r, True

    # Push the circle out along the collision normal by the penetration depth
    penetration = r - dist
    nx = dist_x / dist
    ny = dist_y / dist
    push = penetration + 0.5    # small bias to prevent re-tunnelling
    return px + nx * push, py + ny * push, True


def resolve_against_walls(
    px: float,
    py: float,
    r: float,
    walls: list,
    iterations: int = 3,
) -> Tuple[float, float, bool]:
    """
    Resolve a circle against a list of rectangles, iterating for stability
    at corners where two walls meet.

    Args:
        px, py:     Initial circle centre.
        r:          Circle radius.
        walls:      List of (wx, wy, ww, wh) tuples.
        iterations: Number of solver passes (3 is enough for most cases).

    Returns:
        (new_px, new_py, any_hit)
    """
    any_hit = False
    for _ in range(iterations):
        for wall in walls:
            wx, wy, ww, wh = wall
            px, py, hit = resolve_circle_rect(px, py, r, wx, wy, ww, wh)
            if hit:
                any_hit = True
    return px, py, any_hit
