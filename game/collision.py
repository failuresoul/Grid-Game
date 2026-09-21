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
        # Circle centre is inside the rectangle. Find the closest edge to push out.
        d_left   = px - wx
        d_right  = (wx + ww) - px
        d_top    = py - wy
        d_bottom = (wy + wh) - py
        min_d = min(d_left, d_right, d_top, d_bottom)
        push_bias = 0.5
        if min_d == d_left:
            return wx - r - push_bias, py, True
        elif min_d == d_right:
            return wx + ww + r + push_bias, py, True
        elif min_d == d_top:
            return px, wy - r - push_bias, True
        else:
            return px, wy + wh + r + push_bias, True

    # Push the circle out along the collision normal by the penetration depth
    penetration = r - dist
    nx = dist_x / dist
    ny = dist_y / dist
    push = penetration + 0.5    # small bias to prevent re-tunnelling
    return px + nx * push, py + ny * push, True


def resolve_circle_segment(
    px: float,
    py: float,
    r: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float, bool]:
    """
    Detect and resolve collision between a circle (px, py, r) and a 2D line segment (ax, ay)-(bx, by).
    Pushes the circle away from the segment if penetrating.

    Returns:
        (new_px, new_py, hit)
    """
    abx = bx - ax
    aby = by - ay
    mag_sq = abx * abx + aby * aby

    if mag_sq < 1e-12:
        # Degenerate segment: treat as point (ax, ay)
        dist_x = px - ax
        dist_y = py - ay
        dist_sq = dist_x * dist_x + dist_y * dist_y
        if dist_sq >= r * r:
            return px, py, False
        dist = math.sqrt(dist_sq)
        if dist < 1e-6:
            return px + r + 0.5, py, True
        push = (r - dist) + 0.5
        return px + (dist_x / dist) * push, py + (dist_y / dist) * push, True

    # Project point onto segment clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / mag_sq))
    qx = ax + t * abx
    qy = ay + t * aby

    dist_x = px - qx
    dist_y = py - qy
    dist_sq = dist_x * dist_x + dist_y * dist_y

    if dist_sq >= r * r:
        return px, py, False

    dist = math.sqrt(dist_sq)
    if dist < 1e-6:
        # Circle center directly on segment line: push along perpendicular normal
        seg_len = math.sqrt(mag_sq)
        nx = -aby / seg_len
        ny = abx / seg_len
        return px + nx * (r + 0.5), py + ny * (r + 0.5), True

    penetration = r - dist
    nx = dist_x / dist
    ny = dist_y / dist
    push = penetration + 0.5
    return px + nx * push, py + ny * push, True


def point_in_polygon(px: float, py: float, points: list) -> bool:
    """
    Ray-casting algorithm to test if point (px, py) is strictly inside a polygon.
    """
    n = len(points)
    if n < 3:
        return False
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if min(p1y, p2y) < py <= max(p1y, p2y):
            if px <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or px <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def resolve_circle_polygon(
    px: float,
    py: float,
    r: float,
    points: list,
) -> Tuple[float, float, bool]:
    """
    Detect and resolve collision between a circle (px, py, r) and a 2D polygon.

    Args:
        px, py:  Circle centre.
        r:       Circle radius.
        points:  List of (x, y) polygon vertices in cyclic order.

    Returns:
        (new_px, new_py, hit)
    """
    n = len(points)
    if n < 3:
        return px, py, False

    inside = point_in_polygon(px, py, points)
    any_hit = False

    if inside:
        # Circle centre is inside the polygon: find the closest edge and eject the circle
        best_dist = float("inf")
        best_qx, best_qy = px, py
        best_nx, best_ny = 0.0, 0.0

        for i in range(n):
            ax, ay = points[i]
            bx, by = points[(i + 1) % n]
            abx, aby = bx - ax, by - ay
            mag_sq = abx * abx + aby * aby
            if mag_sq < 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / mag_sq))
            qx = ax + t * abx
            qy = ay + t * aby
            dx, dy = px - qx, py - qy
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist = d
                best_qx, best_qy = qx, qy
                seg_len = math.sqrt(mag_sq)
                # Outward-pointing normal approximation
                best_nx = -aby / seg_len
                best_ny = abx / seg_len

        # Push circle out to outside edge + radius
        push_bias = 0.5
        # Ensure direction points out
        test_x = best_qx + best_nx * 2.0
        test_y = best_qy + best_ny * 2.0
        if point_in_polygon(test_x, test_y, points):
            best_nx = -best_nx
            best_ny = -best_ny

        return best_qx + best_nx * (r + push_bias), best_qy + best_ny * (r + push_bias), True

    # Circle centre is outside: check penetration with each polygon edge segment
    cur_px, cur_py = px, py
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        cur_px, cur_py, hit = resolve_circle_segment(cur_px, cur_py, r, ax, ay, bx, by)
        if hit:
            any_hit = True

    return cur_px, cur_py, any_hit


def resolve_against_obstacles(
    px: float,
    py: float,
    r: float,
    obstacles: list,
    iterations: int = 3,
) -> Tuple[float, float, bool]:
    """
    Resolve a circle against a list of geometric obstacles (rectangles and polygons).
    Supports:
      - (x, y, w, h) tuples
      - Objects with .is_polygon and .points (PolygonObstacle)
      - Objects with .as_rect() or .x, .y, .w, .h (RectObstacle)

    Iterates for stability at multi-obstacle corners.
    """
    any_hit = False
    for _ in range(iterations):
        for obs in obstacles:
            if hasattr(obs, "is_polygon") and obs.is_polygon:
                px, py, hit = resolve_circle_polygon(px, py, r, obs.points)
            elif hasattr(obs, "as_rect"):
                wx, wy, ww, wh = obs.as_rect()
                px, py, hit = resolve_circle_rect(px, py, r, wx, wy, ww, wh)
            elif isinstance(obs, (tuple, list)):
                if len(obs) == 4:
                    wx, wy, ww, wh = obs
                    px, py, hit = resolve_circle_rect(px, py, r, wx, wy, ww, wh)
                elif len(obs) >= 3 and isinstance(obs[0], (tuple, list)):
                    # Passed raw list of polygon points
                    px, py, hit = resolve_circle_polygon(px, py, r, obs)
                else:
                    continue
            else:
                # Duck-type x, y, w, h attributes
                if hasattr(obs, "x") and hasattr(obs, "w"):
                    px, py, hit = resolve_circle_rect(px, py, r, obs.x, obs.y, obs.w, obs.h)
                else:
                    continue

            if hit:
                any_hit = True

    return px, py, any_hit


def resolve_against_walls(
    px: float,
    py: float,
    r: float,
    walls: list,
    iterations: int = 3,
) -> Tuple[float, float, bool]:
    """
    Resolve a circle against obstacles (backward-compatible wrapper around resolve_against_obstacles).
    """
    return resolve_against_obstacles(px, py, r, walls, iterations=iterations)

