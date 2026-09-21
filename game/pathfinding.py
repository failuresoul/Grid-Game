"""
game/pathfinding.py — Robust Continuous 2D Geometric Pathfinding & Minimum Path Calculation

Computes the true shortest collision-free path between START and END across arbitrary
2D continuous geometric mazes, taking the circular player cursor's radius and all
obstacles (rectangles and arbitrary polygons) into account.

Key Features:
  - Exact Euclidean distance transform (cv2.distanceTransform) for continuous obstacle clearance.
  - Physical clearance constraint: clearance >= player_radius + margin.
  - A* search on continuous traversable grid with diagonal movement.
  - Continuous line-of-sight shortcutting / string-pulling with exact circle clearance checks.
  - Returns both the scalar minimum_path_distance (px) and the sequence of 2D waypoints.
"""

from __future__ import annotations
import heapq
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import config

Point = Tuple[float, float]


def calculate_minimum_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    obstacles: Sequence[Any],
    player_radius: float,
    width: int = config.CANVAS_WIDTH,
    height: int = config.CANVAS_HEIGHT,
    margin: float = 1.0,
    grid_step: int = 6,
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Compute the shortest collision-free route from start to end considering player radius.

    Args:
        start:         (sx, sy) START coordinates in continuous pixels.
        end:           (ex, ey) END coordinates in continuous pixels.
        obstacles:     Sequence of RectObstacle, PolygonObstacle, or bounding tuples.
        player_radius: Physical radius of the player cursor (px).
        width:         Canvas width (px).
        height:        Canvas height (px).
        margin:        Safety clearance margin added to player_radius (px).
        grid_step:     A* discretization step (px). Default 6px for high fidelity.

    Returns:
        (minimum_path_distance, shortest_path_points)
        If no path exists (blocked maze), returns (float('inf'), []).
    """
    W, H = int(width), int(height)
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    effective_radius = float(player_radius) + float(margin)

    # ── 1. Rasterize Geometric Obstacles onto Binary Mask ─────────────────────
    mask = np.zeros((H, W), dtype=np.uint8)
    for obs in obstacles:
        if getattr(obs, "is_polygon", False) and hasattr(obs, "points"):
            pts = np.array(obs.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        elif hasattr(obs, "as_rect"):
            x, y, w, h = obs.as_rect()
            cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
        elif isinstance(obs, (tuple, list)):
            if len(obs) == 4:
                x, y, w, h = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            elif len(obs) >= 3:
                pts = np.array(obs, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

    # ── 2. Boundary Screen Border Clearance ───────────────────────────────────
    free_space = cv2.bitwise_not(mask)
    border = max(1, int(math.ceil(player_radius)))
    free_space[:border, :] = 0
    free_space[-border:, :] = 0
    free_space[:, :border] = 0
    free_space[:, -border:] = 0

    # ── 3. Exact Euclidean Distance Transform ─────────────────────────────────
    # dist_map[y, x] gives continuous Euclidean distance to nearest wall/obstacle pixel
    dist_map = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

    # Quick line-of-sight check: if direct straight line has full clearance, return straight line
    def check_line_of_sight(p1: Point, p2: Point, test_radius: float) -> bool:
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1.0:
            return True
        sub_steps = max(2, int(dist // 3))
        dx = (p2[0] - p1[0]) / sub_steps
        dy = (p2[1] - p1[1]) / sub_steps
        for s in range(sub_steps + 1):
            px = int(round(p1[0] + dx * s))
            py = int(round(p1[1] + dy * s))
            if px < 0 or px >= W or py < 0 or py >= H:
                return False
            if dist_map[py, px] < test_radius:
                return False
        return True

    if check_line_of_sight((sx, sy), (ex, ey), effective_radius):
        straight_dist = math.hypot(ex - sx, ey - sy)
        return float(round(straight_dist, 2)), [(sx, sy), (ex, ey)]

    # ── 4. Subsampled Traversable Clearance Grid ──────────────────────────────
    step = max(2, int(grid_step))
    grid_h = H // step
    grid_w = W // step
    ys = np.arange(grid_h) * step + step // 2
    xs = np.arange(grid_w) * step + step // 2

    sampled_dist = dist_map[ys[:, None], xs[None, :]]
    passable = sampled_dist >= effective_radius

    sc = max(0, min(grid_w - 1, int(sx // step)))
    sr = max(0, min(grid_h - 1, int(sy // step)))
    ec = max(0, min(grid_w - 1, int(ex // step)))
    er = max(0, min(grid_h - 1, int(ey // step)))

    # Ensure start and end cells are marked passable if start/end positions are valid
    if dist_map[int(round(sy)), int(round(sx))] >= (player_radius * 0.8):
        passable[sr, sc] = True
    if dist_map[int(round(ey)), int(round(ex))] >= (player_radius * 0.8):
        passable[er, ec] = True

    # If start or end cell itself is not passable, find closest passable neighbor
    def find_closest_passable(target_r: int, target_c: int) -> Tuple[int, int]:
        if passable[target_r, target_c]:
            return target_r, target_c
        best = (target_r, target_c)
        best_d = float("inf")
        search_rad = 5
        for dr in range(-search_rad, search_rad + 1):
            for dc in range(-search_rad, search_rad + 1):
                nr, nc = target_r + dr, target_c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w and passable[nr, nc]:
                    d = math.hypot(dr, dc)
                    if d < best_d:
                        best_d = d
                        best = (nr, nc)
        return best

    sr, sc = find_closest_passable(sr, sc)
    er, ec = find_closest_passable(er, ec)

    # ── 5. A* Search on Clearance Grid ────────────────────────────────────────
    def heuristic(r: int, c: int) -> float:
        return math.hypot((r - er) * step, (c - ec) * step)

    SQRT2 = 1.41421356
    moves = [
        (-1, 0, float(step)),
        (1, 0, float(step)),
        (0, -1, float(step)),
        (0, 1, float(step)),
        (-1, -1, step * SQRT2),
        (-1, 1, step * SQRT2),
        (1, -1, step * SQRT2),
        (1, 1, step * SQRT2),
    ]

    frontier: List[Tuple[float, float, int, int]] = []
    heapq.heappush(frontier, (heuristic(sr, sc), 0.0, sr, sc))
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    cost_so_far: dict[Tuple[int, int], float] = {(sr, sc): 0.0}

    found = False
    while frontier:
        _, g, cr, cc = heapq.heappop(frontier)
        if cr == er and cc == ec:
            found = True
            break

        for dr, dc, dcost in moves:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w and passable[nr, nc]:
                # Prevent cutting corners between two touching diagonal obstacles
                if dr != 0 and dc != 0:
                    if not (passable[cr, nc] and passable[nr, cc]):
                        continue
                new_cost = g + dcost
                if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                    cost_so_far[(nr, nc)] = new_cost
                    priority = new_cost + heuristic(nr, nc)
                    heapq.heappush(frontier, (priority, new_cost, nr, nc))
                    came_from[(nr, nc)] = (cr, cc)

    if not found:
        # Maze is physically impassable for this player radius
        return float("inf"), []

    # ── 6. Reconstruct Dense Grid Path ────────────────────────────────────────
    curr = (er, ec)
    raw_path: List[Point] = []
    while curr in came_from:
        raw_path.append((float(curr[1] * step + step / 2.0), float(curr[0] * step + step / 2.0)))
        curr = came_from[curr]
    raw_path.append((sx, sy))
    raw_path.reverse()
    raw_path.append((ex, ey))

    # ── 7. Continuous Line-of-Sight Shortcut Smoothing ────────────────────────
    # Greedy string-pulling: connect as far forward as possible without obstacle collision
    smoothed: List[Point] = [raw_path[0]]
    curr_idx = 0
    n = len(raw_path)

    while curr_idx < n - 1:
        furthest = curr_idx + 1
        for nxt in range(n - 1, curr_idx, -1):
            if check_line_of_sight(raw_path[curr_idx], raw_path[nxt], effective_radius):
                furthest = nxt
                break
        smoothed.append(raw_path[furthest])
        curr_idx = furthest

    # ── 8. Calculate Total Euclidean Path Distance ────────────────────────────
    total_dist = 0.0
    for i in range(1, len(smoothed)):
        total_dist += math.hypot(
            smoothed[i][0] - smoothed[i - 1][0],
            smoothed[i][1] - smoothed[i - 1][1],
        )

    return float(round(total_dist, 2)), smoothed


def minimum_path_distance(
    start_or_level: Any,
    end: Optional[Tuple[float, float]] = None,
    obstacles: Optional[Sequence[Any]] = None,
    player_radius: Optional[float] = None,
    width: int = config.CANVAS_WIDTH,
    height: int = config.CANVAS_HEIGHT,
    margin: float = 1.0,
) -> float:
    """
    Convenience function to compute and return purely the minimum_path_distance.

    Can be called either as:
        minimum_path_distance(level, player_radius=14.0)
    or
        minimum_path_distance(start, end, obstacles, player_radius=14.0)
    """
    if hasattr(start_or_level, "walls") and hasattr(start_or_level, "start") and hasattr(start_or_level, "end"):
        lvl = start_or_level
        pr = player_radius if player_radius is not None else float(getattr(config, "PLAYER_RADIUS", 16))
        dist, _ = calculate_minimum_path(
            start=lvl.start,
            end=lvl.end,
            obstacles=lvl.walls,
            player_radius=pr,
            width=getattr(lvl, "width", width),
            height=getattr(lvl, "height", height),
            margin=margin,
        )
        return dist
    else:
        if end is None or obstacles is None or player_radius is None:
            raise ValueError("minimum_path_distance requires start, end, obstacles, and player_radius")
        dist, _ = calculate_minimum_path(
            start=start_or_level,
            end=end,
            obstacles=obstacles,
            player_radius=player_radius,
            width=width,
            height=height,
            margin=margin,
        )
        return dist


def solve_level_path(
    level: Any,
    player_radius: float,
    margin: float = 1.0,
) -> Tuple[float, List[Point]]:
    """
    Calculate the minimum path for a Level instance and store the results
    in level.min_path_distance and level.optimal_waypoints.

    Returns:
        (minimum_path_distance, optimal_waypoints)
    """
    dist, waypoints = calculate_minimum_path(
        start=level.start,
        end=level.end,
        obstacles=level.walls,
        player_radius=player_radius,
        width=level.width,
        height=level.height,
        margin=margin,
    )
    if waypoints:
        level.min_path_distance = dist
        level.optimal_waypoints = waypoints
    return dist, waypoints
