"""
metrics/distance.py — Spatial Metrics (Path Length, Efficiency, ROM)

Pure numpy/math — no project imports.  All functions operate on
position arrays and return scalar floats.

Functions:
    path_length(xs, ys)             → total arc length of the trajectory
    ideal_distance(start, end)      → straight-line distance between endpoints
    path_efficiency(xs, ys, start, end) → ideal / actual  (0–1, higher = better)
    rom_dimensions(xs, ys)          → (width_px, height_px) bounding box
    segment_speeds(xs, ys, ts)      → per-segment speed array (px/s)
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


def actual_distance(
    points_or_xs: Any,
    ys: Optional[Sequence[float]] = None,
    handle_missing: bool = True,
    min_step_px: float = 1e-6,
) -> float:
    """
    Calculate actual continuous trajectory distance from smoothed hand/player movement.

    For consecutive trajectory points:
        P1 = (x1, y1)
        P2 = (x2, y2)
        distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Total distance:
        Sum of all consecutive movement distances in pixels/game units.

    Requirements met:
        - Ignores invalid points (None, non-numeric, NaN, Inf, malformed tuples).
        - Handles missing hand frames (sentinel None / invalid values do NOT accumulate
          a false teleport distance across the missing tracking interval).
        - Avoids double-counting (consecutive identical points or zero displacement).
        - Keeps coordinate system consistent with continuous 2D game pixels.

    Args:
        points_or_xs: Sequence of 2D points [(x0, y0), (x1, y1), ...] or 1D array of x-coordinates.
        ys:           Optional sequence of y-coordinates if points_or_xs is xs.
        handle_missing: If True, missing hand frames (represented by None or invalid values)
                        break the consecutive chain so no leap is added across the missing interval.
        min_step_px:  Minimum displacement threshold to avoid adding static jitter / zero.

    Returns:
        float: Cumulative actual trajectory distance in pixels.
    """
    if points_or_xs is None:
        return 0.0

    if ys is not None:
        raw_points = list(zip(points_or_xs, ys))
    else:
        raw_points = points_or_xs

    total_distance = 0.0
    last_valid_point: Optional[Tuple[float, float]] = None

    for item in raw_points:
        # Handle missing hand frames / None values
        if item is None:
            if handle_missing:
                last_valid_point = None
            continue

        if not isinstance(item, (tuple, list, np.ndarray)) or len(item) < 2:
            if handle_missing:
                last_valid_point = None
            continue

        try:
            x_val = float(item[0])
            y_val = float(item[1])
        except (ValueError, TypeError):
            if handle_missing:
                last_valid_point = None
            continue

        # Ignore invalid points (NaN, Inf)
        if math.isnan(x_val) or math.isnan(y_val) or math.isinf(x_val) or math.isinf(y_val):
            if handle_missing:
                last_valid_point = None
            continue

        curr_point = (x_val, y_val)

        if last_valid_point is not None:
            dx = curr_point[0] - last_valid_point[0]
            dy = curr_point[1] - last_valid_point[1]
            seg_dist = math.hypot(dx, dy)

            # Avoid double-counting static points
            if seg_dist >= min_step_px:
                total_distance += seg_dist

        last_valid_point = curr_point

    return float(round(total_distance, 4))


def path_length(xs: Any, ys: Optional[Any] = None) -> float:
    """Total arc length / actual trajectory distance in pixels."""
    return actual_distance(xs, ys)


def ideal_distance(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """Euclidean straight-line distance from start to end (pixels)."""
    return math.dist(start, end)


def minimum_path_distance(
    start: Tuple[float, float],
    end: Tuple[float, float],
    obstacles: Sequence[Any],
    player_radius: float,
    width: int = 1000,
    height: int = 700,
) -> float:
    """
    Theoretical shortest collision-free path distance from start to end,
    accounting for obstacles and player cursor radius.
    """
    from game.pathfinding import calculate_minimum_path
    dist, _ = calculate_minimum_path(
        start=start,
        end=end,
        obstacles=obstacles,
        player_radius=player_radius,
        width=width,
        height=height,
    )
    return dist


def path_efficiency(
    xs: np.ndarray,
    ys: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    min_path: Optional[float] = None,
) -> float:
    """
    Path efficiency = shortest_route / actual_path_length.

    If min_path is provided, uses the true collision-free minimum path distance.
    Otherwise falls back to Euclidean straight-line distance.

    Range: (0, 1]. A value of 1.0 means the player took the optimal path.
    """
    actual = path_length(xs, ys)
    if actual < 1e-6:
        return 0.0
    ref_dist = min_path if (min_path is not None and min_path > 0.0) else ideal_distance(start, end)
    return min(ref_dist / actual, 1.0)


def rom_dimensions(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[float, float]:
    """
    Range of Motion estimate as the bounding box of recorded positions.

    Returns:
        (width_px, height_px) — larger values indicate greater arm excursion.
    """
    if len(xs) < 1:
        return 0.0, 0.0
    return float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys))


def segment_speeds(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """
    Per-segment instantaneous speed (px / second).

    Returns an array of length (n-1), or an empty array if fewer than 2 samples.
    """
    if len(xs) < 2:
        return np.array([])
    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts)
    dt = np.where(dt < 1e-6, 1e-6, dt)   # prevent division by zero
    lengths = np.sqrt(dx * dx + dy * dy)
    return lengths / dt
