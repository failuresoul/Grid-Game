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
        - Ignores invalid points (non-numeric, NaN, Inf, malformed tuples).
        - Handles missing hand frames (sentinel None values do NOT accumulate
          a false teleport distance across the missing tracking interval).
        - Avoids double-counting (consecutive identical points or zero displacement).
        - Keeps coordinate system consistent with continuous 2D game pixels.

    Args:
        points_or_xs: Sequence of 2D points [(x0, y0), (x1, y1), ...] or 1D array of x-coordinates.
        ys:           Optional sequence of y-coordinates if points_or_xs is xs.
        handle_missing: If True, missing hand frames (represented by None)
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
        # 1. Handle missing hand frames (None sentinel)
        if item is None:
            if handle_missing:
                last_valid_point = None
            continue

        # 2. Ignore invalid points (malformed types, wrong length)
        if not isinstance(item, (tuple, list, np.ndarray)) or len(item) < 2:
            continue

        try:
            x_val = float(item[0])
            y_val = float(item[1])
        except (ValueError, TypeError):
            continue

        # Ignore invalid points (NaN, Inf)
        if math.isnan(x_val) or math.isnan(y_val) or math.isinf(x_val) or math.isinf(y_val):
            continue

        curr_point = (x_val, y_val)

        # 3. Calculate consecutive distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        if last_valid_point is not None:
            dx = curr_point[0] - last_valid_point[0]
            dy = curr_point[1] - last_valid_point[1]
            seg_dist = math.hypot(dx, dy)

            # Avoid double-counting static / duplicate consecutive points
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
    actual: Any = None,
    minimum: Optional[float] = None,
    start: Optional[Tuple[float, float]] = None,
    end: Optional[Tuple[float, float]] = None,
    min_path: Optional[float] = None,
    actual_distance: Optional[float] = None,
    minimum_distance: Optional[float] = None,
    min_movement_px: float = 5.0,
) -> float:
    """
    Calculate Path Efficiency as a percentage [0.0% - 100.0%].

    Formula:
        Path Efficiency = Minimum Path Distance / Actual Distance * 100

    Example:
        Minimum = 300, Actual = 420
        Efficiency = 300 / 420 * 100 = 71.43% -> 71.4%

    Requirements met:
        - Clamps sensible values between 0.0% and 100.0%.
        - Avoids division by zero (returns 0.0%).
        - Calculates only after meaningful movement occurs (actual_distance >= min_movement_px).
        - Returns float percentage rounded to one decimal place.

    Supports both:
        path_efficiency(actual_distance=420.0, minimum_distance=300.0)
        path_efficiency(actual=420.0, minimum=300.0)
    and trajectory array form:
        path_efficiency(xs, ys, start, end, min_path=300.0)
    """
    # 1. Resolve actual distance
    act_val: Optional[float] = None
    if actual_distance is not None:
        act_val = float(actual_distance)
    elif isinstance(actual, (int, float)):
        act_val = float(actual)
    elif actual is not None and minimum is not None and isinstance(minimum, (np.ndarray, list, tuple)):
        # Trajectory arrays passed as positional args: (xs, ys)
        act_val = float(path_length(actual, minimum))

    # 2. Resolve minimum distance
    min_val: Optional[float] = None
    if minimum_distance is not None:
        min_val = float(minimum_distance)
    elif min_path is not None:
        min_val = float(min_path)
    elif isinstance(minimum, (int, float)):
        min_val = float(minimum)
    elif start is not None and end is not None:
        min_val = float(ideal_distance(start, end))

    if act_val is None or min_val is None:
        return 0.0

    # 3. Calculate only after meaningful movement occurs
    if act_val < min_movement_px:
        return 0.0

    # 4. Avoid division by zero
    if act_val <= 0.0:
        return 0.0

    # 5. Formula: Minimum / Actual * 100
    raw_efficiency = (min_val / act_val) * 100.0

    # 6. Clamp to sensible values: [0.0, 100.0]
    clamped_efficiency = max(0.0, min(100.0, raw_efficiency))

    # 7. Show percentage with one decimal place
    return float(round(clamped_efficiency, 1))


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
