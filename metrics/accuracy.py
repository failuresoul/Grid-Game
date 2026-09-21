"""
metrics/accuracy.py — Trajectory Accuracy & Path Deviation Kinematics

Evaluates how closely the patient's continuous 2D hand/cursor trajectory follows
the intended reference route (defined by the optimal path waypoints).

Key Principles:
  1. Separation of Concerns:
     - Accuracy: Spatial closeness / adherence to the intended route polyline.
     - Path Efficiency: Ratio of minimum path distance to actual distance.
     - Distance: Actual trajectory length vs. minimum required length.
     - Collision Count: Wall contacts.
     - Time: Total elapsed time vs. time spent outside the route corridor.
  2. "Reaching the END" is NOT synonymous with high accuracy.
     A patient may finish the level while making erratic loops, wandering off-course,
     or following long detours. Accuracy strictly quantifies route fidelity.
  3. No arbitrary ungrounded scoring: All formulas are explicitly defined and documented.

Formulas:
  - Point-to-Segment Orthogonal Distance:
      d(P, [A, B]) = ||P - (A + t*(B - A))|| where t = clamp(((P-A)·(B-A)) / ||B-A||^2, 0, 1)
  - Path Deviation:
      d_i = min_{segment s in route} d(P_i, s)
  - Mean Path Deviation:
      mean_dev = (1 / N) * sum_{i=1}^N d_i  (pixels)
  - Max Path Deviation:
      max_dev  = max_{1 <= i <= N} d_i      (pixels)
  - Root-Mean-Square Deviation (RMSD):
      rms_dev  = sqrt((1 / N) * sum_{i=1}^N d_i^2)  (pixels)
  - Time Outside Intended Route:
      T_outside = sum_{i=1}^{N-1} dt_i * I(d_i > corridor_tolerance)  (seconds)
  - Deviation Events (Wrong Movement Excursions):
      Count of distinct, contiguous temporal episodes where cursor wanders outside
      the intended corridor tolerance for longer than a jitter threshold.
  - Trajectory Accuracy Percentage:
      accuracy_pct = (100 / N) * sum_{i=1}^N exp(-d_i^2 / (2 * sigma^2))
      where sigma = corridor_tolerance (default 40.0 px).
      Bounded strictly within [0.0%, 100.0%], rounded to 1 decimal place.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Orthogonal Euclidean distance from point P(px, py) to line segment AB.

    Clamps projection parameter t to [0.0, 1.0] to respect segment endpoints.
    """
    abx = bx - ax
    aby = by - ay
    seg_len_sq = abx * abx + aby * aby

    if seg_len_sq < 1e-9:
        # Segment is degenerate (a single point)
        return math.hypot(px - ax, py - ay)

    # Orthogonal projection scalar
    apx = px - ax
    apy = py - ay
    t = (apx * abx + apy * aby) / seg_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def point_to_polyline_distance(
    px: float, py: float,
    waypoints: Sequence[Tuple[float, float]],
) -> float:
    """
    Minimum Euclidean distance from point P(px, py) to any segment of a polyline.

    Args:
        px, py: Query point coordinates.
        waypoints: Ordered sequence of 2D vertices [(x0, y0), (x1, y1), ...].

    Returns:
        float: Shortest distance in pixels.
    """
    if not waypoints:
        return 0.0
    if len(waypoints) == 1:
        return math.hypot(px - waypoints[0][0], py - waypoints[0][1])

    min_dist = float("inf")
    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]
        dist = point_to_segment_distance(px, py, ax, ay, bx, by)
        if dist < min_dist:
            min_dist = dist
            if min_dist < 1e-4:
                return 0.0

    return min_dist


def trajectory_deviations(
    xs: Sequence[float],
    ys: Sequence[float],
    waypoints: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """
    Calculate the perpendicular deviation distance d_i for every sample point
    relative to the intended route polyline.

    Returns:
        np.ndarray of shape (N,) containing non-negative deviation distances in pixels.
    """
    n = min(len(xs), len(ys))
    if n == 0 or not waypoints:
        return np.array([], dtype=float)

    deviations = np.empty(n, dtype=float)
    for i in range(n):
        x = float(xs[i])
        y = float(ys[i])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            deviations[i] = 0.0
        else:
            deviations[i] = point_to_polyline_distance(x, y, waypoints)

    return deviations


def path_deviation_summary(
    xs: Sequence[float],
    ys: Sequence[float],
    waypoints: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """
    Compute summary statistics of spatial path deviation from the intended route.

    Returns:
        dict:
          - mean_path_deviation: Mean orthogonal deviation in pixels.
          - max_path_deviation:  Peak orthogonal excursion in pixels.
          - rms_path_deviation:  Root-Mean-Square Deviation (RMSD) in pixels.
    """
    devs = trajectory_deviations(xs, ys, waypoints)
    if len(devs) == 0:
        return {
            "mean_path_deviation": 0.0,
            "max_path_deviation":  0.0,
            "rms_path_deviation":  0.0,
        }

    mean_dev = float(np.mean(devs))
    max_dev  = float(np.max(devs))
    rms_dev  = float(np.sqrt(np.mean(devs ** 2)))

    return {
        "mean_path_deviation": round(mean_dev, 2),
        "max_path_deviation":  round(max_dev, 2),
        "rms_path_deviation":  round(rms_dev, 2),
    }


def route_corridor_metrics(
    xs: Sequence[float],
    ys: Sequence[float],
    ts: Sequence[float],
    waypoints: Sequence[Tuple[float, float]],
    corridor_tolerance_px: float = 40.0,
    min_event_duration_s: float = 0.08,
) -> Dict[str, Any]:
    """
    Quantify excursions beyond the intended corridor tolerance.

    Args:
        xs, ys: Trajectory coordinates.
        ts: Elapsed timestamp array (seconds).
        waypoints: Route waypoints.
        corridor_tolerance_px: Radius around optimal route considered "on-track" (px).
        min_event_duration_s: Minimum duration for an excursion to register as a
                              distinct wrong movement / deviation event (filters frame jitter).

    Returns:
        dict:
          - time_outside_route_s: Total seconds spent outside the corridor.
          - pct_time_outside: Percentage of session time spent outside corridor [0.0, 100.0].
          - deviation_events: Count of discrete wrong-movement / excursion episodes.
          - corridor_adherence_pct: Percentage of time spent inside intended corridor [0.0, 100.0].
    """
    n = min(len(xs), len(ys), len(ts))
    if n < 2 or not waypoints:
        return {
            "time_outside_route_s":  0.0,
            "pct_time_outside":      0.0,
            "deviation_events":      0,
            "corridor_adherence_pct": 100.0,
        }

    devs = trajectory_deviations(xs[:n], ys[:n], waypoints)
    is_outside = devs > corridor_tolerance_px

    time_outside = 0.0
    total_time = max(float(ts[-1] - ts[0]), 1e-6)

    # Deviation events counting (temporal hysteresis / debounce)
    deviation_events = 0
    in_event = False
    event_start_time = 0.0

    for i in range(n - 1):
        dt = max(0.0, float(ts[i + 1] - ts[i]))
        if is_outside[i]:
            time_outside += dt
            if not in_event:
                in_event = True
                event_start_time = float(ts[i])
        else:
            if in_event:
                duration = float(ts[i]) - event_start_time
                if duration >= min_event_duration_s:
                    deviation_events += 1
                in_event = False

    # Check trailing event at end of session
    if in_event:
        duration = float(ts[-1]) - event_start_time
        if duration >= min_event_duration_s:
            deviation_events += 1

    pct_outside = min(100.0, max(0.0, (time_outside / total_time) * 100.0))
    corridor_adherence = max(0.0, min(100.0, 100.0 - pct_outside))

    return {
        "time_outside_route_s":   round(time_outside, 2),
        "pct_time_outside":       round(pct_outside, 1),
        "deviation_events":       int(deviation_events),
        "corridor_adherence_pct": round(corridor_adherence, 1),
    }


def trajectory_accuracy_percentage(
    xs: Sequence[float],
    ys: Sequence[float],
    waypoints: Sequence[Tuple[float, float]],
    corridor_tolerance_px: float = 40.0,
) -> float:
    """
    Calculate Trajectory Accuracy Percentage [0.0% - 100.0%].

    Formula:
        Accuracy = (100 / N) * sum_{i=1}^N exp( - d_i^2 / (2 * sigma^2) )
        where d_i is the orthogonal distance from point (x_i, y_i) to the route polyline,
        and sigma = corridor_tolerance_px.

    Properties:
        - Perfectly following the route line (all d_i = 0): exactly 100.0%.
        - Cursor at corridor boundary (d_i = sigma): contributes exp(-0.5) ≈ 60.7%.
        - Large excursions (d_i >= 3*sigma): contribute exp(-4.5) < 1.1%.
        - Smooth, continuous, and strictly bounded in [0.0, 100.0].
        - Reaching the END does not guarantee high accuracy if unnecessary detours occurred.

    Returns:
        float: Accuracy percentage rounded to 1 decimal place.
    """
    devs = trajectory_deviations(xs, ys, waypoints)
    if len(devs) == 0:
        return 0.0

    sigma = max(float(corridor_tolerance_px), 1.0)
    scores = np.exp(-(devs ** 2) / (2.0 * (sigma ** 2)))
    mean_score = float(np.mean(scores)) * 100.0

    clamped_score = max(0.0, min(100.0, mean_score))
    return float(round(clamped_score, 1))
