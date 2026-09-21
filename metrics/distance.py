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
from typing import List, Tuple

import numpy as np


def path_length(xs: np.ndarray, ys: np.ndarray) -> float:
    """Total arc length of the recorded trajectory in pixels."""
    if len(xs) < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def ideal_distance(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """Euclidean straight-line distance from start to end (pixels)."""
    return math.dist(start, end)


def path_efficiency(
    xs: np.ndarray,
    ys: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """
    Path efficiency = ideal_distance / actual_path_length.

    Range: (0, 1].  A value of 1.0 means the player took the perfect
    straight-line path.  Lower values indicate detours or oscillations.
    """
    actual = path_length(xs, ys)
    if actual < 1e-6:
        return 0.0
    return min(ideal_distance(start, end) / actual, 1.0)


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
