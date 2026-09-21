"""
metrics/smoothness.py — Movement Smoothness Metrics

Pure numpy/math — no project imports.

Provides:
    normalised_jerk_score(xs, ys, ts, start, end)
        Dimensionless NJS — lower = smoother.
        Reference: Hogan & Sternad (2009), Flash & Hogan (1985).

    tremor_index(xs, ys, ts)
        Ratio of high-frequency power (3–12 Hz) to total signal power.
        Higher = more tremor.

These are standard clinical movement analysis measures used in
upper-limb stroke rehabilitation research.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


def normalised_jerk_score(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """
    Normalised Jerk Score (NJS).

    NJS = sqrt( 0.5 * ∫ jerk²(t) dt  *  T⁵ / D² )

    where:
        jerk = d³position/dt³
        T    = total movement time (s)
        D    = ideal straight-line distance (px)

    Interpretation:
        Healthy adult: NJS ≈ 0–5
        Stroke patient: significantly higher (10–50+)
        Lower is always better.

    Returns 0.0 if there is insufficient data.
    """
    n = len(xs)
    total_time = float(ts[-1]) if n > 1 else 0.0

    if n < 4 or total_time < 0.05:
        return 0.0

    try:
        dt = np.diff(ts).astype(float)
        dt = np.where(dt < 1e-6, 1e-6, dt)

        vx = np.diff(xs) / dt
        vy = np.diff(ys) / dt

        ax = np.diff(vx) / dt[:-1]
        ay = np.diff(vy) / dt[:-1]

        jx = np.diff(ax) / dt[:-2]
        jy = np.diff(ay) / dt[:-2]

        jerk_sq = jx * jx + jy * jy

        # Trapezoidal integration with mean dt step
        mean_dt = float(np.mean(dt[:-2]))
        jerk_integral = float(np.trapz(jerk_sq, dx=mean_dt))

        D = max(math.dist(start, end), 1.0)
        T = total_time
        njs = math.sqrt(0.5 * jerk_integral * (T ** 5 / D ** 2))
        return njs

    except Exception:
        return 0.0


def tremor_index(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
) -> float:
    """
    Tremor Index = power in the tremor frequency band (3–12 Hz)
                  divided by total signal power.

    Range: [0, 1].  Higher = more tremor.

    Interpretation:
        Healthy adult:  < 0.05
        Mild tremor:    0.05 – 0.15
        Severe tremor:  > 0.15

    Returns 0.0 if there is insufficient data (< 16 samples or < 0.5 s).
    """
    n = len(xs)
    if n < 16 or len(ts) < 2:
        return 0.0

    total_time = float(ts[-1] - ts[0])
    if total_time < 0.5:
        return 0.0

    try:
        mean_dt = total_time / (n - 1)
        if mean_dt < 1e-6:
            return 0.0

        # Resultant displacement signal (removes directional bias)
        signal = np.sqrt(xs.astype(float) ** 2 + ys.astype(float) ** 2)
        signal -= float(np.mean(signal))   # remove DC component

        # FFT power spectrum
        fft_vals = np.fft.rfft(signal)
        power    = np.abs(fft_vals) ** 2
        freqs    = np.fft.rfftfreq(n, d=mean_dt)

        # Tremor band: 3–12 Hz (pathological upper-limb tremor range)
        tremor_mask  = (freqs >= 3.0) & (freqs <= 12.0)
        total_power  = float(np.sum(power))
        tremor_power = float(np.sum(power[tremor_mask]))

        if total_power < 1e-6:
            return 0.0
        return min(tremor_power / total_power, 1.0)

    except Exception:
        return 0.0
