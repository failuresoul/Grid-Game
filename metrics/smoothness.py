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
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        jerk_integral = float(trapz_fn(jerk_sq, dx=mean_dt))

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


def movement_smoothness_score(
    xs: Any,
    ys: Optional[Any] = None,
    ts: Optional[Any] = None,
    min_step_px: float = 0.5,
) -> float:
    """
    Game-Derived Movement Smoothness Metric (0–100 scale).

    NOTE: This is a GAME-DERIVED movement smoothness metric, NOT a clinical measure.
    For formal clinical kinematic evaluations, refer to Normalised Jerk Score (NJS).

    Analyzes changes in velocity magnitude and direction between consecutive trajectory points:
      - Directional continuity: cosine of angle between successive movement vectors.
      - Velocity variation: relative speed changes between successive movement vectors.

    Score interpretation (0 to 100):
        100.0: Perfectly smooth, continuous, straight or gentle sweeping motion.
        80-99: Smooth natural motion with gradual turns and minor micro-adjustments.
        50-79: Moderate jerkiness, hesitated corrections, or angular zigzags.
        < 50:  Highly erratic, jittery, or frequent sharp back-and-forth reversals.

    Requirements met:
        - Normalized to a useful 0–100 scale.
        - Avoids division by zero.
        - Handles short trajectories safely (returns 100.0 for < 3 points).
        - Labeled explicitly as a game-derived movement smoothness metric.

    Supports:
        movement_smoothness_score(xs, ys, ts)
        movement_smoothness_score([(x0, y0), (x1, y1), ...])
    """
    # 1. Unpack input formats
    pts_x: List[float] = []
    pts_y: List[float] = []
    pts_t: List[float] = []

    if ys is None and isinstance(xs, (list, tuple, np.ndarray)):
        # xs is a sequence of points [(x, y), ...] or [(x, y, t), ...]
        for item in xs:
            if isinstance(item, (tuple, list, np.ndarray)) and len(item) >= 2:
                try:
                    px = float(item[0])
                    py = float(item[1])
                    if not (math.isnan(px) or math.isnan(py) or math.isinf(px) or math.isinf(py)):
                        pts_x.append(px)
                        pts_y.append(py)
                        if len(item) >= 3:
                            pts_t.append(float(item[2]))
                except (ValueError, TypeError):
                    continue
    else:
        # Separate xs and ys sequences
        nx = len(xs) if xs is not None else 0
        ny = len(ys) if ys is not None else 0
        n = min(nx, ny)
        has_ts = ts is not None and len(ts) >= n
        for i in range(n):
            try:
                px = float(xs[i])
                py = float(ys[i])
                if not (math.isnan(px) or math.isnan(py) or math.isinf(px) or math.isinf(py)):
                    pts_x.append(px)
                    pts_y.append(py)
                    if has_ts:
                        pts_t.append(float(ts[i]))
            except (ValueError, TypeError):
                continue

    # 2. Handle short trajectories safely
    if len(pts_x) < 3:
        return 100.0

    # 3. Filter out stationary consecutive duplicates / camera noise (< min_step_px)
    filt_x: List[float] = [pts_x[0]]
    filt_y: List[float] = [pts_y[0]]
    filt_t: List[float] = [pts_t[0]] if pts_t else []

    for i in range(1, len(pts_x)):
        dx = pts_x[i] - filt_x[-1]
        dy = pts_y[i] - filt_y[-1]
        dist = math.hypot(dx, dy)
        if dist >= min_step_px:
            filt_x.append(pts_x[i])
            filt_y.append(pts_y[i])
            if pts_t and i < len(pts_t):
                filt_t.append(pts_t[i])

    # Check length after filtering static points
    m = len(filt_x)
    if m < 3:
        return 100.0

    # 4. Compute consecutive displacement vectors and speeds
    displacements: List[Tuple[float, float, float, float]] = []  # (dx, dy, len, speed)
    use_time = len(filt_t) == m

    for i in range(m - 1):
        dx = filt_x[i + 1] - filt_x[i]
        dy = filt_y[i + 1] - filt_y[i]
        seg_len = math.hypot(dx, dy)
        if use_time:
            dt = max(filt_t[i + 1] - filt_t[i], 1e-4)
        else:
            dt = 1.0
        speed = seg_len / dt
        displacements.append((dx, dy, seg_len, speed))

    if len(displacements) < 2:
        return 100.0

    # 5. Evaluate directional changes and velocity fluctuations
    jerk_penalties: List[float] = []

    for i in range(len(displacements) - 1):
        dx1, dy1, len1, spd1 = displacements[i]
        dx2, dy2, len2, spd2 = displacements[i + 1]

        denom_len = max(len1 * len2, 1e-9)
        dot = dx1 * dx2 + dy1 * dy2
        cos_theta = max(-1.0, min(1.0, dot / denom_len))

        # Direction penalty: 0.0 for straight continuation (cos=1), 1.0 for 180° reversal (cos=-1)
        p_dir = (1.0 - cos_theta) / 2.0

        # Velocity variation penalty: relative change in speed
        max_spd = max(spd1, spd2, 1e-4)
        p_spd = min(1.0, abs(spd2 - spd1) / max_spd)

        # Segment jerkiness (60% directional continuity, 40% velocity stability)
        seg_jerk = 0.6 * p_dir + 0.4 * p_spd
        jerk_penalties.append(seg_jerk)

    if not jerk_penalties:
        return 100.0

    mean_jerk = float(np.mean(jerk_penalties))
    smoothness = max(0.0, min(100.0, 100.0 * (1.0 - mean_jerk)))
    return float(round(smoothness, 1))


# Alias for explicit metric name
smoothness_score = movement_smoothness_score

