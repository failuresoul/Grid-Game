"""
metrics.py — Clinical Rehabilitation Metrics

Computes and stores per-session metrics relevant to upper-limb rehabilitation:
  - Task completion time
  - Path efficiency  (ideal distance / actual path length)
  - Normalised jerk score  (NJS — standard rehabilitation measure)
  - Peak speed  (maximum inter-frame displacement)
  - Tremor index  (high-frequency oscillation power via FFT)
  - Range of motion estimate  (bounding box of hand positions)

Metrics are saved to a CSV file in the 'sessions/' directory after each session.
"""

from __future__ import annotations
import csv
import logging
import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

import config

log = logging.getLogger(__name__)


class MetricsCollector:
    """
    Records position samples throughout a game session and computes
    rehabilitation metrics at the end.

    Usage:
        metrics = MetricsCollector(start=(50,650), end=(950,50))
        # each frame:
        metrics.record(cx, cy, dt)
        # at session end:
        summary = metrics.compute()
        metrics.save_csv(summary, difficulty="Easy", level_name="Easy-1")
    """

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        self.start = start
        self.end   = end

        # Raw sample lists (one per frame while player is moving)
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._ts: List[float] = []   # cumulative time (seconds)

        self._total_time: float = 0.0
        self._active = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Data collection
    # ─────────────────────────────────────────────────────────────────────────

    def start_recording(self) -> None:
        """Call when the player starts moving (leaves Start zone)."""
        self._xs.clear()
        self._ys.clear()
        self._ts.clear()
        self._total_time = 0.0
        self._active = True

    def record(self, cx: float, cy: float, dt: float) -> None:
        """
        Record one position sample.

        Args:
            cx, cy: canvas pixel position of the cursor
            dt:     elapsed seconds since last frame
        """
        if not self._active:
            return
        self._total_time += dt
        self._xs.append(float(cx))
        self._ys.append(float(cy))
        self._ts.append(self._total_time)

    def stop_recording(self) -> None:
        self._active = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Metric computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute all metrics from the recorded session data.

        Returns:
            dict of metric name → value (floats, unless noted)
        """
        xs = np.array(self._xs)
        ys = np.array(self._ys)
        ts = np.array(self._ts)

        n = len(xs)
        result: dict = {}

        # ── Task completion time ──────────────────────────────────────────────
        result["completion_time_s"] = round(self._total_time, 3)

        if n < 2:
            # Not enough data
            result.update({
                "path_length_px":       0.0,
                "ideal_distance_px":    0.0,
                "path_efficiency":      0.0,
                "peak_speed_px_s":      0.0,
                "mean_speed_px_s":      0.0,
                "normalised_jerk":      0.0,
                "tremor_index":         0.0,
                "rom_width_px":         0.0,
                "rom_height_px":        0.0,
            })
            return result

        # ── Path length ───────────────────────────────────────────────────────
        dx = np.diff(xs)
        dy = np.diff(ys)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        path_length = float(np.sum(segment_lengths))
        result["path_length_px"] = round(path_length, 1)

        # ── Path efficiency ───────────────────────────────────────────────────
        ideal_dist = math.dist(self.start, self.end)
        result["ideal_distance_px"] = round(ideal_dist, 1)
        if path_length > 0:
            efficiency = min(ideal_dist / path_length, 1.0)
        else:
            efficiency = 0.0
        result["path_efficiency"] = round(efficiency, 4)

        # ── Speed ─────────────────────────────────────────────────────────────
        dt_arr = np.diff(ts)
        dt_arr[dt_arr < 1e-6] = 1e-6   # prevent division by zero
        speeds = segment_lengths / dt_arr
        result["peak_speed_px_s"] = round(float(np.max(speeds)), 1)
        result["mean_speed_px_s"] = round(float(np.mean(speeds)), 1)

        # ── Normalised Jerk Score (NJS) ───────────────────────────────────────
        # NJS = sqrt(0.5 * integral(jerk²) dt) * (T⁵ / D²)
        # where jerk = d³position/dt³, T = movement time, D = ideal distance
        # Reference: Hogan & Sternad (2009), Flash & Hogan (1985)
        njs = self._normalised_jerk_score(xs, ys, ts)
        result["normalised_jerk"] = round(njs, 4)

        # ── Tremor index ──────────────────────────────────────────────────────
        tremor = self._tremor_index(xs, ys, ts)
        result["tremor_index"] = round(tremor, 4)

        # ── Range of motion (ROM) ─────────────────────────────────────────────
        result["rom_width_px"]  = round(float(np.max(xs) - np.min(xs)), 1)
        result["rom_height_px"] = round(float(np.max(ys) - np.min(ys)), 1)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal metric helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalised_jerk_score(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
    ) -> float:
        """
        Compute the dimensionless Normalised Jerk Score.

        A lower NJS indicates smoother movement (closer to minimum-jerk trajectory).
        Typical healthy adult: NJS ≈ 0–5.  Stroke patients: significantly higher.
        """
        n = len(xs)
        if n < 4 or self._total_time < 0.05:
            return 0.0
        try:
            # Compute velocity, acceleration, jerk via finite differences
            dt = np.diff(ts)
            dt[dt < 1e-6] = 1e-6

            vx = np.diff(xs) / dt
            vy = np.diff(ys) / dt

            ax = np.diff(vx) / dt[:-1]
            ay = np.diff(vy) / dt[:-1]

            jx = np.diff(ax) / dt[:-2]
            jy = np.diff(ay) / dt[:-2]

            jerk_sq = jx**2 + jy**2
            # Trapezoidal integration of jerk²
            dt_j = dt[:-2]
            jerk_integral = float(np.trapz(jerk_sq, dx=np.mean(dt_j)))

            T = self._total_time
            D = max(math.dist(self.start, self.end), 1.0)

            # Standard NJS formula (dimensionless)
            njs = math.sqrt(0.5 * jerk_integral * (T**5 / D**2))
            return njs
        except Exception:
            return 0.0

    def _tremor_index(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
    ) -> float:
        """
        Tremor index = ratio of high-frequency power (3–12 Hz) to total power
        in the hand-position signal.

        A higher value indicates more pathological tremor.
        Healthy adult: < 0.05.  Tremor patients: > 0.15.
        """
        n = len(xs)
        if n < 16 or self._total_time < 0.5:
            return 0.0
        try:
            # Assume roughly uniform sampling; estimate sample rate
            mean_dt = self._total_time / (n - 1)
            fs = 1.0 / max(mean_dt, 1e-3)

            # Use the resultant displacement signal
            signal = np.sqrt(xs**2 + ys**2)
            signal -= np.mean(signal)   # remove DC

            # FFT power spectrum
            fft_vals = np.fft.rfft(signal)
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(n, d=mean_dt)

            # Power in tremor band (3–12 Hz)
            tremor_mask = (freqs >= 3.0) & (freqs <= 12.0)
            total_power = float(np.sum(power))
            tremor_power = float(np.sum(power[tremor_mask]))

            if total_power < 1e-6:
                return 0.0
            return tremor_power / total_power
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_csv(
        self,
        metrics: dict,
        difficulty: str = "Unknown",
        level_name: str = "Unknown",
    ) -> Optional[str]:
        """
        Append session metrics to a dated CSV file in the sessions/ directory.

        Returns:
            Absolute path to the CSV file, or None on failure.
        """
        save_dir = config.METRICS_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now()
        filename = now.strftime("%Y-%m-%d") + "_sessions.csv"
        filepath = os.path.join(save_dir, filename)

        row = {
            "timestamp":       now.strftime("%Y-%m-%d %H:%M:%S"),
            "difficulty":      difficulty,
            "level":           level_name,
            **metrics,
        }

        file_exists = os.path.isfile(filepath)
        try:
            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            log.info(f"Session metrics saved → {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Failed to save session CSV: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Live HUD values (called every frame during gameplay)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def elapsed_time(self) -> float:
        return self._total_time

    @property
    def sample_count(self) -> int:
        return len(self._xs)

    def live_path_length(self) -> float:
        """Running path length for live HUD display."""
        if len(self._xs) < 2:
            return 0.0
        xs = np.array(self._xs)
        ys = np.array(self._ys)
        return float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))

    def live_speed(self) -> float:
        """Approximate current speed (px/s) from last two samples."""
        if len(self._xs) < 2 or len(self._ts) < 2:
            return 0.0
        dx = self._xs[-1] - self._xs[-2]
        dy = self._ys[-1] - self._ys[-2]
        dt = max(self._ts[-1] - self._ts[-2], 1e-6)
        return math.sqrt(dx**2 + dy**2) / dt
