"""
metrics/performance.py — Session Performance Collector

Owns the per-session data buffer, delegates computation to
metrics.distance and metrics.smoothness, and persists results to CSV.

Dependency chain (no circular imports):
    metrics.performance
        ← metrics.distance   (pure math)
        ← metrics.smoothness (pure math)
        ← config             (METRICS_SAVE_DIR, SAVE_* flags)
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
from metrics.distance import (
    path_length,
    ideal_distance,
    path_efficiency,
    rom_dimensions,
    segment_speeds,
)
from metrics.smoothness import normalised_jerk_score, tremor_index

log = logging.getLogger(__name__)


class MetricsCollector:
    """
    Accumulates hand-cursor position samples during a game session
    and computes clinical rehabilitation metrics at the end.

    Usage:
        collector = MetricsCollector(start=(90, 650), end=(910, 50))

        # During gameplay (called every frame):
        collector.start_recording()
        collector.record(cx, cy, dt)
        collector.stop_recording()

        # At session end:
        summary = collector.compute()
        collector.save_csv(summary, difficulty="Easy", level_name="Easy-1")

        # Live HUD:
        speed = collector.live_speed()
    """

    def __init__(
        self,
        start: Tuple[int, int],
        end:   Tuple[int, int],
        min_path_distance: Optional[float] = None,
    ) -> None:
        self.start = start
        self.end   = end
        self.min_path_distance = min_path_distance

        self._xs: List[float] = []
        self._ys: List[float] = []
        self._ts: List[float] = []   # cumulative elapsed time (seconds)

        self._total_time: float = 0.0
        self._active: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Data collection
    # ─────────────────────────────────────────────────────────────────────────

    def start_recording(self) -> None:
        """Begin a new recording session (resets all buffers)."""
        self._xs.clear()
        self._ys.clear()
        self._ts.clear()
        self._total_time = 0.0
        self._active = True

    def record(self, cx: float, cy: float, dt: float) -> None:
        """
        Append one position sample.

        Args:
            cx, cy: Canvas pixel position of the cursor.
            dt:     Seconds elapsed since the previous sample.
        """
        if not self._active:
            return
        self._total_time += dt
        self._xs.append(float(cx))
        self._ys.append(float(cy))
        self._ts.append(self._total_time)

    def stop_recording(self) -> None:
        """Freeze the buffer (no more samples accepted)."""
        self._active = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Metric computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute all clinical metrics from the recorded buffer.

        Returns:
            dict mapping metric name → float value.
        """
        xs = np.array(self._xs)
        ys = np.array(self._ys)
        ts = np.array(self._ts)
        n  = len(xs)

        result: dict = {
            "completion_time_s": round(self._total_time, 3),
        }

        if n < 2:
            result.update({
                "path_length_px":    0.0,
                "ideal_distance_px": 0.0,
                "path_efficiency":   0.0,
                "peak_speed_px_s":   0.0,
                "mean_speed_px_s":   0.0,
                "normalised_jerk":   0.0,
                "tremor_index":      0.0,
                "rom_width_px":      0.0,
                "rom_height_px":     0.0,
            })
            return result

        # ── Spatial ───────────────────────────────────────────────────────────
        plen = path_length(xs, ys)
        idist = ideal_distance(self.start, self.end)
        min_p = self.min_path_distance if (self.min_path_distance is not None and self.min_path_distance > 0.0) else idist
        eff   = path_efficiency(xs, ys, self.start, self.end, min_path=min_p)
        rom_w, rom_h = rom_dimensions(xs, ys)

        # ── Speed ─────────────────────────────────────────────────────────────
        speeds = segment_speeds(xs, ys, ts)
        peak_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0
        mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0

        # ── Smoothness ────────────────────────────────────────────────────────
        njs    = normalised_jerk_score(xs, ys, ts, self.start, self.end)
        tremor = tremor_index(xs, ys, ts)

        result.update({
            "path_length_px":       round(plen,       1),
            "ideal_distance_px":    round(idist,      1),
            "min_path_distance_px": round(min_p,      1),
            "path_efficiency":      round(eff,         4),
            "peak_speed_px_s":      round(peak_speed, 1),
            "mean_speed_px_s":      round(mean_speed, 1),
            "normalised_jerk":      round(njs,         4),
            "tremor_index":         round(tremor,      4),
            "rom_width_px":         round(rom_w,       1),
            "rom_height_px":        round(rom_h,       1),
        })
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_csv(
        self,
        metrics_dict: dict,
        difficulty:  str = "Unknown",
        level_name:  str = "Unknown",
        maze_seed:   Optional[Any] = None,
    ) -> Optional[str]:
        """
        Append session results to a date-stamped CSV file.

        Args:
            metrics_dict: Calculated clinical motion metrics.
            difficulty:   Active difficulty name.
            level_name:   Title of the level.
            maze_seed:    Random seed used for procedural generation (or "FIXED").

        Returns:
            Absolute path to the saved file, or None on failure.
        """
        save_dir = config.METRICS_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now()
        filename = now.strftime("%Y-%m-%d") + "_sessions.csv"
        filepath = os.path.join(save_dir, filename)

        row = {
            "timestamp":  now.strftime("%Y-%m-%d %H:%M:%S"),
            "difficulty": difficulty,
            "level":      level_name,
            "maze_seed":  str(maze_seed) if maze_seed is not None else "FIXED",
            **metrics_dict,
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
    #  Live HUD helpers (called every frame)
    # ─────────────────────────────────────────────────────────────────────────

    def live_speed(self) -> float:
        """Instantaneous speed from the last two samples (px/s)."""
        if len(self._xs) < 2 or len(self._ts) < 2:
            return 0.0
        dx = self._xs[-1] - self._xs[-2]
        dy = self._ys[-1] - self._ys[-2]
        dt = max(self._ts[-1] - self._ts[-2], 1e-6)
        return math.sqrt(dx * dx + dy * dy) / dt

    def live_path_length(self) -> float:
        """Running arc length for HUD display."""
        if len(self._xs) < 2:
            return 0.0
        xs = np.array(self._xs)
        ys = np.array(self._ys)
        return path_length(xs, ys)

    @property
    def elapsed_time(self) -> float:
        return self._total_time

    @property
    def sample_count(self) -> int:
        return len(self._xs)
