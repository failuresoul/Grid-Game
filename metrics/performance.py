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
from metrics.accuracy import (
    path_deviation_summary,
    route_corridor_metrics,
    trajectory_accuracy_percentage,
)
from metrics.distance import (
    actual_distance,
    path_length,
    ideal_distance,
    path_efficiency,
    rom_dimensions,
    segment_speeds,
)
from metrics.smoothness import (
    normalised_jerk_score,
    tremor_index,
    movement_smoothness_score,
    smoothness_score,
)

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
        optimal_waypoints: Optional[Sequence[Tuple[float, float]]] = None,
        corridor_tolerance_px: float = 40.0,
    ) -> None:
        self.start = start
        self.end   = end
        self.min_path_distance = min_path_distance
        self.optimal_waypoints: List[Tuple[float, float]] = (
            [tuple(p) for p in optimal_waypoints]
            if optimal_waypoints and len(optimal_waypoints) >= 2
            else [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
        )
        self.corridor_tolerance_px = float(corridor_tolerance_px)

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
            min_p = self.minimum_distance
            result.update({
                "minimum_distance":       round(min_p, 1),
                "actual_distance":        0.0,
                "actual_distance_px":     0.0,
                "path_length_px":         0.0,
                "ideal_distance_px":      round(min_p, 1),
                "min_path_distance_px":   round(min_p, 1),
                "path_efficiency":        0.0,
                "path_efficiency_ratio":  0.0,
                "trajectory_accuracy":    0.0,
                "mean_path_deviation_px": 0.0,
                "max_path_deviation_px":  0.0,
                "rms_path_deviation_px":  0.0,
                "time_outside_route_s":   0.0,
                "pct_time_outside":       0.0,
                "deviation_events":       0,
                "corridor_adherence_pct": 100.0,
                "smoothness_score":       100.0,
                "game_smoothness_score":  100.0,
                "peak_speed_px_s":        0.0,
                "mean_speed_px_s":        0.0,
                "normalised_jerk":        0.0,
                "tremor_index":           0.0,
                "rom_width_px":           0.0,
                "rom_height_px":          0.0,
            })
            return result

        # ── Spatial & Distance ────────────────────────────────────────────────
        act_dist = actual_distance(xs, ys)
        idist = ideal_distance(self.start, self.end)
        min_p = self.min_path_distance if (self.min_path_distance is not None and self.min_path_distance > 0.0) else idist
        eff   = path_efficiency(actual=act_dist, minimum=min_p)
        rom_w, rom_h = rom_dimensions(xs, ys)

        # ── Trajectory Accuracy & Route Deviation ─────────────────────────────
        dev_summary = path_deviation_summary(xs, ys, self.optimal_waypoints)
        corr_metrics = route_corridor_metrics(
            xs, ys, ts, self.optimal_waypoints, corridor_tolerance_px=self.corridor_tolerance_px
        )
        traj_acc = trajectory_accuracy_percentage(
            xs, ys, self.optimal_waypoints, corridor_tolerance_px=self.corridor_tolerance_px
        )

        # ── Speed ─────────────────────────────────────────────────────────────
        speeds = segment_speeds(xs, ys, ts)
        peak_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0
        mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0

        # ── Smoothness ────────────────────────────────────────────────────────
        njs    = normalised_jerk_score(xs, ys, ts, self.start, self.end)
        tremor = tremor_index(xs, ys, ts)
        smooth = movement_smoothness_score(xs, ys, ts)

        result.update({
            # Distance dimension
            "minimum_distance":       round(min_p,      1),
            "actual_distance":        round(act_dist,   1),
            "actual_distance_px":     round(act_dist,   1),
            "path_length_px":         round(act_dist,   1),
            "ideal_distance_px":      round(idist,      1),
            "min_path_distance_px":   round(min_p,      1),
            # Path efficiency dimension
            "path_efficiency":        round(eff,        1),
            "path_efficiency_ratio":  round(eff / 100.0, 4),
            # Accuracy & deviation dimension (strictly separate from efficiency)
            "trajectory_accuracy":    round(traj_acc,   1),
            "mean_path_deviation_px": round(dev_summary["mean_path_deviation"], 1),
            "max_path_deviation_px":  round(dev_summary["max_path_deviation"],  1),
            "rms_path_deviation_px":  round(dev_summary["rms_path_deviation"],  1),
            "time_outside_route_s":   round(corr_metrics["time_outside_route_s"], 2),
            "pct_time_outside":       round(corr_metrics["pct_time_outside"], 1),
            "deviation_events":       int(corr_metrics["deviation_events"]),
            "corridor_adherence_pct": round(corr_metrics["corridor_adherence_pct"], 1),
            # Speed dimension
            "peak_speed_px_s":        round(peak_speed, 1),
            "mean_speed_px_s":        round(mean_speed, 1),
            # Smoothness dimension
            "smoothness_score":       round(smooth,      1),
            "game_smoothness_score":  round(smooth,      1),
            "normalised_jerk":        round(njs,         4),
            "tremor_index":           round(tremor,      4),
            "rom_width_px":           round(rom_w,       1),
            "rom_height_px":          round(rom_h,       1),
        })
        return result

    def get_trajectory_samples(self) -> List[Dict[str, float]]:
        """Return list of timestamped trajectory samples [{'x': x, 'y': y, 't': t}, ...]."""
        return [
            {"x": round(float(x), 2), "y": round(float(y), 2), "t": round(float(t), 3)}
            for x, y, t in zip(self._xs, self._ys, self._ts)
        ]

    def save_session(
        self,
        metrics_dict: dict,
        difficulty: str = "Unknown",
        level_name: str = "Unknown",
        maze_seed: Optional[Any] = None,
        completion_status: str = "COMPLETED",
        wall_hits: int = 0,
        save_dir: str = config.METRICS_SAVE_DIR,
    ) -> Optional[str]:
        """
        Record complete session to JSON and daily CSV via metrics.session_recorder.
        """
        from metrics.session_recorder import save_session as recorder_save
        traj = self.get_trajectory_samples()
        return recorder_save(
            difficulty=difficulty,
            maze_seed=maze_seed,
            completion_status=completion_status,
            completion_time=float(metrics_dict.get("completion_time_s", self._total_time)),
            actual_distance=float(metrics_dict.get("actual_distance", metrics_dict.get("path_length_px", 0.0))),
            minimum_distance=float(metrics_dict.get("minimum_distance", metrics_dict.get("min_path_distance_px", 0.0))),
            path_efficiency=float(metrics_dict.get("path_efficiency", 0.0)),
            accuracy=float(metrics_dict.get("trajectory_accuracy", 100.0)),
            smoothness=float(metrics_dict.get("smoothness_score", 100.0)),
            collision_count=int(wall_hits),
            deviation_count=int(metrics_dict.get("deviation_events", 0)),
            trajectory=traj,
            optimal_path=self.optimal_waypoints,
            level_name=level_name,
            game_performance_metrics=metrics_dict,
            save_dir=save_dir,
            save_csv_also=True,
        )

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
        return actual_distance(self._xs, self._ys)

    def live_actual_distance(self) -> float:
        """Running actual trajectory distance in pixels."""
        return actual_distance(self._xs, self._ys)

    @property
    def actual_distance(self) -> float:
        """Total actual trajectory distance in pixels."""
        return actual_distance(self._xs, self._ys)

    @property
    def minimum_distance(self) -> float:
        """Minimum / optimal path distance in pixels."""
        if self.min_path_distance is not None and self.min_path_distance > 0.0:
            return float(self.min_path_distance)
        return float(ideal_distance(self.start, self.end))

    @property
    def path_efficiency(self) -> float:
        """Path efficiency as a percentage [0.0%, 100.0%] rounded to 1 decimal place."""
        return path_efficiency(actual=self.actual_distance, minimum=self.minimum_distance)

    @property
    def trajectory_accuracy(self) -> float:
        """Trajectory accuracy percentage [0.0%, 100.0%] along intended route."""
        if len(self._xs) < 2:
            return 0.0
        return trajectory_accuracy_percentage(
            self._xs, self._ys, self.optimal_waypoints, self.corridor_tolerance_px
        )

    @property
    def mean_path_deviation(self) -> float:
        """Mean orthogonal path deviation from intended route in pixels."""
        if len(self._xs) < 1:
            return 0.0
        return path_deviation_summary(self._xs, self._ys, self.optimal_waypoints)["mean_path_deviation"]

    @property
    def max_path_deviation(self) -> float:
        """Maximum orthogonal path deviation from intended route in pixels."""
        if len(self._xs) < 1:
            return 0.0
        return path_deviation_summary(self._xs, self._ys, self.optimal_waypoints)["max_path_deviation"]

    @property
    def time_outside_route_s(self) -> float:
        """Cumulative seconds spent outside the intended corridor."""
        if len(self._xs) < 2:
            return 0.0
        return route_corridor_metrics(
            self._xs, self._ys, self._ts, self.optimal_waypoints, self.corridor_tolerance_px
        )["time_outside_route_s"]

    @property
    def deviation_events(self) -> int:
        """Count of distinct wrong movement / excursion episodes outside intended corridor."""
        if len(self._xs) < 2:
            return 0
        return route_corridor_metrics(
            self._xs, self._ys, self._ts, self.optimal_waypoints, self.corridor_tolerance_px
        )["deviation_events"]

    @property
    def smoothness_score(self) -> float:
        """
        Game-derived movement smoothness score [0.0, 100.0] (NOT a clinical measure).
        Higher = smoother movement. Evaluates directional and speed variations.
        """
        if len(self._xs) < 3:
            return 100.0
        return movement_smoothness_score(self._xs, self._ys, self._ts)

    @property
    def raw_trajectory(self) -> List[Tuple[float, float, float]]:
        """
        Raw continuous trajectory samples [(x, y, timestamp_s), ...]
        Retained in full without loss so smoothing algorithms can be refined.
        """
        return list(zip(self._xs, self._ys, self._ts))

    def get_raw_trajectory(self) -> List[Tuple[float, float, float]]:
        """Return a copy of the complete raw trajectory samples [(x, y, t), ...]."""
        return list(zip(self._xs, self._ys, self._ts))

    @property
    def elapsed_time(self) -> float:
        return self._total_time

    @property
    def sample_count(self) -> int:
        return len(self._xs)
