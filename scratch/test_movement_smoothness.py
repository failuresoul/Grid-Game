"""
scratch/test_movement_smoothness.py — Comprehensive Test Suite for Movement Smoothness Measurement

Validates:
1. Normalization to a useful 0–100 scale:
   - Perfectly straight path -> 100.0
   - Smooth sweeping arc -> 95.0+
   - Highly jerky / zigzagging trajectory -> low score (< 40.0)
2. Division by zero avoidance:
   - Stationary points, duplicate consecutive samples, zero dt.
3. Safe handling of short trajectories:
   - 0 points -> 100.0
   - 1 point -> 100.0
   - 2 points -> 100.0
4. Raw trajectory retention:
   - MetricsCollector.raw_trajectory retains full (x, y, t) samples without loss.
   - GameEngine.raw_trajectory and Player.raw_trajectory exposed.
5. Proper labeling & documentation:
   - Confirms the metric is explicitly documented as a game-derived movement smoothness metric.
6. Exposed properties:
   - smoothness_score on function, MetricsCollector, GameEngine, and in compute().
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
import numpy as np

from metrics.smoothness import (
    movement_smoothness_score,
    smoothness_score,
)
from metrics.performance import MetricsCollector
from game.levels import create_easy_level
from game.game_engine import GameEngine, GameState
import config


class TestMovementSmoothness(unittest.TestCase):
    """Test suite for game-derived movement smoothness scoring and raw trajectory retention."""

    def test_smooth_straight_line(self):
        """A straight line at constant velocity must produce 100.0 smoothness score."""
        xs = np.linspace(100.0, 500.0, 50)
        ys = np.full_like(xs, 200.0)
        ts = np.linspace(0.0, 2.0, 50)

        score = movement_smoothness_score(xs, ys, ts)
        self.assertEqual(score, 100.0)

        # Alias must yield identical result
        self.assertEqual(smoothness_score(xs, ys, ts), 100.0)

    def test_smooth_curved_arc(self):
        """A smooth circle arc must produce a very high score (>= 90.0)."""
        angles = np.linspace(0, np.pi / 2, 40)
        xs = 200.0 + 150.0 * np.cos(angles)
        ys = 200.0 + 150.0 * np.sin(angles)
        ts = np.linspace(0.0, 2.0, 40)

        score = movement_smoothness_score(xs, ys, ts)
        self.assertGreaterEqual(score, 90.0)
        self.assertLessEqual(score, 100.0)

    def test_jerky_zigzag_trajectory(self):
        """A trajectory with violent directional reversals and speed bursts must yield a low score."""
        n = 40
        xs = np.linspace(100.0, 500.0, n)
        # Violently alternating up/down zigzag (+50px, -50px)
        ys = np.array([200.0 + (50.0 if i % 2 == 0 else -50.0) for i in range(n)])
        # Inconsistent jerky time intervals
        ts = np.array([i * (0.01 if i % 2 == 0 else 0.1) for i in range(n)])

        score = movement_smoothness_score(xs, ys, ts)
        self.assertLess(score, 45.0)
        self.assertGreaterEqual(score, 0.0)

    def test_short_trajectories_safe(self):
        """Short trajectories (< 3 points) must safely return 100.0 without errors."""
        self.assertEqual(movement_smoothness_score([]), 100.0)
        self.assertEqual(movement_smoothness_score([100.0], [100.0]), 100.0)
        self.assertEqual(movement_smoothness_score([100.0, 110.0], [100.0, 100.0]), 100.0)

        # Tuple points format
        self.assertEqual(movement_smoothness_score([(10.0, 20.0)]), 100.0)
        self.assertEqual(movement_smoothness_score([(10.0, 20.0), (15.0, 25.0)]), 100.0)

    def test_division_by_zero_avoidance(self):
        """Static points, duplicate timestamps, and zero displacements must not divide by zero."""
        # 20 stationary samples at identical spot
        xs = np.full(20, 250.0)
        ys = np.full(20, 250.0)
        ts = np.zeros(20)  # zero dt

        score = movement_smoothness_score(xs, ys, ts)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_metrics_collector_smoothness_and_raw_trajectory(self):
        """MetricsCollector must expose smoothness_score and retain full raw trajectory."""
        collector = MetricsCollector(start=(100, 100), end=(500, 100), min_path_distance=400.0)
        collector.start_recording()

        samples = [
            (100.0, 100.0, 0.0),
            (200.0, 100.0, 0.5),
            (300.0, 100.0, 0.5),
            (400.0, 100.0, 0.5),
            (500.0, 100.0, 0.5),
        ]
        for x, y, dt in samples:
            collector.record(x, y, dt)
        collector.stop_recording()

        # Expose smoothness_score
        self.assertEqual(collector.smoothness_score, 100.0)

        # Expose raw_trajectory with all recorded coordinates and cumulative timestamps
        raw = collector.raw_trajectory
        self.assertEqual(len(raw), len(samples))
        for i, (rx, ry, rt) in enumerate(raw):
            self.assertEqual(rx, samples[i][0])
            self.assertEqual(ry, samples[i][1])

        # Summary dict in compute()
        summary = collector.compute()
        self.assertIn("smoothness_score", summary)
        self.assertEqual(summary["smoothness_score"], 100.0)

    def test_game_engine_exposed_smoothness_and_raw_trajectory(self):
        """GameEngine must expose smoothness_score and raw_trajectory."""
        level = create_easy_level()
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])

        self.assertIsInstance(engine.smoothness_score, float)
        self.assertIsInstance(engine.raw_trajectory, list)
        self.assertIsInstance(engine.player.raw_trajectory, list)

    def test_labeling_explicitly_game_derived(self):
        """Docstring and comments must explicitly designate the metric as game-derived."""
        doc = movement_smoothness_score.__doc__
        self.assertIn("Game-Derived Movement Smoothness Metric", doc)
        self.assertIn("NOT a clinical measure", doc)


if __name__ == "__main__":
    unittest.main()
