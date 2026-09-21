"""
scratch/test_path_efficiency.py — Comprehensive Test Suite for Path Efficiency

Validates:
1. Exact User Example:
   Minimum = 300, Actual = 420 -> 300 / 420 * 100 = 71.428...% -> 71.4%
2. Clamping:
   Sensible values [0.0%, 100.0%] (caps at 100.0% even if actual < theoretical minimum).
3. Avoid Division by Zero:
   Returns 0.0% when actual distance is 0.0.
4. Meaningful Movement Requirement:
   Returns 0.0% when actual distance < min_movement_px (default 5.0 px).
5. One Decimal Place Formatting:
   Returns float rounded to 1 decimal place.
6. Exposed Properties & Return Keys:
   - minimum_distance
   - actual_distance
   - path_efficiency
   on MetricsCollector, GameEngine, and in compute() summary dict.
7. Final Results Screen Integration:
   draw_win_overlay renders cleanly with minimum_distance, actual_distance, and path_efficiency.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
import numpy as np

from metrics.distance import (
    path_efficiency,
    actual_distance,
    ideal_distance,
)
from metrics.performance import MetricsCollector
from game.player import Player
from game.levels import Level, RectObstacle
from game.game_engine import GameEngine, GameState
import config


class TestPathEfficiency(unittest.TestCase):
    """Unit tests for Path Efficiency formula, constraints, and exposed attributes."""

    def test_user_formula_and_example(self):
        """Verify the user's exact specification:
        Minimum = 300, Actual = 420 -> Efficiency = 71.4%
        """
        eff = path_efficiency(minimum=300.0, actual=420.0)
        self.assertEqual(eff, 71.4)
        self.assertIsInstance(eff, float)

        # Keyword arguments: actual_distance and minimum_distance
        eff_kw = path_efficiency(minimum_distance=300.0, actual_distance=420.0)
        self.assertEqual(eff_kw, 71.4)

    def test_division_by_zero(self):
        """Division by zero must safely return 0.0%."""
        self.assertEqual(path_efficiency(minimum=300.0, actual=0.0), 0.0)
        self.assertEqual(path_efficiency(minimum=0.0, actual=0.0), 0.0)
        self.assertEqual(path_efficiency(minimum=-10.0, actual=0.0), 0.0)

    def test_meaningful_movement_threshold(self):
        """Path efficiency is only calculated after meaningful movement occurs (>= 5.0 px)."""
        # Under default threshold of 5.0 px:
        self.assertEqual(path_efficiency(minimum=300.0, actual=4.9), 0.0)
        self.assertEqual(path_efficiency(minimum=300.0, actual=2.0), 0.0)
        self.assertEqual(path_efficiency(minimum=300.0, actual=0.1), 0.0)

        # At or above threshold:
        eff = path_efficiency(minimum=5.0, actual=5.0)
        self.assertEqual(eff, 100.0)

        # Custom threshold
        self.assertEqual(path_efficiency(minimum=100.0, actual=8.0, min_movement_px=10.0), 0.0)
        self.assertEqual(path_efficiency(minimum=10.0, actual=10.0, min_movement_px=10.0), 100.0)

    def test_clamping_sensible_values(self):
        """Efficiency must be clamped between 0.0% and 100.0%."""
        # Perfect path: actual == minimum -> 100.0%
        self.assertEqual(path_efficiency(minimum=300.0, actual=300.0), 100.0)

        # Theoretical shortcut (e.g. direct leap): actual < minimum -> clamped to 100.0%
        self.assertEqual(path_efficiency(minimum=300.0, actual=150.0), 100.0)

        # Negative minimum: clamped to 0.0%
        self.assertEqual(path_efficiency(minimum=-50.0, actual=100.0), 0.0)

        # Negative actual distance (malformed): returns 0.0%
        self.assertEqual(path_efficiency(minimum=300.0, actual=-420.0), 0.0)

    def test_one_decimal_place_precision(self):
        """Percentage values must be rounded to exactly one decimal place."""
        # 300 / 350 * 100 = 85.7142... -> 85.7%
        self.assertEqual(path_efficiency(minimum=300.0, actual=350.0), 85.7)

        # 100 / 300 * 100 = 33.3333... -> 33.3%
        self.assertEqual(path_efficiency(minimum=100.0, actual=300.0), 33.3)

        # 200 / 300 * 100 = 66.6666... -> 66.7%
        self.assertEqual(path_efficiency(minimum=200.0, actual=300.0), 66.7)

    def test_legacy_trajectory_signature_support(self):
        """Trajectory arrays (xs, ys) passed to path_efficiency work seamlessly."""
        # Straight line of 420 px
        xs = np.linspace(100.0, 520.0, 50)
        ys = np.zeros_like(xs)
        eff = path_efficiency(xs, ys, min_path=300.0)
        self.assertEqual(eff, 71.4)

    def test_metrics_collector_exposed_properties_and_compute(self):
        """MetricsCollector must expose minimum_distance, actual_distance, path_efficiency."""
        start = (100, 100)
        end = (400, 100)  # ideal = 300 px
        collector = MetricsCollector(start=start, end=end, min_path_distance=300.0)

        # Initial state (no movement)
        self.assertEqual(collector.minimum_distance, 300.0)
        self.assertEqual(collector.actual_distance, 0.0)
        self.assertEqual(collector.path_efficiency, 0.0)

        # Record movement to 420 px total
        collector.start_recording()
        collector.record(100.0, 100.0, 0.0)
        collector.record(520.0, 100.0, 1.0)  # distance = 420 px
        collector.stop_recording()

        self.assertEqual(collector.minimum_distance, 300.0)
        self.assertAlmostEqual(collector.actual_distance, 420.0, places=1)
        self.assertEqual(collector.path_efficiency, 71.4)

        # Test compute() output dictionary
        summary = collector.compute()
        self.assertIn("minimum_distance", summary)
        self.assertIn("actual_distance", summary)
        self.assertIn("path_efficiency", summary)

        self.assertEqual(summary["minimum_distance"], 300.0)
        self.assertAlmostEqual(summary["actual_distance"], 420.0, places=1)
        self.assertEqual(summary["path_efficiency"], 71.4)
        self.assertAlmostEqual(summary["path_efficiency_ratio"], 0.714, places=2)

    def test_game_engine_exposed_properties(self):
        """GameEngine must expose minimum_distance, actual_distance, path_efficiency."""
        from game.levels import create_easy_level
        level = create_easy_level()
        level.minimum_path_distance = 300.0

        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        self.assertEqual(engine.minimum_distance, 300.0)
        self.assertEqual(engine.actual_distance, 0.0)
        self.assertEqual(engine.path_efficiency, 0.0)

        # Move player 420 px
        engine.state = GameState.RUNNING
        start_x, start_y = float(level.start[0]), float(level.start[1])
        engine.player.px = start_x
        engine.player.py = start_y
        engine.player.move_to_cursor(start_x + 420.0, start_y, walls=[])

        self.assertEqual(engine.minimum_distance, 300.0)
        self.assertAlmostEqual(engine.actual_distance, 420.0, places=1)
        self.assertEqual(engine.path_efficiency, 71.4)

    def test_win_overlay_renders_with_path_efficiency(self):
        """Verify draw_win_overlay formats and renders path efficiency row properly."""
        from ui.screens import draw_win_overlay
        import cv2

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        metrics = {
            "completion_time_s": 12.34,
            "minimum_distance": 300.0,
            "actual_distance": 420.0,
            "path_efficiency": 71.4,
            "normalised_jerk": 12.5,
            "tremor_index": 0.04,
            "peak_speed_px_s": 250.0,
            "rom_width_px": 500.0,
            "rom_height_px": 300.0,
        }

        # Should execute without throwing any exception
        draw_win_overlay(
            canvas=canvas,
            W=1280,
            H=720,
            anim_t=1.0,
            final_metrics=metrics,
            wall_hits=0,
            elapsed=12.34,
        )
        self.assertEqual(canvas.shape, (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
