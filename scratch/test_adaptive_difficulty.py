"""
scratch/test_adaptive_difficulty.py — Test Suite for Adaptive Difficulty

Validates:
  1. Multi-dimensional clinical criteria (accuracy, efficiency, smoothness, completion, collisions).
  2. Fast completion time alone does NOT trigger difficulty increase.
  3. Consistent high performance triggers INCREASE suggestion.
  4. Motor struggle (low accuracy, low efficiency, or excessive collisions) triggers DECREASE suggestion.
  5. Moderate steady performance triggers MAINTAIN suggestion.
  6. Configurable thresholds and consistency window.
  7. Optional toggle ADAPTIVE_DIFFICULTY = True / False.
  8. Results screen visual rendering with recommendation badge and guidance.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from metrics.adaptive import (
    AdaptiveRecommendation,
    evaluate_adaptive_difficulty,
    _is_high_performance,
)
from ui.screens import draw_results_screen


class TestAdaptiveDifficulty(unittest.TestCase):
    def test_high_performance_suggests_increase_when_consistent(self):
        """Consistent high performance across all 5 dimensions triggers INCREASE suggestion."""
        strong_session_1 = {
            "trajectory_accuracy": 95.0,
            "path_efficiency": 88.0,
            "smoothness_score": 82.0,
            "completion_status": "COMPLETED",
            "completion_time_s": 15.0,
            "difficulty": "EASY",
            "collision_count": 0,
        }
        current_metrics = {
            "trajectory_accuracy": 96.0,
            "path_efficiency": 89.0,
            "smoothness_score": 85.0,
            "completion_time_s": 14.5,
        }

        rec = evaluate_adaptive_difficulty(
            current_metrics=current_metrics,
            current_difficulty="EASY",
            wall_hits=0,
            completion_status="COMPLETED",
            session_history=[strong_session_1],
            consistency_window=2,
        )

        self.assertIsNotNone(rec)
        self.assertEqual(rec.action, "INCREASE")
        self.assertEqual(rec.target_difficulty, "MEDIUM")
        self.assertIn("MEDIUM", rec.headline)
        self.assertEqual(rec.current_difficulty, "EASY")

    def test_fast_time_with_poor_accuracy_or_collisions_does_not_suggest_increase(self):
        """
        CRITICAL CLINICAL REQUIREMENT:
        Faster time must NEVER be assumed to mean better rehabilitation.
        Fast flailing with low accuracy or wall collisions must NOT suggest difficulty increase.
        """
        rushed_metrics = {
            "completion_time_s": 2.1,  # extremely fast time!
            "trajectory_accuracy": 62.0,  # poor accuracy!
            "path_efficiency": 55.0,
            "smoothness_score": 38.0,  # jerky movement!
        }

        rec = evaluate_adaptive_difficulty(
            current_metrics=rushed_metrics,
            current_difficulty="EASY",
            wall_hits=5,  # 5 wall collisions!
            completion_status="COMPLETED",
            consistency_window=1,
        )

        self.assertIsNotNone(rec)
        # Must NOT suggest increase despite lightning fast completion time!
        self.assertNotEqual(rec.action, "INCREASE")
        self.assertEqual(rec.action, "DECREASE")
        self.assertIn("collision", rec.rationale)

    def test_struggling_performance_suggests_decrease_or_easier_level(self):
        """Low accuracy or excessive collisions on MEDIUM/HARD suggests easier level."""
        struggle_metrics = {
            "trajectory_accuracy": 60.0,
            "path_efficiency": 48.0,
            "smoothness_score": 45.0,
            "completion_time_s": 45.0,
        }

        rec_med = evaluate_adaptive_difficulty(
            current_metrics=struggle_metrics,
            current_difficulty="MEDIUM",
            wall_hits=4,
            completion_status="COMPLETED",
        )
        self.assertIsNotNone(rec_med)
        self.assertEqual(rec_med.action, "DECREASE")
        self.assertEqual(rec_med.target_difficulty, "EASY")
        self.assertIn("EASY", rec_med.headline)

        rec_hard = evaluate_adaptive_difficulty(
            current_metrics=struggle_metrics,
            current_difficulty="HARD",
            wall_hits=6,
            completion_status="COMPLETED",
        )
        self.assertIsNotNone(rec_hard)
        self.assertEqual(rec_hard.action, "DECREASE")
        self.assertEqual(rec_hard.target_difficulty, "MEDIUM")
        self.assertIn("MEDIUM", rec_hard.headline)

    def test_moderate_steady_performance_suggests_maintain(self):
        """Mid-range solid performance suggests MAINTAIN to reinforce motor patterns."""
        steady_metrics = {
            "trajectory_accuracy": 82.0,
            "path_efficiency": 74.0,
            "smoothness_score": 68.0,
            "completion_time_s": 22.0,
        }

        rec = evaluate_adaptive_difficulty(
            current_metrics=steady_metrics,
            current_difficulty="EASY",
            wall_hits=1,
            completion_status="COMPLETED",
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.action, "MAINTAIN")
        self.assertEqual(rec.target_difficulty, "EASY")
        self.assertIn("practice", rec.headline.lower())

    def test_consistency_window_delays_increase_until_repeated(self):
        """When consistency_window=2, a single strong run alone suggests MAINTAIN until verified."""
        current_metrics = {
            "trajectory_accuracy": 96.0,
            "path_efficiency": 90.0,
            "smoothness_score": 85.0,
            "completion_time_s": 15.0,
        }

        # Window = 2, but NO prior history
        rec_first_run = evaluate_adaptive_difficulty(
            current_metrics=current_metrics,
            current_difficulty="EASY",
            wall_hits=0,
            completion_status="COMPLETED",
            session_history=[],
            consistency_window=2,
        )
        self.assertIsNotNone(rec_first_run)
        self.assertEqual(rec_first_run.action, "MAINTAIN")
        self.assertIn("1 more", rec_first_run.headline)

        # Now supply prior strong history:
        prior_strong = {
            "trajectory_accuracy": 94.0,
            "path_efficiency": 88.0,
            "smoothness_score": 80.0,
            "completion_status": "COMPLETED",
            "collision_count": 0,
            "difficulty": "EASY",
        }
        rec_second_run = evaluate_adaptive_difficulty(
            current_metrics=current_metrics,
            current_difficulty="EASY",
            wall_hits=0,
            completion_status="COMPLETED",
            session_history=[prior_strong],
            consistency_window=2,
        )
        self.assertIsNotNone(rec_second_run)
        self.assertEqual(rec_second_run.action, "INCREASE")

    def test_adaptive_difficulty_toggle_disabled(self):
        """When ADAPTIVE_DIFFICULTY is False, returns None."""
        metrics = {
            "trajectory_accuracy": 98.0,
            "path_efficiency": 92.0,
            "smoothness_score": 90.0,
        }
        with patch("config.ADAPTIVE_DIFFICULTY", False):
            rec = evaluate_adaptive_difficulty(
                current_metrics=metrics,
                current_difficulty="EASY",
                wall_hits=0,
                completion_status="COMPLETED",
            )
            self.assertIsNone(rec)

    def test_configurable_thresholds_override(self):
        """Custom threshold overrides must be respected."""
        metrics = {
            "trajectory_accuracy": 82.0,  # below default 90, but above custom 80
            "path_efficiency": 80.0,      # below default 85, but above custom 75
            "smoothness_score": 72.0,     # below default 75, but above custom 70
        }
        # With default thresholds: does NOT qualify for high performance
        is_high_default, _ = _is_high_performance(
            metrics, wall_hits=0, is_completed=True,
            acc_th=90.0, eff_th=85.0, sm_th=75.0, max_coll=1
        )
        self.assertFalse(is_high_default)

        # With customized lower thresholds: qualifies as high performance
        is_high_custom, _ = _is_high_performance(
            metrics, wall_hits=0, is_completed=True,
            acc_th=80.0, eff_th=75.0, sm_th=70.0, max_coll=1
        )
        self.assertTrue(is_high_custom)

    def test_results_screen_renders_with_recommendation_card(self):
        """Results screen must render recommendation card and guidance cleanly without errors."""
        W, H = 800, 600
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        rec = AdaptiveRecommendation(
            action="INCREASE",
            current_difficulty="EASY",
            target_difficulty="MEDIUM",
            badge_title="READY TO ADVANCE",
            headline="Consistent high motor control. Ready for MEDIUM difficulty!",
            rationale="High accuracy (96%), efficiency (88%), and smoothness (85).",
            color=(90, 225, 140),
        )
        metrics = {
            "completion_time_s": 12.5,
            "actual_distance": 450.0,
            "minimum_distance": 380.0,
            "path_efficiency": 84.4,
            "trajectory_accuracy": 96.0,
            "smoothness_score": 85.0,
            "deviation_events": 0,
        }

        buttons = draw_results_screen(
            canvas=canvas,
            W=W,
            H=H,
            final_metrics=metrics,
            wall_hits=0,
            elapsed=12.5,
            difficulty_name="EASY",
            adaptive_recommendation=rec,
        )

        self.assertGreaterEqual(len(buttons), 5)
        self.assertGreater(np.count_nonzero(canvas), 10000)


if __name__ == "__main__":
    unittest.main()
