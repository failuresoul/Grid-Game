"""
scratch/test_actual_distance.py — Test Suite for Actual Trajectory Distance Calculation

Tests:
  1. Mathematical correctness:
     Consecutive points: P1 = (x1, y1), P2 = (x2, y2)
     distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
     Total distance = sum of all consecutive movement distances.
  2. Ignore invalid points:
     (None, NaN, Inf, non-numeric strings, bad tuple sizes) are discarded.
  3. Handle missing hand frames:
     Sentinel None indicating lost tracking does NOT create a teleport leap.
  4. Avoid double-counting:
     Static frames or identical consecutive coordinates (0 displacement) add 0 distance.
  5. Keep complete trajectory:
     Player maintains full_trajectory without truncation.
  6. Consistent pixel coordinate system:
     Distance is in pixels/game units matching canvas coordinates.
  7. Expose actual_distance on:
     - Player: player.actual_distance
     - GameEngine: engine.actual_distance
     - MetricsCollector: metrics.actual_distance & metrics.compute()['actual_distance']
     - Function: actual_distance(points)
  8. End-to-end simulation with GameEngine.
"""

from __future__ import annotations
import math
import os
import sys
import unittest

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from game.player import Player
from game.game_engine import GameEngine, GameState
from game.levels import create_easy_level
from metrics.distance import actual_distance, path_length
from metrics.performance import MetricsCollector


class TestActualTrajectoryDistance(unittest.TestCase):
    """Test suite for actual hand/player trajectory distance calculation."""

    def test_mathematical_distance_formula(self):
        """Verify formula sqrt((x2-x1)^2 + (y2-y1)^2) summed over consecutive points."""
        # Right triangle 3-4-5: (0,0) -> (3,4) = 5.0
        # Horizontal 5: (3,4) -> (8,4) = 5.0
        # Vertical 12: (8,4) -> (8,16) = 12.0
        # Total = 5.0 + 5.0 + 12.0 = 22.0
        pts = [(0.0, 0.0), (3.0, 4.0), (8.0, 4.0), (8.0, 16.0)]
        expected = 22.0
        computed = actual_distance(pts)
        self.assertAlmostEqual(computed, expected, places=3)

    def test_ignore_invalid_points(self):
        """Verify that invalid points (NaN, Inf, strings, bad tuples) are safely ignored."""
        pts = [
            (0.0, 0.0),
            (float("nan"), 4.0),     # NaN coordinate
            (3.0, 4.0),              # Valid (dist from (0,0) = 5.0)
            "invalid_string",        # Bad type
            (float("inf"), 10.0),    # Inf coordinate
            (8.0, 4.0),              # Valid (dist from (3,4) = 5.0)
            (1.0,),                  # Malformed tuple (length 1)
            (8.0, 16.0),             # Valid (dist from (8,4) = 12.0)
        ]
        # Valid points are: (0,0), (3,4), (8,4), (8,16) -> distance = 22.0
        computed = actual_distance(pts)
        self.assertAlmostEqual(computed, 22.0, places=3)

    def test_handle_missing_hand_frames(self):
        """
        Verify that missing hand frames (sentinel None in tracking stream)
        do NOT cause a false teleport leap across the gap.
        """
        pts_with_gap = [
            (0.0, 0.0),
            (3.0, 4.0),              # dist = 5.0
            None,                    # Missing hand frame (tracking lost)
            None,                    # Missing hand frame
            (500.0, 500.0),          # Hand re-appears here (new segment start)
            (503.0, 504.0),          # dist = 5.0
        ]
        # Without handling missing frames, (3,4) -> (500,500) would add ~700 px of false leap!
        # With handle_missing=True (default), distance is 5.0 + 5.0 = 10.0.
        computed = actual_distance(pts_with_gap, handle_missing=True)
        self.assertAlmostEqual(computed, 10.0, places=3)

    def test_avoid_double_counting_static_points(self):
        """Verify that stationary hand or duplicate consecutive points add zero distance."""
        pts = [
            (10.0, 10.0),
            (10.0, 10.0),            # Identical (dt elapsed while stationary)
            (10.0, 10.0),            # Identical
            (13.0, 14.0),            # Movement: dist = 5.0
            (13.0, 14.0),            # Stationary
            (13.00000001, 14.0),     # Sub-epsilon noise (< min_step_px)
        ]
        computed = actual_distance(pts)
        self.assertAlmostEqual(computed, 5.0, places=3)

    def test_array_inputs(self):
        """Verify actual_distance works with 1D xs and ys arrays."""
        xs = np.array([0.0, 3.0, 8.0, 8.0])
        ys = np.array([0.0, 4.0, 4.0, 16.0])
        computed = actual_distance(xs, ys)
        self.assertAlmostEqual(computed, 22.0, places=3)

    def test_player_entity_distance_accumulation(self):
        """Verify Player entity tracks actual_distance and complete full_trajectory."""
        p = Player(start_x=100.0, start_y=100.0, radius=16)
        self.assertEqual(p.actual_distance, 0.0)
        self.assertEqual(len(p.full_trajectory), 1)

        # Move by dx=30, dy=40 -> distance = 50.0
        moved = p.move_to_cursor(130.0, 140.0, walls=[], min_move_threshold=0.0)
        self.assertTrue(moved)
        self.assertAlmostEqual(p.actual_distance, 50.0, places=1)
        self.assertAlmostEqual(p.px, 130.0, places=1)
        self.assertAlmostEqual(p.py, 140.0, places=1)

        # Move by dx=0, dy=50 -> distance = 50.0 (total = 100.0)
        p.move_to_cursor(130.0, 190.0, walls=[], min_move_threshold=0.0)
        self.assertAlmostEqual(p.actual_distance, 100.0, places=1)

        # Verify full trajectory retains all points without truncation
        traj = p.full_trajectory
        self.assertGreaterEqual(len(traj), 3)
        self.assertAlmostEqual(actual_distance(traj), 100.0, places=1)

        # Reset clears distance and trajectory
        p.reset(50.0, 50.0)
        self.assertEqual(p.actual_distance, 0.0)
        self.assertEqual(p.px, 50.0)
        self.assertEqual(p.py, 50.0)
        self.assertEqual(len(p.full_trajectory), 1)

    def test_game_engine_exposes_actual_distance(self):
        """Verify GameEngine exposes actual_distance and delegates to Player."""
        level = create_easy_level()
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])

        self.assertEqual(engine.actual_distance, 0.0)
        self.assertIsInstance(engine.actual_distance, float)
        self.assertIsInstance(engine.full_trajectory, list)

        # Start session
        engine.state = GameState.RUNNING
        # Move cursor
        start_x, start_y = float(level.start[0]), float(level.start[1])
        engine.update(cursor=(start_x + 30.0, start_y + 40.0), dt=0.016)

        # Player moved 50 px
        self.assertAlmostEqual(engine.actual_distance, 50.0, places=1)
        self.assertAlmostEqual(engine.player.actual_distance, 50.0, places=1)

        # Missing hand frame (cursor is None) -> player freezes, distance does NOT increase
        engine.update(cursor=None, dt=0.016)
        self.assertAlmostEqual(engine.actual_distance, 50.0, places=1)

    def test_metrics_collector_actual_distance(self):
        """Verify MetricsCollector exposes actual_distance and computes metrics accurately."""
        collector = MetricsCollector(start=(100, 100), end=(500, 500), min_path_distance=400.0)
        collector.start_recording()

        # Record samples
        collector.record(100.0, 100.0, dt=0.016)
        collector.record(103.0, 104.0, dt=0.016)   # dist = 5.0
        collector.record(103.0, 104.0, dt=0.016)   # stationary (0.0)
        collector.record(109.0, 112.0, dt=0.016)   # dist = 10.0 (total = 15.0)

        # Property access
        self.assertAlmostEqual(collector.actual_distance, 15.0, places=1)

        collector.stop_recording()
        summary = collector.compute()

        # Both 'actual_distance' and 'actual_distance_px' exposed
        self.assertIn("actual_distance", summary)
        self.assertIn("actual_distance_px", summary)
        self.assertAlmostEqual(summary["actual_distance"], 15.0, places=1)
        self.assertAlmostEqual(summary["actual_distance_px"], 15.0, places=1)
        self.assertAlmostEqual(summary["path_length_px"], 15.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
