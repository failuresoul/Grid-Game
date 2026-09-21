"""
scratch/test_trajectory_accuracy.py — Comprehensive Test Suite for Trajectory Accuracy & Path Deviation

Validates:
1. Point-to-segment & Point-to-polyline orthogonal distance calculations.
2. Mathematical Formula:
   Accuracy = (100 / N) * sum( exp(-d_i^2 / (2 * sigma^2)) )
   - Straight route tracking -> 100.0%
   - Off-course detour -> low accuracy despite reaching END.
3. Separation of 5 Core Dimensions:
   - Accuracy (trajectory_accuracy, mean_path_deviation, deviation_events)
   - Path Efficiency (path_efficiency)
   - Distance (actual_distance, minimum_distance)
   - Collision Count (wall_hits)
   - Time (completion_time_s, time_outside_route_s)
4. Intended Route Corridor & Deviation Events:
   - Time spent outside route corridor
   - Count of distinct excursion / wrong movement events
5. Exposed Properties & Summary Dict:
   - MetricsCollector
   - GameEngine
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
import numpy as np

from metrics.accuracy import (
    point_to_segment_distance,
    point_to_polyline_distance,
    trajectory_deviations,
    path_deviation_summary,
    route_corridor_metrics,
    trajectory_accuracy_percentage,
)
from metrics.performance import MetricsCollector
from game.levels import create_easy_level
from game.game_engine import GameEngine, GameState
import config


class TestTrajectoryAccuracy(unittest.TestCase):
    """Test suite for trajectory accuracy, deviation kinematics, and metric separation."""

    def test_point_to_segment_distance(self):
        """Verify orthogonal projection and segment endpoint clamping."""
        # Segment from (0, 0) to (100, 0)
        # Point directly above midpoint
        self.assertAlmostEqual(point_to_segment_distance(50.0, 30.0, 0.0, 0.0, 100.0, 0.0), 30.0)
        # Point on the segment
        self.assertAlmostEqual(point_to_segment_distance(25.0, 0.0, 0.0, 0.0, 100.0, 0.0), 0.0)
        # Point past B (clamped to B)
        self.assertAlmostEqual(point_to_segment_distance(140.0, 30.0, 0.0, 0.0, 100.0, 0.0), 50.0)
        # Point before A (clamped to A)
        self.assertAlmostEqual(point_to_segment_distance(-30.0, 40.0, 0.0, 0.0, 100.0, 0.0), 50.0)

    def test_point_to_polyline_distance(self):
        """Verify distance to multi-segment L-shaped route."""
        # Route: (0, 0) -> (100, 0) -> (100, 100)
        route = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        # Along segment 1
        self.assertAlmostEqual(point_to_polyline_distance(50.0, 10.0, route), 10.0)
        # Along segment 2
        self.assertAlmostEqual(point_to_polyline_distance(120.0, 50.0, route), 20.0)
        # At vertex corner
        self.assertAlmostEqual(point_to_polyline_distance(100.0, 0.0, route), 0.0)

    def test_reaching_end_does_not_equal_perfect_accuracy(self):
        """Crucial Requirement: Reaching the END while making unnecessary movements
        must produce low accuracy, high deviation, and deviation events.
        """
        route = [(100.0, 100.0), (500.0, 100.0)]  # Straight 400px route

        # Case A: Accurate direct path from start to end
        xs_direct = np.linspace(100.0, 500.0, 41)
        ys_direct = np.full_like(xs_direct, 100.0)
        ts_direct = np.linspace(0.0, 4.0, 41)

        acc_direct = trajectory_accuracy_percentage(xs_direct, ys_direct, route, corridor_tolerance_px=40.0)
        summary_direct = path_deviation_summary(xs_direct, ys_direct, route)
        corridor_direct = route_corridor_metrics(xs_direct, ys_direct, ts_direct, route, corridor_tolerance_px=40.0)

        self.assertEqual(acc_direct, 100.0)
        self.assertEqual(summary_direct["mean_path_deviation"], 0.0)
        self.assertEqual(summary_direct["max_path_deviation"], 0.0)
        self.assertEqual(corridor_direct["deviation_events"], 0)
        self.assertEqual(corridor_direct["time_outside_route_s"], 0.0)

        # Case B: Meandering detour that wanders 150px off-course and back, but still reaches (500, 100)
        xs_wander = np.linspace(100.0, 500.0, 41)
        # Sine wave excursion: peak deviation = 150 px
        ys_wander = 100.0 + 150.0 * np.sin(np.pi * (xs_wander - 100.0) / 400.0)
        ts_wander = np.linspace(0.0, 8.0, 41)

        acc_wander = trajectory_accuracy_percentage(xs_wander, ys_wander, route, corridor_tolerance_px=40.0)
        summary_wander = path_deviation_summary(xs_wander, ys_wander, route)
        corridor_wander = route_corridor_metrics(xs_wander, ys_wander, ts_wander, route, corridor_tolerance_px=40.0)

        # Reached END, but accuracy is substantially degraded!
        self.assertLess(acc_wander, 50.0)
        self.assertGreater(summary_wander["mean_path_deviation"], 50.0)
        self.assertGreaterEqual(summary_wander["max_path_deviation"], 140.0)
        self.assertGreater(corridor_wander["deviation_events"], 0)
        self.assertGreater(corridor_wander["time_outside_route_s"], 2.0)

    def test_deviation_events_and_time_outside(self):
        """Verify multiple wrong movement episodes are counted correctly."""
        route = [(0.0, 0.0), (1000.0, 0.0)]
        # 3 excursions outside corridor (> 40 px)
        # times: 0.0 to 10.0 seconds
        xs = np.linspace(0.0, 1000.0, 101)
        ys = np.zeros(101)
        ts = np.linspace(0.0, 10.0, 101)

        # Excursion 1: indices 15-25 (y = 80 px, dt = 1.0 s)
        ys[15:25] = 80.0
        # Excursion 2: indices 45-55 (y = 90 px, dt = 1.0 s)
        ys[45:55] = 90.0
        # Excursion 3: indices 75-85 (y = 70 px, dt = 1.0 s)
        ys[75:85] = 70.0

        res = route_corridor_metrics(xs, ys, ts, route, corridor_tolerance_px=40.0)
        self.assertEqual(res["deviation_events"], 3)
        self.assertAlmostEqual(res["time_outside_route_s"], 3.0, places=1)
        self.assertAlmostEqual(res["pct_time_outside"], 30.0, places=1)
        self.assertAlmostEqual(res["corridor_adherence_pct"], 70.0, places=1)

    def test_metrics_collector_accuracy_integration(self):
        """Verify MetricsCollector computes and exposes accuracy metrics."""
        start = (100, 100)
        end = (500, 100)
        waypoints = [(100.0, 100.0), (500.0, 100.0)]

        collector = MetricsCollector(
            start=start,
            end=end,
            min_path_distance=400.0,
            optimal_waypoints=waypoints,
            corridor_tolerance_px=40.0,
        )

        collector.start_recording()
        # Move along route with a small 20px offset (within 40px corridor)
        for i in range(11):
            collector.record(100.0 + i * 40.0, 120.0, 0.1)
        collector.stop_recording()

        # Properties
        self.assertAlmostEqual(collector.mean_path_deviation, 20.0, places=1)
        self.assertAlmostEqual(collector.max_path_deviation, 20.0, places=1)
        self.assertEqual(collector.deviation_events, 0)
        self.assertEqual(collector.time_outside_route_s, 0.0)
        # Exp(-20^2 / (2*40^2)) = exp(-0.125) ≈ 88.2%
        self.assertAlmostEqual(collector.trajectory_accuracy, 88.2, places=1)

        # compute() dictionary check
        summary = collector.compute()
        self.assertIn("trajectory_accuracy", summary)
        self.assertIn("mean_path_deviation_px", summary)
        self.assertIn("max_path_deviation_px", summary)
        self.assertIn("time_outside_route_s", summary)
        self.assertIn("deviation_events", summary)
        self.assertEqual(summary["deviation_events"], 0)

    def test_separation_of_five_dimensions(self):
        """Verify the 5 dimensions remain strictly distinct in collector.compute():
        1. Accuracy
        2. Path Efficiency
        3. Distance
        4. Collision Count
        5. Time
        """
        collector = MetricsCollector(start=(0, 0), end=(100, 0), min_path_distance=100.0)
        collector.start_recording()
        collector.record(0.0, 0.0, 0.0)
        collector.record(100.0, 0.0, 2.0)
        collector.stop_recording()

        summary = collector.compute()

        # 1. Accuracy
        self.assertIn("trajectory_accuracy", summary)
        self.assertIn("mean_path_deviation_px", summary)
        self.assertIn("deviation_events", summary)

        # 2. Path Efficiency
        self.assertIn("path_efficiency", summary)

        # 3. Distance
        self.assertIn("actual_distance", summary)
        self.assertIn("minimum_distance", summary)

        # 4. Collision Count (passed separately via engine / wall_hits)
        self.assertNotIn("wall_collisions", summary)  # engine tracks wall_hit_count independently

        # 5. Time
        self.assertIn("completion_time_s", summary)
        self.assertIn("time_outside_route_s", summary)

    def test_game_engine_exposed_accuracy_properties(self):
        """Verify GameEngine exposes accuracy properties."""
        level = create_easy_level()
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])

        self.assertIsInstance(engine.trajectory_accuracy, float)
        self.assertIsInstance(engine.mean_path_deviation, float)
        self.assertIsInstance(engine.time_outside_route_s, float)
        self.assertIsInstance(engine.deviation_events, int)
        self.assertIsInstance(engine.wall_hit_count, int)


if __name__ == "__main__":
    unittest.main()
