"""
scratch/test_minimum_path.py — Comprehensive Test Suite for Minimum/Optimal Path Distance

Tests:
  1. Return value types and structure:
     - minimum_path_distance: float
     - optimal_waypoints: List[Tuple[float, float]]
  2. Physical solvability & collision-free guarantee:
     - Every point along every segment must have clearance >= player_radius.
     - 0 collisions with RectObstacles and PolygonObstacles.
  3. Non-triviality (Detour vs. Euclidean):
     - For mazes with blocking walls, minimum_path_distance > straight_line_distance.
  4. Specific known levels across all difficulties:
     - Easy-1: Gentle Flow
     - Easy-2: Conceptual Rehab Maze (navigating doorway entrance and inner hook)
     - Medium-1: Dual Pathways
     - Medium-3: Spiral (navigating through the spiral turns)
     - Hard-1: The Labyrinth (navigating switchbacks, chicanes, and cul-de-sacs)
     - Hard-4: Polygonal Corridors (navigating slanted polygon walls)
  5. Debug visual rendering verification:
     - Minimum path rendered strictly when debug_mode is True.
"""

from __future__ import annotations
import math
import os
import sys
import unittest

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from game.collision import resolve_against_obstacles
from game.levels import (
    create_easy_level,
    create_medium_level,
    create_hard_level,
    conceptual_level,
    easy_levels,
    medium_levels,
    hard_levels,
    Level,
)
from game.maze import MazeGenerator
from game.pathfinding import (
    calculate_minimum_path,
    minimum_path_distance,
    solve_level_path,
)
from ui.renderer import Renderer
from game.game_engine import GameEngine


class TestMinimumPathCalculation(unittest.TestCase):
    """Test suite for continuous geometric minimum path calculation."""

    def test_return_types_and_structure(self):
        """Verify return values are correct types: float distance and list of float 2D tuples."""
        level = create_easy_level()
        dist, waypoints = calculate_minimum_path(
            start=level.start,
            end=level.end,
            obstacles=level.walls,
            player_radius=18.0,
        )
        self.assertIsInstance(dist, float)
        self.assertGreater(dist, 0.0)
        self.assertIsInstance(waypoints, list)
        self.assertGreaterEqual(len(waypoints), 2)
        for pt in waypoints:
            self.assertEqual(len(pt), 2)
            self.assertIsInstance(pt[0], float)
            self.assertIsInstance(pt[1], float)

        # Helper accessor returns float
        dist_scalar = minimum_path_distance(level, player_radius=18.0)
        self.assertIsInstance(dist_scalar, float)
        self.assertAlmostEqual(dist, dist_scalar, places=1)

    def test_easy_levels_minimum_path(self):
        """Verify all Easy levels have valid, collision-free minimum paths."""
        mg = MazeGenerator()
        for idx in range(mg.level_count(1)):
            level = mg.get_level(1, idx)
            pr = float(config.DIFFICULTIES[1].player_radius)
            dist, waypoints = calculate_minimum_path(
                level.start, level.end, level.walls, player_radius=pr
            )
            self.assertFalse(math.isinf(dist), f"Easy level {level.name} reported infinite distance")
            self.assertGreaterEqual(len(waypoints), 2)
            self.verify_path_collision_free(waypoints, level.walls, pr, level.name)

    def test_medium_levels_minimum_path(self):
        """Verify all Medium levels have valid, collision-free minimum paths."""
        mg = MazeGenerator()
        for idx in range(mg.level_count(2)):
            level = mg.get_level(2, idx)
            pr = float(config.DIFFICULTIES[2].player_radius)
            dist, waypoints = calculate_minimum_path(
                level.start, level.end, level.walls, player_radius=pr
            )
            self.assertFalse(math.isinf(dist), f"Medium level {level.name} reported infinite distance")
            self.assertGreaterEqual(len(waypoints), 2)
            self.verify_path_collision_free(waypoints, level.walls, pr, level.name)

    def test_hard_levels_minimum_path(self):
        """Verify all Hard levels have valid, collision-free minimum paths."""
        mg = MazeGenerator()
        for idx in range(mg.level_count(3)):
            level = mg.get_level(3, idx)
            pr = float(config.DIFFICULTIES[3].player_radius)
            dist, waypoints = calculate_minimum_path(
                level.start, level.end, level.walls, player_radius=pr
            )
            self.assertFalse(math.isinf(dist), f"Hard level {level.name} reported infinite distance")
            self.assertGreaterEqual(len(waypoints), 2)
            self.verify_path_collision_free(waypoints, level.walls, pr, level.name)

    def test_conceptual_maze_navigates_doorway(self):
        """
        In the conceptual maze:
        START is outside, END is inside behind an obstacle hook.
        A straight line cuts directly through the enclosure wall.
        The minimum path MUST detour through the left doorway.
        """
        level = conceptual_level()
        pr = float(config.DIFFICULTIES[1].player_radius)
        euclidean = math.hypot(level.end[0] - level.start[0], level.end[1] - level.start[1])

        dist, waypoints = calculate_minimum_path(
            level.start, level.end, level.walls, player_radius=pr
        )
        self.assertGreater(dist, euclidean, "Conceptual maze min path should be strictly longer than straight line")
        # Must have at least 3 waypoints (START -> Entrance doorway -> around hook -> END)
        self.assertGreaterEqual(len(waypoints), 3)

        # Verify that the path physically passes through the doorway opening (x=231, y in [170, 310])
        door_crossed = False
        for i in range(1, len(waypoints)):
            p1, p2 = waypoints[i - 1], waypoints[i]
            if min(p1[0], p2[0]) <= 231 <= max(p1[0], p2[0]):
                t = (231.0 - p1[0]) / (p2[0] - p1[0])
                y_cross = p1[1] + (p2[1] - p1[1]) * t
                if 170.0 <= y_cross <= 310.0:
                    door_crossed = True
                    break
        self.assertTrue(door_crossed, f"Path waypoints {waypoints} did not cross through the doorway opening at x=231")
        self.verify_path_collision_free(waypoints, level.walls, pr, "Conceptual Rehab Maze")

    def test_spiral_maze_detour(self):
        """
        In Medium-3 Spiral, walls force an inward spiral.
        Straight-line Euclidean distance is short (~477 px).
        Minimum path must be significantly longer (> 1000 px).
        """
        mg = MazeGenerator()
        spiral = None
        for i in range(mg.level_count(2)):
            lvl = mg.get_level(2, i)
            if "Spiral" in lvl.name:
                spiral = lvl
                break
        self.assertIsNotNone(spiral)

        pr = float(config.DIFFICULTIES[2].player_radius)
        euclidean = math.hypot(spiral.end[0] - spiral.start[0], spiral.end[1] - spiral.start[1])
        dist, waypoints = calculate_minimum_path(
            spiral.start, spiral.end, spiral.walls, player_radius=pr
        )
        self.assertGreater(dist, euclidean * 2.0, "Spiral min path must be > 2x Euclidean distance")
        self.verify_path_collision_free(waypoints, spiral.walls, pr, "Medium-3: Spiral")

    def test_hard_labyrinth_switchbacks(self):
        """
        In Hard-1 The Labyrinth, 15 turns and baffles force a long route.
        Minimum path must be > 2000 px and pass cleanly through all switchbacks.
        """
        level = create_hard_level()
        pr = float(config.DIFFICULTIES[3].player_radius)
        euclidean = math.hypot(level.end[0] - level.start[0], level.end[1] - level.start[1])
        dist, waypoints = calculate_minimum_path(
            level.start, level.end, level.walls, player_radius=pr
        )
        self.assertGreater(dist, 2000.0, f"Labyrinth min path should be > 2000 px, got {dist:.1f}")
        self.assertGreaterEqual(len(waypoints), 12, "Labyrinth should have >= 12 waypoints")
        self.verify_path_collision_free(waypoints, level.walls, pr, "Hard-1: The Labyrinth")

    def test_polygonal_maze_clearance(self):
        """
        In Hard-4 Polygonal Corridors, obstacles are slanted geometric walls and central diamond polygon.
        Verify full clearance around polygonal obstacles.
        """
        mg = MazeGenerator()
        poly_lvl = None
        for i in range(mg.level_count(3)):
            lvl = mg.get_level(3, i)
            if "Polygonal" in lvl.name:
                poly_lvl = lvl
                break
        self.assertIsNotNone(poly_lvl)
        pr = float(config.DIFFICULTIES[3].player_radius)
        dist, waypoints = calculate_minimum_path(
            poly_lvl.start, poly_lvl.end, poly_lvl.walls, player_radius=pr
        )
        self.assertFalse(math.isinf(dist))
        self.verify_path_collision_free(waypoints, poly_lvl.walls, pr, "Hard-4: Polygonal Corridors")

    def verify_path_collision_free(self, waypoints, obstacles, player_radius, name):
        """Verify every continuous step along waypoints has clearance >= player_radius."""
        for i in range(1, len(waypoints)):
            p1, p2 = waypoints[i - 1], waypoints[i]
            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(1, int(seg_len // 2))
            for s in range(steps + 1):
                t = s / steps
                cx = p1[0] + (p2[0] - p1[0]) * t
                cy = p1[1] + (p2[1] - p1[1]) * t
                _, _, hit = resolve_against_obstacles(cx, cy, player_radius, obstacles)
                self.assertFalse(
                    hit,
                    f"Collision detected at ({cx:.1f}, {cy:.1f}) along path in {name} with radius {player_radius}",
                )

    def test_visual_debug_mode_toggle(self):
        """Verify that the minimum path is rendered when debug_mode=True and hidden when False."""
        renderer = Renderer()
        level = create_hard_level()
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[3])

        # Render without debug mode
        frame_normal = renderer.draw(engine, dt=0.016, debug_mode=False)
        # Render with debug mode
        frame_debug = renderer.draw(engine, dt=0.016, debug_mode=True)

        # The two frames MUST differ because debug overlay & minimum path waypoints are drawn
        diff = cv2.absdiff(frame_normal, frame_debug)
        changed_pixels = np.count_nonzero(diff)
        self.assertGreater(changed_pixels, 500, "Debug mode should visually render elements not present in normal mode")


if __name__ == "__main__":
    unittest.main(verbosity=2)
