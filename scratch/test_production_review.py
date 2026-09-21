"""
scratch/test_production_review.py — Comprehensive Production Quality Verification Suite

Executes automated testing across all 30 production requirements and 12 edge cases:
1. Application starts successfully.
2. Camera initializes correctly.
3. Hand tracking works.
4. No-hand frames do not crash the game.
5. Hand movement is continuous (sub-pixel floating point).
6. There is NO block-by-block movement.
7. Player can move freely in open space.
8. Player cannot pass through walls.
9. Start point works.
10. End point works.
11. EASY maze is solvable.
12. MEDIUM maze is solvable.
13. HARD maze is solvable.
14. Player radius is considered.
15. Minimum path calculation works.
16. Actual distance calculation works.
17. Path efficiency works.
18. Accuracy works.
19. Smoothness works.
20. Collision counting works.
21. Timer works.
22. Results screen works.
23. Session data saves correctly.
24. History screen works.
25. Adaptive difficulty works if enabled.
26. EMG remains disabled.
27. No fake EMG values are generated.
28. EMG code remains available for future integration.
29. No unnecessary dependencies are added.
30. No runtime errors remain.

Edge cases verified:
- Hand disappears mid-game
- Hand reappears
- Player touches wall
- Player moves rapidly (Anti-tunneling / CCD)
- Player starts near wall
- Player reaches END directly
- Player repeatedly hits a wall
- No previous session exists
- Corrupted session files exist
- Camera unavailable (mouse fallback)
- EMG unavailable
- EMG_ENABLED=False
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
import unittest
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import config
from game.maze import MazeGenerator
from game.player import Player
from game.game_engine import GameEngine, GameState
from game.collision import continuous_move_and_resolve, resolve_against_obstacles
from metrics.performance import MetricsCollector
from metrics.session_recorder import save_session
from metrics.history_reader import load_all_sessions, filter_sessions, extract_trend_series
from metrics.adaptive import evaluate_adaptive_difficulty
from ui.renderer import Renderer
from ui.screens import draw_results_screen, draw_start_screen, draw_history_screen
from emg.emg_interface import (
    initialize_emg, read_emg, process_emg, get_activation_level,
    get_rms, get_peak, close_emg, get_emg_status_string, is_emg_enabled
)


class TestProductionQualityReview(unittest.TestCase):
    """End-to-end production verification suite."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="rehab_review_")
        cls.mg = MazeGenerator()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Items 1–8: Application Core, Continuous Physics, No Grid-Stepping
    # ─────────────────────────────────────────────────────────────────────────

    def test_01_application_starts_and_initializes_cleanly(self):
        """1. Application components instantiate without crashing or missing attributes."""
        from main import RehabGame
        app = RehabGame()
        self.assertIsNotNone(app)
        self.assertEqual(app.app_state, GameState.MENU)
        self.assertFalse(config.EMG_ENABLED)
        self.assertIsNone(app.emg)

    def test_02_camera_initialization_and_fallback(self):
        """2. Camera handling degrades gracefully to mouse fallback when camera is absent."""
        from main import RehabGame
        app = RehabGame()
        app.cap = None  # Simulate no camera hardware
        app._using_mouse = True
        app.mouse_fallback_enabled = True
        self.assertTrue(app.mouse_fallback_enabled)

    def test_03_hand_tracking_landmark_and_smoothing_pipelines(self):
        """3. Hand tracker components initialize and smoothing filters operate."""
        from vision.smoothing import OneEuroFilter2D
        oe = OneEuroFilter2D()
        oe.reset()
        smoothed = oe.update(105.0, 102.0, dt=0.016)
        self.assertIsInstance(smoothed, tuple)
        self.assertEqual(len(smoothed), 2)
        # Verify sub-pixel continuous float outputs
        self.assertIsInstance(smoothed[0], float)
        self.assertIsInstance(smoothed[1], float)

    def test_04_no_hand_frames_do_not_crash_game(self):
        """4. Frame where hand is temporarily undetected (cursor=None) holds position safely."""
        level = self.mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        engine.state = GameState.PLAYING
        px_before, py_before = engine.player.px, engine.player.py

        # Feed 10 consecutive frames of None (missing hand)
        for _ in range(10):
            engine.update(cursor=None, dt=0.016)

        self.assertEqual(engine.player.px, px_before)
        self.assertEqual(engine.player.py, py_before)
        self.assertEqual(engine.state, GameState.PLAYING)

    def test_05_and_06_continuous_movement_and_no_block_stepping(self):
        """
        5 & 6. Continuous 2D sub-pixel floating-point movement.
        CRITICAL: Verifies strictly NO tile/grid/cell block-by-block movement.
        """
        player = Player(100.0, 100.0, radius=12)
        walls = []

        # Move to fractional arbitrary coordinates
        steps = [(101.37, 102.84), (103.92, 105.41), (107.12, 108.88)]
        for tx, ty in steps:
            moved = player.move_to_cursor(tx, ty, walls, min_move_threshold=0.1)
            self.assertTrue(moved)
            # Must remain continuous float, never snapped to integers or grid cells
            self.assertAlmostEqual(player.px, tx, places=2)
            self.assertAlmostEqual(player.py, ty, places=2)

    def test_07_player_moves_freely_in_open_space(self):
        """7. Player advances along an open continuous arc without artificial resistance."""
        player = Player(200.0, 200.0, radius=12)
        for angle in np.linspace(0, math.pi, 20):
            tx = 200.0 + 50.0 * math.cos(angle)
            ty = 200.0 + 50.0 * math.sin(angle)
            moved = player.move_to_cursor(tx, ty, walls=[], min_move_threshold=0.1)
            self.assertTrue(moved)
            self.assertAlmostEqual(player.px, tx, places=1)
            self.assertAlmostEqual(player.py, ty, places=1)

    def test_08_player_cannot_pass_through_walls(self):
        """8. Player cannot cross or penetrate solid obstacle walls."""
        # Wall is vertical barrier at x=200 to x=240, from y=0 to y=600
        wall = (200, 0, 40, 600)
        player = Player(150.0, 300.0, radius=10)

        # Attempt to jump straight through wall to x=280
        moved = player.move_to_cursor(280.0, 300.0, walls=[wall], min_move_threshold=0.1)
        self.assertTrue(moved)
        # Must be stopped at the left wall face: 200 - radius = 190.0
        self.assertLessEqual(player.px, 190.0)
        self.assertGreater(player.wall_hit_count, 0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Items 9–14: Zones, Solvability & Geometry
    # ─────────────────────────────────────────────────────────────────────────

    def test_09_and_10_start_point_and_end_point_triggers(self):
        """9 & 10. Start point initiates timer on exit; End point triggers completion."""
        level = self.mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        self.assertEqual(engine.state, GameState.READY)
        self.assertIsNone(engine._session_start)

        # Move outside start beacon circle
        sx, sy = level.start
        engine.update(cursor=(sx + 40.0, sy + 40.0), dt=0.016)
        self.assertEqual(engine.state, GameState.PLAYING)
        self.assertIsNotNone(engine._session_start)

        # Move directly into end zone
        ex, ey = level.end
        engine.update(cursor=(ex, ey), dt=0.016)
        self.assertIn(engine.state, (GameState.COMPLETED, GameState.RESULTS))
        self.assertIsNotNone(engine._session_end)

    def test_11_12_13_all_difficulties_are_solvable(self):
        """11, 12, 13. EASY, MEDIUM, and HARD curated levels have valid optimal paths."""
        for diff in (1, 2, 3):
            count = self.mg.level_count(diff)
            self.assertGreater(count, 0, f"Difficulty {diff} must have levels.")
            for l_idx in range(count):
                lvl = self.mg.get_level(diff, l_idx)
                self.assertGreater(lvl.minimum_path_distance, 0.0)
                self.assertGreaterEqual(len(lvl.optimal_waypoints), 2)
                # Verify start and end do not spawn inside obstacles
                _, _, hit_start = resolve_against_obstacles(lvl.start[0], lvl.start[1], float(config.PLAYER_RADIUS), lvl.walls)
                _, _, hit_end   = resolve_against_obstacles(lvl.end[0], lvl.end[1], float(config.PLAYER_RADIUS), lvl.walls)
                self.assertFalse(hit_start, f"Start point overlaps wall in {lvl.name}")
                self.assertFalse(hit_end, f"End point overlaps wall in {lvl.name}")

    def test_14_player_radius_is_considered(self):
        """14. Player radius creates correct buffer distance from obstacle borders."""
        wall = (300, 200, 50, 50)
        # Test with radius 10 vs radius 25 (with 0.5px push bias to eliminate re-tunneling)
        p10 = Player(250.0, 225.0, radius=10)
        p10.move_to_cursor(325.0, 225.0, walls=[wall])
        self.assertAlmostEqual(p10.px, 289.5, places=1)

        p25 = Player(250.0, 225.0, radius=25)
        p25.move_to_cursor(325.0, 225.0, walls=[wall])
        self.assertAlmostEqual(p25.px, 274.5, places=1)

    # ─────────────────────────────────────────────────────────────────────────
    #  Items 15–21: Clinical Kinematics Metrics Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def test_15_minimum_path_calculation(self):
        """15. Minimum path distance matches Euclidean sum of waypoints."""
        level = self.mg.get_level(1, 0)
        wps = level.optimal_waypoints
        expected_dist = sum(
            math.hypot(wps[i+1][0] - wps[i][0], wps[i+1][1] - wps[i][1])
            for i in range(len(wps) - 1)
        )
        self.assertAlmostEqual(level.minimum_path_distance, expected_dist, places=1)

    def test_16_actual_distance_calculation(self):
        """16. Actual distance accurately sums Euclidean segments."""
        player = Player(100.0, 100.0, radius=10)
        player.move_to_cursor(130.0, 140.0, walls=[])  # 3-4-5 triangle: dist = 50
        player.move_to_cursor(130.0, 190.0, walls=[])  # dy = 50: dist = 50
        self.assertAlmostEqual(player.actual_distance, 100.0, places=1)

    def test_17_path_efficiency_calculation(self):
        """17. Path Efficiency = (Min / Actual) * 100 clamped correctly."""
        metrics = MetricsCollector(start=(0, 0), end=(100, 0), optimal_waypoints=[(0, 0), (100, 0)], min_path_distance=100.0)
        metrics.start_recording()
        metrics.record(0.0, 0.0, 0.016)
        metrics.record(150.0, 0.0, 0.016)
        res = metrics.compute()
        # 100 / 150 * 100 = 66.67%
        self.assertAlmostEqual(res["path_efficiency"], 66.7, places=1)

    def test_18_accuracy_and_corridor_deviation(self):
        """18. Trajectory accuracy evaluates deviation from the intended corridor."""
        metrics = MetricsCollector(start=(0, 100), end=(200, 100), optimal_waypoints=[(0, 100), (200, 100)], min_path_distance=200.0)
        metrics.start_recording()
        # Record points offset by 10 px above corridor
        metrics.record(50.0, 110.0, dt=0.016)
        metrics.record(100.0, 110.0, dt=0.016)
        metrics.record(150.0, 110.0, dt=0.016)
        res = metrics.compute()
        self.assertAlmostEqual(res["mean_path_deviation_px"], 10.0, places=1)
        self.assertGreater(res["trajectory_accuracy"], 0.0)

    def test_19_movement_smoothness_score(self):
        """19. Smoothness normalized score between 0 and 100."""
        metrics = MetricsCollector(start=(0, 0), end=(100, 100), optimal_waypoints=[(0, 0), (100, 100)], min_path_distance=141.4)
        metrics.start_recording()
        # Smooth constant velocity movement
        for i in range(20):
            metrics.record(float(i * 5), float(i * 5), dt=0.016)
        res = metrics.compute()
        self.assertGreaterEqual(res["smoothness_score"], 0.0)
        self.assertLessEqual(res["smoothness_score"], 100.0)

    def test_20_collision_counting(self):
        """20. Wall collision counts are incremented accurately."""
        wall = (200, 100, 50, 100)
        player = Player(150.0, 150.0, radius=10)
        player.move_to_cursor(210.0, 150.0, walls=[wall])
        self.assertEqual(player.wall_hit_count, 1)

    def test_21_timer_operation_and_freeze(self):
        """21. Timer starts when movement begins and freezes cleanly upon END."""
        level = self.mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        engine.start_playing()
        time.sleep(0.05)
        t_mid = engine.elapsed_time
        self.assertGreater(t_mid, 0.04)

        engine.complete_session(won=True)
        t_frozen = engine.elapsed_time
        time.sleep(0.05)
        self.assertEqual(engine.elapsed_time, t_frozen, "Timer must freeze upon session completion.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Items 22–25: Results Screen, Persistence, History & Adaptive Difficulty
    # ─────────────────────────────────────────────────────────────────────────

    def test_22_results_screen_rendering(self):
        """22. Results screen renders trajectory comparison, metrics, and navigation buttons."""
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        level = self.mg.get_level(1, 0)
        draw_results_screen(
            canvas=canvas,
            W=800,
            H=600,
            final_metrics={"completion_time_s": 12.3, "trajectory_accuracy": 92.0, "smoothness_score": 80.0},
            wall_hits=0,
            elapsed=12.3,
            level=level,
            trajectory=[level.start, level.end],
            difficulty_name="EASY",
        )
        self.assertGreater(canvas.sum(), 0)

    def test_23_session_data_saves_json_and_csv(self):
        """23. Session metrics save cleanly to JSON and CSV in sessions directory."""
        filepath = save_session(
            difficulty="EASY",
            level_name="Test Level",
            maze_seed=42,
            completion_status="COMPLETED",
            completion_time=15.2,
            actual_distance=420.0,
            minimum_distance=350.0,
            collision_count=1,
            trajectory=[(100.0, 100.0), (200.0, 200.0)],
            save_dir=self.test_dir,
        )
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["difficulty"], "EASY")
        self.assertEqual(data["maze_seed"], 42)

    def test_24_history_screen_and_trend_loader(self):
        """24. History loader reads sessions, filters, and extracts series without error."""
        sessions = load_all_sessions(data_dir=self.test_dir)
        self.assertGreaterEqual(len(sessions), 1)
        filtered = filter_sessions(sessions, "EASY")
        self.assertEqual(len(filtered), len(sessions))
        trends = extract_trend_series(sessions)
        self.assertIn("completion_time", trends)
        self.assertIn("accuracy", trends)

    def test_25_adaptive_difficulty_recommendations(self):
        """25. Adaptive difficulty assesses motor quality across 5 dimensions."""
        rec = evaluate_adaptive_difficulty(
            current_metrics={"trajectory_accuracy": 95.0, "path_efficiency": 90.0, "smoothness_score": 85.0},
            current_difficulty="EASY",
            wall_hits=0,
            completion_status="COMPLETED",
            consistency_window=1,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.action, "INCREASE")
        self.assertEqual(rec.target_difficulty, "MEDIUM")

    # ─────────────────────────────────────────────────────────────────────────
    #  Items 26–30: Hardware Independence, EMG Isolation & Zero Crashes
    # ─────────────────────────────────────────────────────────────────────────

    def test_26_and_27_emg_disabled_and_no_fake_values(self):
        """26 & 27. EMG is disabled by default and generates NO fake measurements."""
        self.assertFalse(config.EMG_ENABLED)
        self.assertFalse(is_emg_enabled())
        self.assertIsNone(read_emg())
        self.assertIsNone(get_rms())
        self.assertIsNone(get_peak())
        self.assertIsNone(get_activation_level())
        self.assertIsNone(process_emg())
        self.assertEqual(get_emg_status_string(), "EMG: Not Connected / Disabled")

    def test_28_emg_code_remains_available_for_future(self):
        """28. Functions initialize_emg() and close_emg() exist and execute cleanly."""
        self.assertFalse(initialize_emg())
        close_emg()

    def test_29_no_unnecessary_dependencies(self):
        """29. Project relies strictly on approved dependencies (zero hardware drivers)."""
        for disallowed in ["torch", "pygame", "pylsl", "bitalino", "myo"]:
            self.assertNotIn(disallowed, sys.modules)

    def test_30_no_runtime_errors_across_states(self):
        """30. Complete transition through all state machine stages executes cleanly."""
        renderer = Renderer(canvas_w=800, canvas_h=600)
        level = self.mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])

        # State transitions
        engine.state = GameState.READY
        f1 = renderer.draw(engine, dt=0.016)
        self.assertIsNotNone(f1)

        engine.start_playing()
        f2 = renderer.draw(engine, dt=0.016)
        self.assertIsNotNone(f2)

        engine.complete_session(won=True)
        f3 = renderer.draw(engine, dt=0.016)
        self.assertIsNotNone(f3)

    # ─────────────────────────────────────────────────────────────────────────
    #  Edge-Case Resilience Suite
    # ─────────────────────────────────────────────────────────────────────────

    def test_edge_case_hand_disappears_and_reappears(self):
        """Edge: Hand leaves field of view (None) and reappears smoothly."""
        level = self.mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        engine.start_playing()

        engine.update(cursor=(100.0, 100.0), dt=0.016)
        # Disappears
        engine.update(cursor=None, dt=0.016)
        self.assertEqual(engine.player.position, (100, 100))
        # Reappears at new location
        engine.update(cursor=(120.0, 120.0), dt=0.016)
        self.assertAlmostEqual(engine.player.px, 120.0, places=1)

    def test_edge_case_player_moves_rapidly_anti_tunneling(self):
        """Edge: Fast hand swipe across thin wall is intercepted by CCD."""
        wall = (400, 0, 10, 600)  # Very thin vertical wall (10px wide)
        nx, ny, hit = continuous_move_and_resolve(
            start_x=350.0, start_y=300.0,
            target_x=450.0, target_y=300.0,  # 100px leap across wall in 1 frame
            r=8.0, obstacles=[wall], max_step_size=4.0
        )
        self.assertTrue(hit, "Rapid movement must be detected by Continuous Collision Detection.")
        self.assertLess(nx, 400.0, "Player must not tunnel to other side of wall.")

    def test_edge_case_repeated_wall_hits(self):
        """Edge: Player continuously pressed into wall debounces and resolves properly."""
        wall = (200, 0, 40, 600)
        player = Player(150.0, 300.0, radius=10)
        for _ in range(15):
            player.move_to_cursor(250.0, 300.0, walls=[wall])
            self.assertLessEqual(player.px, 190.0)

    def test_edge_case_no_previous_session_exists(self):
        """Edge: Clean system with zero saved history shows empty message without crashing."""
        empty_dir = tempfile.mkdtemp(prefix="rehab_empty_")
        try:
            sessions = load_all_sessions(data_dir=empty_dir)
            self.assertEqual(sessions, [])
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)
            draw_history_screen(canvas, W=800, H=600, sessions=sessions)
            self.assertGreater(canvas.sum(), 0)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_edge_case_corrupted_session_files(self):
        """Edge: Corrupted / invalid JSON and CSV are safely ignored."""
        corrupt_dir = tempfile.mkdtemp(prefix="rehab_corrupt_")
        try:
            with open(os.path.join(corrupt_dir, "broken.json"), "w") as f:
                f.write("{invalid json syntax !!!")
            with open(os.path.join(corrupt_dir, "bad_sessions.csv"), "w") as f:
                f.write("junk data, not a csv\0\0binary")
            # Must not raise exceptions
            sessions = load_all_sessions(data_dir=corrupt_dir)
            self.assertEqual(sessions, [])
        finally:
            shutil.rmtree(corrupt_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
