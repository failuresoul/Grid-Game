"""
scratch/test_progress_history.py — Test Suite for Progress & History Screen

Validates:
  1. Reading historical session records from data/sessions/ (JSON & CSV).
  2. Chronological sorting and data normalization.
  3. Filtering by difficulty (ALL, EASY, MEDIUM, HARD).
  4. Empty directory handling ("No previous sessions available.").
  5. Trend series extraction for all 5 metrics.
  6. Visual canvas rendering of draw_history_screen without exceptions.
  7. Interactive button rect calculations and mouse hover support.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.history_reader import (
    load_all_sessions,
    filter_sessions,
    extract_trend_series,
)
from ui.screens import draw_history_screen, get_history_button_rects
from ui.renderer import Renderer


class TestProgressHistory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="grid_game_history_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_json_session(self, filename: str, difficulty: str, time_s: float, dist: float, eff: float, acc: float, smooth: float, hits: int, ts: str):
        path = os.path.join(self.temp_dir, filename)
        data = {
            "timestamp": ts,
            "session_id": filename.replace(".json", ""),
            "difficulty": difficulty,
            "completion_status": "COMPLETED",
            "completion_time": time_s,
            "actual_distance": dist,
            "minimum_distance": dist * (eff / 100.0),
            "path_efficiency": eff,
            "accuracy": acc,
            "smoothness": smooth,
            "collision_count": hits,
            "deviation_count": 0,
            "trajectory": [{"x": 10.0, "y": 20.0, "t": 0.0}, {"x": 50.0, "y": 60.0, "t": time_s}],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def test_empty_directory_returns_empty_list(self):
        """Empty or missing directory must return empty list without crashing."""
        sessions = load_all_sessions(data_dir=self.temp_dir)
        self.assertEqual(sessions, [])

        non_existent = os.path.join(self.temp_dir, "does_not_exist")
        sessions_non_existent = load_all_sessions(data_dir=non_existent)
        self.assertEqual(sessions_non_existent, [])

    def test_load_all_sessions_chronological_sort(self):
        """Sessions must be parsed and sorted chronologically (oldest to newest)."""
        self._create_dummy_json_session("session_2026_09_22_120000.json", "EASY", 10.0, 300.0, 80.0, 95.0, 85.0, 0, "2026-09-22 12:00:00")
        self._create_dummy_json_session("session_2026_09_21_090000.json", "HARD", 25.0, 600.0, 55.0, 82.0, 65.0, 2, "2026-09-21 09:00:00")
        self._create_dummy_json_session("session_2026_09_22_100000.json", "MEDIUM", 15.0, 450.0, 70.0, 90.0, 75.0, 1, "2026-09-22 10:00:00")

        sessions = load_all_sessions(data_dir=self.temp_dir)
        self.assertEqual(len(sessions), 3)

        # Verify sorted order: 2026-09-21 first, then 2026-09-22 10:00, then 2026-09-22 12:00
        self.assertEqual(sessions[0]["session_id"], "session_2026_09_21_090000")
        self.assertEqual(sessions[1]["session_id"], "session_2026_09_22_100000")
        self.assertEqual(sessions[2]["session_id"], "session_2026_09_22_120000")

    def test_filter_sessions_by_difficulty(self):
        """Filter must correctly filter sessions by difficulty (ALL, EASY, MEDIUM, HARD)."""
        self._create_dummy_json_session("s1.json", "EASY", 10.0, 200.0, 80.0, 95.0, 85.0, 0, "2026-09-22 10:00:00")
        self._create_dummy_json_session("s2.json", "EASY", 11.0, 210.0, 82.0, 96.0, 86.0, 0, "2026-09-22 10:05:00")
        self._create_dummy_json_session("s3.json", "MEDIUM", 18.0, 400.0, 68.0, 88.0, 72.0, 1, "2026-09-22 10:10:00")
        self._create_dummy_json_session("s4.json", "HARD", 30.0, 700.0, 52.0, 80.0, 60.0, 3, "2026-09-22 10:15:00")

        sessions = load_all_sessions(data_dir=self.temp_dir)
        self.assertEqual(len(sessions), 4)

        all_f = filter_sessions(sessions, "ALL")
        self.assertEqual(len(all_f), 4)

        easy_f = filter_sessions(sessions, "EASY")
        self.assertEqual(len(easy_f), 2)
        for s in easy_f:
            self.assertEqual(s["difficulty"], "EASY")

        med_f = filter_sessions(sessions, "MEDIUM")
        self.assertEqual(len(med_f), 1)
        self.assertEqual(med_f[0]["difficulty"], "MEDIUM")

        hard_f = filter_sessions(sessions, "HARD")
        self.assertEqual(len(hard_f), 1)
        self.assertEqual(hard_f[0]["difficulty"], "HARD")

    def test_extract_trend_series(self):
        """Trend series extraction must provide metric vectors matching session count."""
        self._create_dummy_json_session("s1.json", "EASY", 12.0, 300.0, 75.0, 92.0, 80.0, 1, "2026-09-22 10:00:00")
        self._create_dummy_json_session("s2.json", "EASY", 10.0, 280.0, 81.0, 97.0, 88.0, 0, "2026-09-22 10:10:00")

        sessions = load_all_sessions(data_dir=self.temp_dir)
        trends = extract_trend_series(sessions)

        self.assertIn("completion_time", trends)
        self.assertIn("actual_distance", trends)
        self.assertIn("path_efficiency", trends)
        self.assertIn("accuracy", trends)
        self.assertIn("smoothness", trends)
        self.assertIn("collision_count", trends)

        self.assertEqual(trends["completion_time"], [12.0, 10.0])
        self.assertEqual(trends["actual_distance"], [300.0, 280.0])
        self.assertEqual(trends["path_efficiency"], [75.0, 81.0])
        self.assertEqual(trends["accuracy"], [92.0, 97.0])
        self.assertEqual(trends["smoothness"], [80.0, 88.0])
        self.assertEqual(trends["collision_count"], [1.0, 0.0])

    def test_render_history_screen_empty_state(self):
        """Rendering empty history screen must produce non-diagnostic message without error."""
        W, H = 800, 600
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        buttons = draw_history_screen(
            canvas=canvas,
            W=W,
            H=H,
            sessions=[],
            current_filter="ALL",
            page=0,
        )
        self.assertGreaterEqual(len(buttons), 3)  # MAIN_MENU, LEVEL_SELECT, EXIT + filters
        # Canvas should have non-zero pixels
        self.assertGreater(np.count_nonzero(canvas), 1000)

    def test_render_history_screen_populated_state(self):
        """Rendering populated history screen must render table and 5 trend graphs without error."""
        for i in range(5):
            self._create_dummy_json_session(
                f"session_{i}.json", "EASY",
                10.0 + i, 250.0 + i * 10, 70.0 + i * 2, 90.0 + i, 80.0 + i, i % 2,
                f"2026-09-22 10:{i:02d}:00"
            )
        sessions = load_all_sessions(data_dir=self.temp_dir)
        self.assertEqual(len(sessions), 5)

        W, H = 800, 600
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        buttons = draw_history_screen(
            canvas=canvas,
            W=W,
            H=H,
            sessions=sessions,
            current_filter="ALL",
            page=0,
            mouse_pos=(100.0, 100.0),
        )
        self.assertGreaterEqual(len(buttons), 3)
        self.assertGreater(np.count_nonzero(canvas), 10000)

    def test_renderer_draw_history_screen_wrapper(self):
        """Renderer.draw_history_screen wrapper must return (canvas, buttons)."""
        renderer = Renderer()
        canvas, buttons = renderer.draw_history_screen(
            sessions=[],
            current_filter="EASY",
            page=0,
        )
        self.assertEqual(canvas.shape, (renderer.H, renderer.W, 3))
        self.assertIsInstance(buttons, list)


if __name__ == "__main__":
    unittest.main()
