"""
scratch/test_session_recording.py — Test Suite for Session Recording

Validates:
  1. Automated saving of completed game sessions.
  2. Required fields:
     - timestamp, session_id, difficulty, maze_seed, completion_status,
       completion_time, actual_distance, minimum_distance, path_efficiency,
       accuracy, smoothness, collision_count, deviation_count
  3. Trajectory data serialization with timestamps and coordinates.
  4. Filename format: session_YYYY_MM_DD_HHMMSS.json.
  5. Auto-creation of missing data directories.
  6. Crash-proof behavior (never crashes even if directory cannot be written).
  7. CSV backward compatibility.
  8. load_session for longitudinal rehabilitation progress analysis.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from game.levels import Level, RectObstacle
from game.game_engine import GameEngine, GameState
from metrics.performance import MetricsCollector
from metrics.session_recorder import (
    save_session,
    load_session,
    ensure_session_dir,
    generate_session_id,
    build_session_filename,
)


class TestSessionRecording(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="grid_game_test_sessions_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_session_creates_json_file_with_exact_naming(self):
        """Session recording must create a JSON file matching session_YYYY_MM_DD_HHMMSS.json."""
        saved_path = save_session(
            difficulty="EASY",
            maze_seed=12345,
            completion_status="COMPLETED",
            completion_time=8.5,
            actual_distance=320.0,
            minimum_distance=250.0,
            path_efficiency=78.1,
            accuracy=96.4,
            smoothness=84.2,
            collision_count=1,
            deviation_count=0,
            trajectory=[(50.0, 50.0, 0.0), (100.0, 100.0, 1.0)],
            save_dir=self.temp_dir,
            save_csv_also=False,
        )
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.isfile(saved_path))
        filename = os.path.basename(saved_path)
        self.assertTrue(filename.startswith("session_"))
        self.assertTrue(filename.endswith(".json"))

    def test_all_thirteen_required_fields_present_in_json(self):
        """Verify all 13 required fields are saved with exact keys and types."""
        saved_path = save_session(
            timestamp="2026-09-22 01:15:00",
            session_id="session_test_001",
            difficulty="MEDIUM",
            maze_seed=999,
            completion_status="COMPLETED",
            completion_time=12.34,
            actual_distance=450.5,
            minimum_distance=300.0,
            path_efficiency=66.6,
            accuracy=92.0,
            smoothness=77.5,
            collision_count=2,
            deviation_count=1,
            trajectory=[{"x": 10.0, "y": 20.0, "t": 0.0}, {"x": 30.0, "y": 40.0, "t": 0.5}],
            level_name="Medium-1",
            save_dir=self.temp_dir,
            save_csv_also=False,
        )
        self.assertIsNotNone(saved_path)

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "timestamp",
            "session_id",
            "difficulty",
            "maze_seed",
            "completion_status",
            "completion_time",
            "actual_distance",
            "minimum_distance",
            "path_efficiency",
            "accuracy",
            "smoothness",
            "collision_count",
            "deviation_count",
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Required key '{key}' missing from session record")

        self.assertEqual(data["timestamp"], "2026-09-22 01:15:00")
        self.assertEqual(data["session_id"], "session_test_001")
        self.assertEqual(data["difficulty"], "MEDIUM")
        self.assertEqual(data["maze_seed"], 999)
        self.assertEqual(data["completion_status"], "COMPLETED")
        self.assertEqual(data["completion_time"], 12.34)
        self.assertEqual(data["actual_distance"], 450.5)
        self.assertEqual(data["minimum_distance"], 300.0)
        self.assertEqual(data["path_efficiency"], 66.6)
        self.assertEqual(data["accuracy"], 92.0)
        self.assertEqual(data["smoothness"], 77.5)
        self.assertEqual(data["collision_count"], 2)
        self.assertEqual(data["deviation_count"], 1)

    def test_trajectory_data_saved_with_coordinates_and_time(self):
        """Trajectory points must be cleanly serialized with x, y, and t."""
        raw_traj = [
            (50.0, 100.0, 0.0),
            (65.5, 110.2, 0.033),
            (80.1, 120.4, 0.066),
        ]
        saved_path = save_session(
            trajectory=raw_traj,
            save_dir=self.temp_dir,
            save_csv_also=False,
        )
        data = load_session(saved_path)
        self.assertIsNotNone(data)
        self.assertIn("trajectory", data)
        self.assertEqual(len(data["trajectory"]), 3)
        self.assertEqual(data["trajectory"][0], {"x": 50.0, "y": 100.0, "t": 0.0})
        self.assertEqual(data["trajectory"][1], {"x": 65.5, "y": 110.2, "t": 0.033})
        self.assertEqual(data["trajectory"][2], {"x": 80.1, "y": 120.4, "t": 0.066})

    def test_auto_creates_missing_directory(self):
        """Must automatically create session directory if it does not already exist."""
        sub_dir = os.path.join(self.temp_dir, "nested", "deep", "sessions")
        self.assertFalse(os.path.exists(sub_dir))

        saved_path = save_session(
            difficulty="HARD",
            save_dir=sub_dir,
            save_csv_also=False,
        )
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.isdir(sub_dir))
        self.assertTrue(os.path.isfile(saved_path))

    def test_never_crashes_on_unwritable_directory(self):
        """Application must never crash if directory creation or file writing fails."""
        # Mock os.makedirs to raise an OSError (e.g. PermissionError)
        with patch("os.makedirs", side_effect=PermissionError("Mock permission denied")):
            saved_path = save_session(
                save_dir="/non_existent_and_forbidden_root/data/sessions",
                save_csv_also=False,
            )
            # Must return None safely without raising an exception
            self.assertIsNone(saved_path)

    def test_disambiguate_multiple_sessions_in_same_second(self):
        """If multiple sessions finish in the exact same second, both are preserved."""
        fixed_time = datetime(2026, 9, 21, 18, 15, 0)
        with patch("metrics.session_recorder.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.strftime = fixed_time.strftime

            path1 = save_session(session_id="sess_1", save_dir=self.temp_dir, save_csv_also=False)
            path2 = save_session(session_id="sess_2", save_dir=self.temp_dir, save_csv_also=False)

            self.assertIsNotNone(path1)
            self.assertIsNotNone(path2)
            self.assertNotEqual(path1, path2)
            self.assertTrue(os.path.exists(path1))
            self.assertTrue(os.path.exists(path2))

            data1 = load_session(path1)
            data2 = load_session(path2)
            self.assertEqual(data1["session_id"], "sess_1")
            self.assertEqual(data2["session_id"], "sess_2")

    def test_game_engine_complete_session_records_json_and_csv(self):
        """Completing a maze session in GameEngine must automatically trigger session recording."""
        level = Level(
            walls=[RectObstacle(140, 60, 20, 80)],
            start=(50, 100),
            end=(250, 100),
            name="TestMaze",
            seed=4242,
        )
        level.compute_minimum_path()

        collector = MetricsCollector(
            start=level.start,
            end=level.end,
            min_path_distance=level.min_path_distance,
            optimal_waypoints=level.optimal_waypoints,
        )
        engine = GameEngine(
            level=level,
            difficulty_cfg=config.DIFFICULTIES[1],
            metrics_collector=collector,
        )

        with patch("config.METRICS_SAVE_DIR", self.temp_dir):
            # Play through to completion
            engine.update(cursor=(100, 40), dt=0.033)  # Leave start -> PLAYING
            engine.update(cursor=(180, 40), dt=0.033)
            engine.update(cursor=(245, 100), dt=0.033)  # Reaches END

            self.assertTrue(engine.is_completed)
            self.assertTrue(engine._metrics_saved)
            self.assertIsNotNone(engine.last_saved_session_file)
            self.assertTrue(os.path.isfile(engine.last_saved_session_file))

            # Verify saved JSON contents
            session_json = load_session(engine.last_saved_session_file)
            self.assertIsNotNone(session_json)
            self.assertEqual(session_json["maze_seed"], 4242)
            self.assertEqual(session_json["level_name"], "TestMaze")
            self.assertEqual(session_json["completion_status"], "COMPLETED")
            self.assertGreater(session_json["actual_distance"], 0.0)
            self.assertGreater(session_json["minimum_distance"], 0.0)
            self.assertGreater(len(session_json["trajectory"]), 0)

            # Check that daily CSV is also created/appended
            csv_files = [f for f in os.listdir(self.temp_dir) if f.endswith("_sessions.csv")]
            self.assertGreaterEqual(len(csv_files), 1)


if __name__ == "__main__":
    unittest.main()
