"""
scratch/test_game_state_machine.py

Comprehensive automated tests verifying the complete game state machine:
    MENU -> LEVEL_SELECT -> READY -> PLAYING -> COMPLETED -> RESULTS

Requirements:
- During PLAYING:
    - Timer starts when movement/game actually begins (cursor moves beyond START zone).
    - Record elapsed time.
    - Record trajectory.
    - Calculate distance.
    - Detect collisions.
    - Track performance.
- When player reaches END:
    - Stop timer immediately.
    - Timer must NOT continue running after completion.
    - Finalize metrics.
    - Save session.
    - Transition to RESULTS.
- Restart option:
    - Resets player position, clears trajectory, zeroes timers, and returns to READY.
"""

import math
import time
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from game.levels import Level
from game.player import Player
from game.game_engine import GameEngine, GameState
from metrics.performance import MetricsCollector


class TestGameStateMachine(unittest.TestCase):

    def setUp(self):
        # Create a simple straight corridor level: START at (50, 100), END at (250, 100)
        # Wall at top and bottom, obstacle at (150, 100)
        self.level = Level(
            name="TestCorridor",
            difficulty_tag="EASY",
            start=(50, 100),
            end=(250, 100),
            walls=[
                (0, 0, 300, 20),      # top border
                (0, 180, 300, 20),    # bottom border
                (140, 80, 20, 40),    # central obstacle
            ],
            start_r=25,
            end_r=25,
            min_path_distance=200.0,
            optimal_waypoints=[(50, 100), (150, 50), (250, 100)],
        )
        self.diff_cfg = config.DIFFICULTIES[1]
        self.collector = MetricsCollector(
            start=self.level.start,
            end=self.level.end,
            min_path_distance=self.level.min_path_distance,
            optimal_waypoints=self.level.optimal_waypoints,
        )
        self.engine = GameEngine(
            level=self.level,
            difficulty_cfg=self.diff_cfg,
            metrics_collector=self.collector,
        )

    def test_state_enum_members_and_aliases(self):
        """Verify all requested states and backward-compatibility aliases exist."""
        self.assertIn("MENU", GameState.__members__)
        self.assertIn("LEVEL_SELECT", GameState.__members__)
        self.assertIn("READY", GameState.__members__)
        self.assertIn("PLAYING", GameState.__members__)
        self.assertIn("COMPLETED", GameState.__members__)
        self.assertIn("RESULTS", GameState.__members__)
        self.assertIn("PAUSED", GameState.__members__)
        self.assertIn("TIMEOUT", GameState.__members__)

        # Backward compatibility aliases
        self.assertEqual(GameState.WAITING, GameState.READY)
        self.assertEqual(GameState.RUNNING, GameState.PLAYING)
        self.assertEqual(GameState.WIN, GameState.RESULTS)

    def test_initial_state_is_ready_and_timer_is_zero(self):
        """Initial engine state is READY and elapsed time is 0.0s."""
        self.assertEqual(self.engine.state, GameState.READY)
        self.assertEqual(self.engine.elapsed_time, 0.0)
        self.assertEqual(len(self.engine.full_trajectory), 1)  # Contains initial (start_x, start_y)
        self.assertEqual(self.engine.actual_distance, 0.0)
        self.assertFalse(self.engine.is_completed)
        self.assertFalse(self.engine.is_finished)

    def test_ready_does_not_start_timer_while_in_start_zone(self):
        """Small movements inside the start zone do NOT trigger timer or PLAYING state."""
        # Start is at (50, 100), start_r=25, player_radius=12. Threshold = 37px
        self.engine.update(cursor=(55, 100), dt=0.016)
        self.assertEqual(self.engine.state, GameState.READY)
        self.assertEqual(self.engine.elapsed_time, 0.0)

        self.engine.update(cursor=(60, 100), dt=0.016)
        self.assertEqual(self.engine.state, GameState.READY)
        self.assertEqual(self.engine.elapsed_time, 0.0)

    def test_transition_ready_to_playing_on_movement_out_of_start(self):
        """Moving cursor beyond start zone transitions state to PLAYING and starts timer."""
        # Move beyond start zone: dist((100, 50), (50, 100)) = 70.7 > 37
        self.engine.update(cursor=(100, 50), dt=0.016)
        self.assertEqual(self.engine.state, GameState.PLAYING)
        self.assertIsNotNone(self.engine._session_start)
        self.assertGreater(self.engine.elapsed_time, 0.0)

    def test_playing_state_records_trajectory_distance_and_performance(self):
        """During PLAYING, trajectory, actual distance, and metrics are actively updated."""
        self.engine.update(cursor=(100, 50), dt=0.016)
        self.assertEqual(self.engine.state, GameState.PLAYING)

        # Feed consecutive positions
        self.engine.update(cursor=(110, 50), dt=0.016)
        self.engine.update(cursor=(120, 50), dt=0.016)

        self.assertGreater(len(self.engine.full_trajectory), 2)
        self.assertGreater(self.engine.actual_distance, 0.0)
        self.assertGreaterEqual(self.engine.smoothness_score, 0.0)
        self.assertGreaterEqual(self.engine.trajectory_accuracy, 0.0)

    def test_collision_detection_during_playing(self):
        """CCD detects collisions with walls, increments counter, and prevents penetration."""
        # Start playing
        self.engine.update(cursor=(100, 50), dt=0.016)
        hits_before = self.engine.wall_hit_count

        # Central obstacle is at (140, 80, 20, 40) -> x from 140 to 160, y from 80 to 120
        # Player attempts to move through obstacle center (150, 100)
        self.engine.update(cursor=(150, 100), dt=0.016)

        # Collision should be detected
        self.assertGreater(self.engine.wall_hit_count, hits_before)
        # Player should be stopped before entering the wall center
        px, py = self.engine.player.position_f
        self.assertLess(py, 100)

    def test_reaching_end_completes_session_and_freezes_timer(self):
        """
        When player reaches END:
        - state transitions through COMPLETED to RESULTS
        - timer stops immediately
        - timer does NOT continue running on subsequent updates or time elapse
        - session is saved
        - final metrics are populated
        """
        # Start playing and navigate around obstacle to END (250, 100)
        self.engine.update(cursor=(100, 50), dt=0.016)
        self.assertEqual(self.engine.state, GameState.PLAYING)
        self.engine.update(cursor=(150, 50), dt=0.016)
        self.engine.update(cursor=(200, 50), dt=0.016)
        self.engine.update(cursor=(245, 100), dt=0.016)

        # Reached END!
        self.assertTrue(self.engine.is_completed)
        self.assertTrue(self.engine.is_finished)
        self.assertIn(self.engine.state, (GameState.COMPLETED, GameState.RESULTS))

        # Capture frozen elapsed time
        time_at_finish = self.engine.elapsed_time
        self.assertGreater(time_at_finish, 0.0)
        self.assertIsNotNone(self.engine.final_metrics)
        self.assertTrue(self.engine._metrics_saved)

        # Wait a real interval and call update multiple times
        time.sleep(0.05)
        self.engine.update(cursor=(250, 100), dt=0.016)
        self.engine.update(cursor=(250, 100), dt=0.016)

        # Verify timer is STRICTLY FROZEN and does NOT continue running
        time_after_updates = self.engine.elapsed_time
        self.assertEqual(time_at_finish, time_after_updates)

    def test_restart_resets_timers_metrics_trajectory_and_state(self):
        """
        Restart option must:
        - Return state to READY
        - Reset elapsed time to 0.0s
        - Reset player position to START
        - Reset trajectory, actual distance, and wall hits
        """
        # Play and finish
        self.engine.update(cursor=(100, 50), dt=0.016)
        self.engine.update(cursor=(150, 50), dt=0.016)
        self.engine.update(cursor=(200, 50), dt=0.016)
        self.engine.update(cursor=(245, 100), dt=0.016)
        self.assertTrue(self.engine.is_completed)

        # Trigger restart
        self.engine.restart()

        self.assertEqual(self.engine.state, GameState.READY)
        self.assertEqual(self.engine.elapsed_time, 0.0)
        self.assertEqual(self.engine.player.position, self.level.start)
        self.assertEqual(len(self.engine.full_trajectory), 1)  # Only initial start point
        self.assertEqual(self.engine.actual_distance, 0.0)
        self.assertEqual(self.engine.wall_hit_count, 0)
        self.assertFalse(self.engine.is_completed)
        self.assertIsNone(self.engine.final_metrics)

        # Verify moving again restarts clean session
        self.engine.update(cursor=(100, 50), dt=0.016)
        self.assertEqual(self.engine.state, GameState.PLAYING)
        self.assertGreater(self.engine.elapsed_time, 0.0)


if __name__ == "__main__":
    unittest.main()
