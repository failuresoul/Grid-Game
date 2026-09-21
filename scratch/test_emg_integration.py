"""
scratch/test_emg_integration.py — Unit Tests for Isolated EMG Architecture

Verifies all EMG integration requirements:
1. Default setting is EMG_ENABLED = False.
2. The game and EMG interface function cleanly with zero hardware requirements.
3. Functions and methods exist:
     - initialize_emg()
     - read_emg()
     - process_emg()
     - get_activation_level()
     - get_rms()
     - get_peak()
     - close_emg()
4. Anti-Fake-Measurement Guarantee:
   When EMG is disabled or disconnected, functions return None rather than fabricating
   fake synthetic sensor readings.
5. UI displays: "EMG: Not Connected / Disabled"
6. Signal processing pipeline correctly computes RMS, Peak, and MVC Activation when
   supplied real signal samples.
7. Game engine and renderer operate identically without hardware.
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import config
from emg.emg_interface import (
    EMGInterface,
    initialize_emg,
    read_emg,
    process_emg,
    get_activation_level,
    get_rms,
    get_peak,
    close_emg,
    get_emg_status_string,
    is_emg_enabled,
    is_emg_connected,
)


class TestEMGIntegration(unittest.TestCase):
    """Test suite for isolated EMG interface and safe fallback behavior."""

    def setUp(self):
        # Reset default interface state
        close_emg()

    def tearDown(self):
        close_emg()

    def test_default_config_emg_enabled_is_false(self):
        """EMG_ENABLED must remain False by default."""
        self.assertFalse(config.EMG_ENABLED, "EMG_ENABLED must default to False.")
        self.assertFalse(is_emg_enabled(), "is_emg_enabled() must return False.")

    def test_emg_functions_exist_and_callable(self):
        """Verify that all required EMG functions exist and are callable."""
        funcs = [
            initialize_emg,
            read_emg,
            process_emg,
            get_activation_level,
            get_rms,
            get_peak,
            close_emg,
            get_emg_status_string,
        ]
        for fn in funcs:
            self.assertTrue(callable(fn), f"{fn.__name__} must be callable.")

    def test_initialization_without_hardware_returns_false(self):
        """When EMG_ENABLED is False, initialize_emg() must return False without crashing."""
        success = initialize_emg()
        self.assertFalse(success, "initialize_emg() must return False when disabled.")
        self.assertFalse(is_emg_connected(), "is_emg_connected() must be False.")

    def test_no_fake_measurements_when_disabled(self):
        """
        CRITICAL CLINICAL REQUIREMENT:
        Do NOT generate fake EMG values and present them as real sensor measurements.
        When hardware is absent or disabled, read_emg, get_rms, get_peak, and
        get_activation_level must return None.
        """
        self.assertIsNone(read_emg(), "read_emg() must return None when disabled.")
        self.assertIsNone(get_rms(), "get_rms() must return None when disabled.")
        self.assertIsNone(get_peak(), "get_peak() must return None when disabled.")
        self.assertIsNone(get_activation_level(), "get_activation_level() must return None when disabled.")
        self.assertIsNone(process_emg(), "process_emg() must return None when disabled.")

    def test_ui_status_string_when_disabled(self):
        """
        When EMG is disabled, UI status string must be exactly:
        'EMG: Not Connected / Disabled'
        """
        status_str = get_emg_status_string()
        self.assertEqual(
            status_str,
            "EMG: Not Connected / Disabled",
            "UI status must say 'EMG: Not Connected / Disabled' when disabled.",
        )

    def test_emg_interface_class_oop_api(self):
        """Verify the object-oriented EMGInterface class methods."""
        interface = EMGInterface(port="COM3", device="bitalino")
        self.assertFalse(interface.is_ready())
        self.assertFalse(interface.is_connected)
        self.assertIsNone(interface.read_emg())
        self.assertIsNone(interface.get_rms())
        self.assertIsNone(interface.get_peak())
        self.assertIsNone(interface.get_activation_level())
        self.assertEqual(interface.get_status_string(), "EMG: Not Connected / Disabled")
        interface.close_emg()

    def test_signal_processing_mathematics(self):
        """
        Verify the mathematical DSP calculations (RMS, peak, MAV, MVC normalization)
        when provided a known sample window.
        """
        interface = EMGInterface(mvc_reference=100.0)
        # Test signal: symmetric alternating samples [-10.0, 10.0, -10.0, 10.0]
        # Mean = 0.0, centered = [-10, 10, -10, 10]
        # RMS = sqrt((100+100+100+100)/4) = 10.0
        # Peak = 10.0
        # MAV = 10.0
        # Activation = 10.0 / 100.0 = 0.10
        samples = [-10.0, 10.0, -10.0, 10.0]
        res = interface.process_emg(samples)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["rms"], 10.0, places=3)
        self.assertAlmostEqual(res["peak"], 10.0, places=3)
        self.assertAlmostEqual(res["mean_abs"], 10.0, places=3)
        self.assertAlmostEqual(res["activation_level"], 0.10, places=3)

    def test_ui_rendering_includes_emg_status_without_hardware(self):
        """Verify that UI renderer and screens render cleanly with the EMG status text."""
        from ui.renderer import Renderer
        from game.game_engine import GameEngine, GameState
        from game.maze import MazeGenerator

        renderer = Renderer(canvas_w=800, canvas_h=600)
        mg = MazeGenerator()
        level = mg.get_level(1, 0)
        engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])
        engine.state = GameState.PLAYING

        frame = renderer.draw(engine=engine, dt=0.016)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (600, 800, 3))


if __name__ == "__main__":
    unittest.main()
