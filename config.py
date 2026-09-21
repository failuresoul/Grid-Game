"""
config.py — Central configuration for the Stroke Rehabilitation Maze Game.

All game constants, camera settings, difficulty presets, and feature flags
live here so they can be adjusted without touching game logic.

EMG NOTE:
  EMG_ENABLED = False  →  EMG hardware is not connected.
  Set to True and configure EMG_PORT/EMG_DEVICE when hardware is available.
  See emg_interface.py for re-activation instructions.
"""

from dataclasses import dataclass, field
from typing import Tuple

# ─────────────────────────────────────────────
#  Feature Flags
# ─────────────────────────────────────────────

# TODO: Set to True when an EMG sensor (e.g. Myo Armband, BITalino) is connected.
#       Also configure EMG_PORT and EMG_DEVICE below.
EMG_ENABLED: bool = False

# Show the live webcam feed as a picture-in-picture in the top-right corner.
CAMERA_PIP_ENABLED: bool = True

# ─────────────────────────────────────────────
#  Camera / OpenCV
# ─────────────────────────────────────────────

CAMERA_INDEX: int = 0          # Webcam device index (0 = default/first camera)
CAMERA_WIDTH: int = 640        # Capture resolution width
CAMERA_HEIGHT: int = 480       # Capture resolution height
TARGET_FPS: int = 30           # Desired game loop FPS

# ─────────────────────────────────────────────
#  Game Canvas
# ─────────────────────────────────────────────

CANVAS_WIDTH: int  = 1000      # Game rendering canvas width  (px)
CANVAS_HEIGHT: int = 700       # Game rendering canvas height (px)

# PiP (picture-in-picture) webcam thumbnail size
PIP_WIDTH: int  = 200
PIP_HEIGHT: int = 150

# ─────────────────────────────────────────────
#  Player Cursor
# ─────────────────────────────────────────────

PLAYER_RADIUS: int = 14        # Collision / visual radius of the player dot (px)

# Exponential Moving Average smoothing factor (0 < α ≤ 1).
# Lower = more smoothing (good for tremor); higher = more responsive.
CURSOR_SMOOTHING: float = 0.30

# Which MediaPipe hand landmark to use as the cursor:
#   8  = index finger tip  (default)
#   9  = middle finger MCP (palm-centre-ish, more stable)
#   0  = wrist
CURSOR_LANDMARK: int = 8

# ─────────────────────────────────────────────
#  EMG Hardware (used only when EMG_ENABLED=True)
# ─────────────────────────────────────────────

# TODO: Set these when EMG hardware is connected.
EMG_PORT: str   = "COM3"       # Serial port (Windows) or "/dev/ttyUSB0" (Linux)
EMG_DEVICE: str = "bitalino"   # "bitalino" | "myo" | "custom"
EMG_SAMPLE_RATE: int = 1000    # Hz
EMG_CHANNELS: Tuple[int, ...] = (0, 1)  # Which analog channels to read

# ─────────────────────────────────────────────
#  Difficulty Presets
# ─────────────────────────────────────────────

@dataclass
class DifficultyConfig:
    """All parameters that change between Easy / Medium / Hard."""
    name: str
    # Maze geometry
    corridor_min_width: int      # Narrowest passage width (px)
    num_obstacles: int           # Number of wall segments placed
    # Player
    player_radius: int           # Smaller = harder to navigate
    cursor_smoothing: float      # More smoothing on easier levels (tremor support)
    # Scoring / time
    time_limit_sec: float        # 0 = no limit
    # Visual hints
    show_path_hint: bool         # Draw ideal-path ghost line
    hint_color: Tuple[int,int,int]  # BGR

EASY: DifficultyConfig = DifficultyConfig(
    name="Easy",
    corridor_min_width=120,
    num_obstacles=3,
    player_radius=16,
    cursor_smoothing=0.22,       # Very smooth — good for severe tremor
    time_limit_sec=0,
    show_path_hint=True,
    hint_color=(60, 180, 60),
)

MEDIUM: DifficultyConfig = DifficultyConfig(
    name="Medium",
    corridor_min_width=70,
    num_obstacles=7,
    player_radius=14,
    cursor_smoothing=0.30,
    time_limit_sec=0,
    show_path_hint=False,
    hint_color=(60, 180, 60),
)

HARD: DifficultyConfig = DifficultyConfig(
    name="Hard",
    corridor_min_width=40,
    num_obstacles=13,
    player_radius=12,
    cursor_smoothing=0.40,       # Less smoothing — full tremor challenge
    time_limit_sec=0,
    show_path_hint=False,
    hint_color=(60, 180, 60),
)

DIFFICULTIES = {1: EASY, 2: MEDIUM, 3: HARD}
DEFAULT_DIFFICULTY: int = 1

# ─────────────────────────────────────────────
#  Metrics / Session Logging
# ─────────────────────────────────────────────

METRICS_SAVE_DIR: str = "data/sessions"  # Directory for per-session CSV files
SAVE_METRICS_ON_TIMEOUT: bool = True # Save CSV even if the player ran out of time
SAVE_METRICS_ON_WIN: bool = True     # Save CSV on successful completion

# Jerk smoothness: number of samples in the rolling derivative window
JERK_WINDOW: int = 5

# ─────────────────────────────────────────────
#  Visual / Colour Palette  (BGR tuples)
# ─────────────────────────────────────────────

# Background gradient top / bottom colours
BG_TOP_COLOR:    Tuple[int,int,int] = (28,  18,  42)   # deep purple
BG_BOTTOM_COLOR: Tuple[int,int,int] = (10,  40,  60)   # dark teal

# Walls
WALL_COLOR:       Tuple[int,int,int] = (40,  80, 130)  # steel blue
WALL_BORDER_COLOR:Tuple[int,int,int] = (80, 160, 220)  # bright blue

# Player
PLAYER_COLOR_SLOW: Tuple[int,int,int] = ( 80, 220, 130)  # teal-green
PLAYER_COLOR_FAST: Tuple[int,int,int] = ( 60, 120, 255)  # blue-white
PLAYER_GLOW_COLOR: Tuple[int,int,int] = (100, 200, 255)  # light cyan

# Start / End markers
START_COLOR: Tuple[int,int,int] = ( 50, 200,  80)   # green
END_COLOR:   Tuple[int,int,int] = ( 30, 200, 255)   # gold-ish cyan

# Path trail
TRAIL_COLOR: Tuple[int,int,int] = (120, 200, 255)   # faint cyan

# HUD text
HUD_TEXT_COLOR:  Tuple[int,int,int] = (220, 240, 255)
HUD_LABEL_COLOR: Tuple[int,int,int] = (120, 160, 200)

# Win / Fail overlays
WIN_COLOR:  Tuple[int,int,int] = ( 50, 220, 120)
FAIL_COLOR: Tuple[int,int,int] = ( 60,  60, 220)
