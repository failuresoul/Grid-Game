"""
config.py — Centralized Configuration
=======================================
Stroke Rehabilitation Maze Game — ALL tunable constants live here.

Rules:
  - No magic numbers in game, vision, metrics, ui, or emg modules.
  - Every module reads this file; nothing writes back to it at runtime.
  - EMG_ENABLED = False  ->  the game runs completely without a sensor.
    The EMG subsystem is never imported or instantiated when this is False.

Sections:
  1. Feature Flags
  2. Camera
  3. Game Canvas
  4. Player Cursor
  5. Zone Radii  (start / end circles)
  6. Walls
  7. Hand Tracking
  8. Frame Timing
  9. EMG Hardware  (used only when EMG_ENABLED = True)
  10. Difficulty Presets
  11. Metrics / Session Logging
  12. Color Palette  (BGR tuples for OpenCV)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# =============================================================================
#  1. Feature Flags
# =============================================================================

# +-------------------------------------------------------------------------+
# |  EMG_ENABLED = False   <-- DEFAULT.  Game works with camera only.        |
# |  Set True only after physically connecting a supported EMG device.       |
# |  See emg/emg_interface.py for re-activation instructions.                |
# +-------------------------------------------------------------------------+
EMG_ENABLED: bool = False

# Show a small live webcam feed as a picture-in-picture overlay.
CAMERA_PIP_ENABLED: bool = True

# Development / testing fallback: allows mouse cursor control when hand tracking
# is inactive or when no webcam is available. Camera hand tracking remains primary.
MOUSE_FALLBACK_ENABLED: bool = True


# =============================================================================
#  2. Camera
# =============================================================================

CAMERA_INDEX:  int = 0     # Device index (0 = first/default webcam)
CAMERA_WIDTH:  int = 640   # Capture resolution width  (px)
CAMERA_HEIGHT: int = 480   # Capture resolution height (px)
FPS:           int = 30    # Target game-loop frames per second

# PiP thumbnail size rendered in the game canvas corner
PIP_WIDTH:  int = 200
PIP_HEIGHT: int = 150

# =============================================================================
#  3. Game Canvas
# =============================================================================

CANVAS_WIDTH:  int = 1000  # Rendering surface width  (px)
CANVAS_HEIGHT: int = 700   # Rendering surface height (px)

# =============================================================================
#  4. Player Cursor
# =============================================================================

# Default collision/visual radius -- overridden per difficulty (see Section 10).
PLAYER_RADIUS: int = 14

# Exponential Moving Average factor for cursor smoothing  (0 < alpha <= 1).
#   Lower = more smoothing  -> good for severe tremor patients.
#   Higher = more responsive -> appropriate for lighter impairment.
#   Overridden per difficulty in DifficultyConfig.cursor_smoothing.
CURSOR_SMOOTHING: float = 0.30

# MediaPipe landmark index used as the cursor control point:
#   8  = index finger tip   (default)
#   9  = middle MCP joint   (more stable, palm-centre-ish)
#   0  = wrist
CURSOR_LANDMARK: int = 8

# Maximum trail points kept in memory (older ones discarded FIFO).
TRAIL_MAX_LENGTH: int = 400

# Noise dead-zone: movements smaller than this Euclidean distance (px) are ignored
# to suppress camera jitter and resting hand micro-tremor without artificial snapping.
MIN_MOVE_PX: float = 1.5

# Minimum displacement (px) required to append a new point to the player trajectory.
MIN_TRAIL_DIST_PX: float = 1.0

# Speed ceiling (px/s) used for player colour lerp (slow->fast colour blend).
PLAYER_SPEED_MAX: float = 300.0

# Number of translucent glow rings drawn around the player cursor.
PLAYER_GLOW_LAYERS: int = 3

# =============================================================================
#  5. Zone Radii
# =============================================================================

# Radius of the green START circle (px).
START_ZONE_RADIUS: int = 24

# Radius of the gold GOAL / END star zone (px).
END_ZONE_RADIUS: int = 24

# =============================================================================
#  6. Walls
# =============================================================================

# Default wall segment thickness used by level generators (px).
# Increasing this narrows corridors; decreasing it widens them.
WALL_THICKNESS: int = 22

# =============================================================================
#  7. Hand Tracking  (MediaPipe HandLandmarker thresholds)
# =============================================================================

# Minimum confidence to classify a frame as containing a hand.
HAND_DETECTION_CONFIDENCE: float = 0.70

# Minimum confidence to consider a hand from the previous frame still present.
HAND_TRACKING_CONFIDENCE:  float = 0.60

# =============================================================================
#  8. Frame Timing
# =============================================================================

# Maximum dt (seconds) per game-loop frame.
# Caps lag spikes (e.g. from camera I/O delays) to prevent cursor teleporting.
FRAME_DT_MAX: float = 0.10

# =============================================================================
#  9. EMG Hardware  (read only when EMG_ENABLED = True)
# =============================================================================
# These values are NEVER used while EMG_ENABLED = False.
# Kept here so the full hardware configuration path is self-contained.

EMG_PORT:        str             = "COM3"       # Windows: "COM3" | Linux: "/dev/ttyUSB0"
EMG_DEVICE:      str             = "bitalino"   # "bitalino" | "myo" | "custom"
EMG_SAMPLE_RATE: int             = 1000         # Hz
EMG_CHANNELS:    Tuple[int, ...] = (0, 1)       # Analog channels to read

# =============================================================================
#  10. Difficulty Presets
# =============================================================================

@dataclass
class DifficultyConfig:
    """
    All parameters that vary between Easy / Medium / Hard.

    Attributes:
        name:               Display name ("Easy", "Medium", "Hard").
        corridor_min_width: Narrowest navigable corridor width (px).
                            Design guideline for level authors.
        num_obstacles:      Reference obstacle count (informational).
        player_radius:      Collision circle radius (px) -- smaller = harder.
        cursor_smoothing:   EMA alpha override for this difficulty.
        time_limit_sec:     Session time limit in seconds (0 = unlimited).
        show_path_hint:     Draw a subtle ideal-path ghost line on the canvas.
        hint_color:         BGR colour for the path-hint line.
    """
    name:               str
    corridor_min_width: int
    num_obstacles:      int
    player_radius:      int
    cursor_smoothing:   float
    time_limit_sec:     float
    show_path_hint:     bool
    hint_color:         Tuple[int, int, int]


EASY: DifficultyConfig = DifficultyConfig(
    name="Easy",
    corridor_min_width=120,
    num_obstacles=3,
    player_radius=16,
    cursor_smoothing=0.22,       # Maximum smoothing -- supports severe tremor
    time_limit_sec=0,            # No time pressure
    show_path_hint=True,         # Ghost line guides the patient
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
    cursor_smoothing=0.40,       # Minimal smoothing -- full tremor challenge
    time_limit_sec=0,
    show_path_hint=False,
    hint_color=(60, 180, 60),
)

DIFFICULTIES: Dict[int, DifficultyConfig] = {1: EASY, 2: MEDIUM, 3: HARD}
DEFAULT_DIFFICULTY: int = 1

# =============================================================================
#  11. Metrics / Session Logging
# =============================================================================

METRICS_SAVE_DIR:        str  = "data/sessions"  # Auto-created if absent
SAVE_METRICS_ON_WIN:     bool = True             # Save CSV on goal reached
SAVE_METRICS_ON_TIMEOUT: bool = True             # Save CSV on time-out

# Sample window for rolling jerk derivative (frames).
JERK_WINDOW: int = 5

# =============================================================================
#  12. Color Palette  (BGR tuples for OpenCV)
# =============================================================================
# All colours are (Blue, Green, Red) tuples in the range [0, 255].
# Change any value here to restyle the entire game without touching game code.

# -- Background ---------------------------------------------------------------

# Solid fallback background colour (used if gradient is disabled).
BACKGROUND_COLOR:  Tuple[int, int, int] = (19,  29,  51)   # dark navy

# Vertical gradient: canvas top -> canvas bottom.
BG_TOP_COLOR:    Tuple[int, int, int] = (28,  18,  42)   # deep purple
BG_BOTTOM_COLOR: Tuple[int, int, int] = (10,  40,  60)   # dark teal

# -- Walls --------------------------------------------------------------------

WALL_COLOR:        Tuple[int, int, int] = (40,  80, 130)  # steel blue
WALL_BORDER_COLOR: Tuple[int, int, int] = (80, 160, 220)  # bright edge glow

# -- Player -------------------------------------------------------------------

# Base / resting player colour.  Also aliased as PLAYER_COLOR_SLOW.
PLAYER_COLOR:      Tuple[int, int, int] = ( 80, 220, 130)  # teal-green

# Speed-dependent colour interpolation endpoints.
PLAYER_COLOR_SLOW: Tuple[int, int, int] = PLAYER_COLOR     # identical to base
PLAYER_COLOR_FAST: Tuple[int, int, int] = ( 60, 120, 255)  # blue-white at speed

# Outer glow colour.
PLAYER_GLOW_COLOR: Tuple[int, int, int] = (100, 200, 255)  # light cyan

# -- Zones --------------------------------------------------------------------

START_COLOR: Tuple[int, int, int] = ( 50, 200,  80)  # green
END_COLOR:   Tuple[int, int, int] = ( 30, 200, 255)  # gold cyan

# -- Path Trail ---------------------------------------------------------------

TRAIL_COLOR: Tuple[int, int, int] = (120, 200, 255)  # faint cyan

# -- HUD ----------------------------------------------------------------------

HUD_TEXT_COLOR:  Tuple[int, int, int] = (220, 240, 255)  # near-white
HUD_LABEL_COLOR: Tuple[int, int, int] = (120, 160, 200)  # muted blue

# -- Result Overlays ----------------------------------------------------------

WIN_COLOR:  Tuple[int, int, int] = ( 50, 220, 120)  # bright green
FAIL_COLOR: Tuple[int, int, int] = ( 60,  60, 220)  # red
