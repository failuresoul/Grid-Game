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

# Real-time telemetry debug overlay: displays Raw X,Y vs. Smoothed X,Y and jitter.
# Can be toggled at runtime using the 'D' key.
DEBUG_COORDINATES: bool = False


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
#  4. Player Cursor & Coordinate Smoothing
# =============================================================================

# Default collision/visual radius -- overridden per difficulty (see Section 10).
PLAYER_RADIUS: int = 14

# Active smoothing algorithm: "ONE_EURO" (adaptive low-pass) or "EMA" (fixed low-pass)
SMOOTHING_ALGORITHM: str = "ONE_EURO"

# ── 1-Euro Filter Parameters (Speed-adaptive low-pass filter) ─────────────────
# Minimum cutoff frequency (Hz): lower values eliminate jitter when stationary.
ONE_EURO_MIN_CUTOFF: float = 1.0

# Speed coefficient (beta): higher values reduce latency during fast movements.
ONE_EURO_BETA: float = 0.007

# Cutoff frequency (Hz) for the derivative filter.
ONE_EURO_D_CUTOFF: float = 1.0

# ── Exponential Moving Average Parameters (fixed low-pass filter) ─────────────
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
# Sufficient to preserve complete session trajectories across long levels.
TRAIL_MAX_LENGTH: int = 4000

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
#  7. Hand Tracking  (MediaPipe HandLandmarker thresholds & landmark presets)
# =============================================================================

# Minimum confidence to classify a frame as containing a hand.
HAND_DETECTION_CONFIDENCE: float = 0.70

# Minimum confidence to consider a hand from the previous frame still present.
HAND_TRACKING_CONFIDENCE:  float = 0.60

# Number of consecutive frames without a detected hand before the EMA filter resets.
# When the hand reappears after this threshold, the cursor immediately locks onto
# the new hand position without dragging or lagging across the canvas.
HAND_REACQUIRE_RESET_FRAMES: int = 5

# Supported landmark presets (MediaPipe landmark indices)
TRACKED_LANDMARKS: Dict[str, int] = {
    "INDEX_TIP":  8,   # Index finger tip (default, intuitive pointing)
    "PALM_MCP":   9,   # Middle MCP joint / palm center (stable for tremors)
    "THUMB_TIP":  4,   # Thumb tip
    "MIDDLE_TIP": 12,  # Middle finger tip
    "WRIST":      0,   # Wrist joint (gross arm motion)
}

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
#  10b. Maze Generation Mode & Procedural Seed
# =============================================================================

# Generation mode: "FIXED" (curated pre-designed levels) or "RANDOM" (procedural mazes)
MAZE_TYPE: str = "FIXED"

# Seed for procedural maze generation.
# Set to an integer (e.g. 42) for reproducible research, or None for random per session.
MAZE_SEED: Optional[int] = None

# Maximum regeneration attempts if a random maze fails solvability validation
MAZE_MAX_REGEN_ATTEMPTS: int = 50

# =============================================================================
#  11. Metrics / Session Logging
# =============================================================================

METRICS_SAVE_DIR:        str  = "data/sessions"  # Auto-created if absent
SAVE_METRICS_ON_WIN:     bool = True             # Save CSV on goal reached
SAVE_METRICS_ON_TIMEOUT: bool = True             # Save CSV on time-out

# Sample window for rolling jerk derivative (frames).
JERK_WINDOW: int = 5

# =============================================================================
#  11b. Adaptive Difficulty Recommendation
# =============================================================================
# Enable/disable adaptive difficulty suggestions on the results screen
ADAPTIVE_DIFFICULTY: bool = True

# Performance thresholds for suggesting difficulty increase (high motor control):
# Note: Raw speed is NEVER assumed to mean recovery; movement quality is prioritised.
ACCURACY_THRESHOLD: float       = 90.0   # % corridor adherence (default >= 90%)
EFFICIENCY_THRESHOLD: float     = 85.0   # % path efficiency (default >= 85%)
SMOOTHNESS_THRESHOLD: float     = 75.0   # movement smoothness score (default >= 75/100)
MAX_COLLISIONS_THRESHOLD: int   = 1      # max wall contacts allowable for promotion

# Thresholds indicating severe motor struggle (suggesting easier level or repeat):
LOW_ACCURACY_THRESHOLD: float   = 65.0   # % corridor adherence
LOW_EFFICIENCY_THRESHOLD: float = 50.0   # % path efficiency
LOW_SMOOTHNESS_THRESHOLD: float = 40.0   # movement smoothness score
HIGH_COLLISIONS_THRESHOLD: int  = 4      # wall contacts indicating difficulty

# Consecutive completed sessions required to establish consistent high performance
ADAPTIVE_CONSISTENCY_WINDOW: int = 2

# =============================================================================
#  12. Color Palette  (BGR tuples for OpenCV)
# =============================================================================
# All colours are (Blue, Green, Red) tuples in the range [0, 255].
# Professional medical/research prototype aesthetic:
# Clean deep slate, clinical mint, high-contrast cyan, and graphite walls.

# -- Background ---------------------------------------------------------------

# Solid fallback background colour (used if gradient is disabled).
BACKGROUND_COLOR:  Tuple[int, int, int] = (18,  20,  26)   # clinical dark slate
CANVAS_BG_COLOR:   Tuple[int, int, int] = BACKGROUND_COLOR

# Vertical gradient: canvas top -> canvas bottom.
BG_TOP_COLOR:    Tuple[int, int, int] = (24,  26,  34)   # deep slate charcoal
BG_BOTTOM_COLOR: Tuple[int, int, int] = (14,  16,  22)   # dark clinical navy-charcoal

# -- Walls / Obstacles --------------------------------------------------------

WALL_COLOR:        Tuple[int, int, int] = (42,  48,  58)   # clean slate graphite
WALL_BORDER_COLOR: Tuple[int, int, int] = (85, 110, 140)   # precision hairline border

# -- Player Cursor ------------------------------------------------------------

# Base / resting player colour (medical mint).
PLAYER_COLOR:      Tuple[int, int, int] = ( 80, 205, 140)  # calm medical mint
PLAYER_COLOR_SLOW: Tuple[int, int, int] = PLAYER_COLOR
PLAYER_COLOR_FAST: Tuple[int, int, int] = ( 90, 215, 245)  # clinical cyan at speed
PLAYER_GLOW_COLOR: Tuple[int, int, int] = (120, 225, 180)  # subtle mint halo

# -- Target Zones -------------------------------------------------------------

START_COLOR: Tuple[int, int, int] = ( 70, 195, 110)  # calm medical green
END_COLOR:   Tuple[int, int, int] = ( 60, 185, 240)  # clinical azure/cyan target

# -- Trajectory Trail ---------------------------------------------------------

TRAIL_COLOR: Tuple[int, int, int] = (215, 180,  85)  # clinical soft sky-blue track

# -- HUD & Typography ---------------------------------------------------------

HUD_TEXT_COLOR:   Tuple[int, int, int] = (240, 245, 250)  # crisp high-contrast white
HUD_LABEL_COLOR:  Tuple[int, int, int] = (140, 165, 190)  # clinical soft steel label
HUD_ACCENT_COLOR: Tuple[int, int, int] = ( 90, 210, 140)  # medical mint accent

# -- Overlays -----------------------------------------------------------------

WIN_COLOR:  Tuple[int, int, int] = ( 80, 210, 140)  # medical success mint
FAIL_COLOR: Tuple[int, int, int] = ( 70,  70, 225)  # soft alert red
