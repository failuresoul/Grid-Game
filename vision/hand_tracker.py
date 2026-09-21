"""
vision/hand_tracker.py — MediaPipe Hand Tracking Wrapper (Tasks API >= 0.10)

Wraps the MediaPipe HandLandmarker Tasks API to deliver:
  - Separate raw vs. smoothed coordinates (camera space and game canvas space)
  - Configurable tracked landmark (index fingertip, palm centroid, wrist, etc.)
  - Safe, crash-free handling of frames where no hand is visible or drops out
  - Auto-reset of smoothing filter upon hand re-acquisition
  - Distinct targeting visual marker with reticle and label on the camera feed
  - Pure camera-based solution with zero EMG dependencies

Pipeline:
  Camera (BGR frame)
    ↓
  Hand Detection (MediaPipe Tasks HandLandmarker)
    ↓
  Landmark Extraction (Configurable, e.g. index tip or palm centroid)
    ↓
  Raw Coordinates (Camera px, normalized [0,1], and raw canvas px)
    ↓
  Smoothing (ExponentialMovingAverage with re-entry auto-reset)
    ↓
  Game Coordinates (Continuous 2D floats clamped to canvas bounds)
    ↓
  Player Movement (GameEngine / Player)

Dependency chain:
    vision.hand_tracker -> vision.smoothing, config
"""

from __future__ import annotations
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

import config
from vision.smoothing import ExponentialMovingAverage, OneEuroFilter2D, create_smoother

log = logging.getLogger(__name__)

# ---- Model file -------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
# Store the model at the project root (one directory above this package)
_MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hand_landmarker.task",
)


def _ensure_model() -> str:
    """Download the model file if it does not exist yet."""
    if not os.path.isfile(_MODEL_FILE):
        log.info(f"Downloading hand landmarker model -> {_MODEL_FILE}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
        log.info(f"Model downloaded ({os.path.getsize(_MODEL_FILE):,} bytes)")
    return _MODEL_FILE


# ---- Hand skeleton connections (same indices as classic MediaPipe) -----------
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm arc
]

# Palm landmark group for centroid calculation
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass
class HandTrackingResult:
    """
    Structured outcome of one tracking frame, keeping raw and smoothed data strictly separate.
    """
    has_hand:             bool
    raw_camera_coords:    Optional[Tuple[float, float]] = None  # (px, py) in camera resolution
    normalized_coords:    Optional[Tuple[float, float]] = None  # (nx, ny) in [0.0, 1.0]
    raw_game_coords:      Optional[Tuple[float, float]] = None  # (x, y) raw unfiltered canvas coords
    smoothed_game_coords: Optional[Tuple[float, float]] = None  # (x, y) EMA smoothed canvas coords
    landmark_id:          int = 8
    landmark_name:        str = "INDEX_TIP"


class HandTracker:
    """
    Delivers a smoothed (cx, cy) cursor in game-canvas pixel space while exposing
    the full tracking pipeline and separate raw coordinates.

    Usage:
        tracker = HandTracker(canvas_w=1000, canvas_h=700)
        cursor  = tracker.process(bgr_frame)          # Smoothed (x, y) or None
        result  = tracker.last_result                 # Full HandTrackingResult
        raw     = tracker.raw_coords                  # Raw unfiltered (x, y)
        pip     = tracker.annotated_frame             # Camera feed with reticle
    """

    # Public landmark index constants
    WRIST             = 0
    THUMB_TIP         = 4
    INDEX_FINGER_TIP  = 8
    MIDDLE_MCP        = 9
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP   = 16
    PINKY_TIP         = 20
    PALM_CENTROID     = -1  # Sentinel for centroid of (0, 5, 9, 13, 17)

    LANDMARK_NAMES: Dict[int, str] = {
        0:  "WRIST",
        4:  "THUMB_TIP",
        8:  "INDEX_TIP",
        9:  "PALM_MCP",
        12: "MIDDLE_TIP",
        16: "RING_TIP",
        20: "PINKY_TIP",
        -1: "PALM_CENTROID",
    }

    def __init__(
        self,
        canvas_w:             int,
        canvas_h:             int,
        smoothing:            float = config.CURSOR_SMOOTHING,
        landmark_id:          int   = config.CURSOR_LANDMARK,
        max_hands:            int   = 1,
        detection_confidence: float = config.HAND_DETECTION_CONFIDENCE,
        tracking_confidence:  float = config.HAND_TRACKING_CONFIDENCE,
    ) -> None:
        self.canvas_w    = int(canvas_w)
        self.canvas_h    = int(canvas_h)
        self.landmark_id = landmark_id

        model_path = _ensure_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,          # frame-by-frame with timestamps
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker  = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0      # monotonically increasing timestamp (ms)

        # Coordinate smoother (adaptive OneEuroFilter2D or EMA, configurable from config.py)
        algo = getattr(config, "SMOOTHING_ALGORITHM", "ONE_EURO")
        self.smoother = create_smoother(
            algorithm=algo,
            alpha=smoothing,
            min_cutoff=getattr(config, "ONE_EURO_MIN_CUTOFF", 1.0),
            beta=getattr(config, "ONE_EURO_BETA", 0.007),
            d_cutoff=getattr(config, "ONE_EURO_D_CUTOFF", 1.0),
        )
        self.smoothing_algo_name: str = algo.upper()
        # Maintain self._ema as an alias for backward compatibility
        self._ema = self.smoother

        # Separate coordinates stored per frame
        self.raw_coords:      Optional[Tuple[float, float]] = None
        self.smoothed_coords: Optional[Tuple[float, float]] = None
        self.last_result:     Optional[HandTrackingResult]  = None

        # Re-acquisition tracking (prevents cursor drag when hand re-enters view)
        self._frames_without_hand: int = 0
        self.reacquire_reset_frames: int = getattr(config, "HAND_REACQUIRE_RESET_FRAMES", 5)

        # Last annotated webcam frame (for PiP overlay)
        self.annotated_frame: Optional[np.ndarray] = None

        log.info(
            f"HandTracker ready | landmark={self.landmark_name} (id={landmark_id}) "
            f"filter={self.smoothing_algo_name} smoothing={smoothing:.2f} canvas={canvas_w}x{canvas_h}"
        )

    # -------------------------------------------------------------------------
    #  Public Configuration & Inspection API
    # -------------------------------------------------------------------------

    @property
    def landmark_name(self) -> str:
        """Human-readable name of the currently tracked landmark."""
        return self.LANDMARK_NAMES.get(self.landmark_id, f"LM_{self.landmark_id}")

    @property
    def raw_position(self) -> Optional[Tuple[float, float]]:
        """Latest raw coordinate sample (rx, ry) from the smoother."""
        return self.smoother.raw_position

    @property
    def smoothed_position(self) -> Optional[Tuple[float, float]]:
        """Latest smoothed coordinate sample (sx, sy) from the smoother."""
        return self.smoother.smoothed_position

    def set_landmark(self, landmark_id: int) -> str:
        """
        Dynamically change the tracked landmark.
        Returns the name of the new landmark.
        """
        self.landmark_id = landmark_id
        name = self.landmark_name
        log.info(f"HandTracker tracked landmark changed to: {name} (id={landmark_id})")
        return name

    def reset_smoothing(self) -> None:
        """Clear smoother state — call on game restart to avoid position lag."""
        self.smoother.reset()
        self.raw_coords      = None
        self.smoothed_coords = None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        log.info("HandTracker closed.")

    # -------------------------------------------------------------------------
    #  Core Tracking Pipeline
    # -------------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Backward-compatible entry point: returns smoothed game coordinates
        Tuple[float, float] or None if no hand is detected.
        Safe against empty frames or dropped hands.
        """
        result = self.process_frame(bgr_frame)
        return result.smoothed_game_coords

    def process_frame(self, bgr_frame: np.ndarray) -> HandTrackingResult:
        """
        Execute the full tracking pipeline on a video frame:
            Camera
              ↓
            Hand detection
              ↓
            Hand landmark
              ↓
            Raw x,y coordinates
              ↓
            Smoothing
              ↓
            Game coordinates
              ↓
            Player

        Returns:
            HandTrackingResult containing both raw and smoothed coordinates.
        """
        # 1. Defensive input validation (handle corrupted/empty frames safely)
        if bgr_frame is None or bgr_frame.size == 0:
            self._handle_no_hand(annotated=None)
            return self.last_result

        # Ensure monotonically increasing timestamp
        self._frame_ts_ms += 33

        # 2. Hand detection via MediaPipe Tasks
        try:
            rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        except Exception as e:
            log.warning(f"MediaPipe detection error (frame ignored): {e}")
            self._handle_no_hand(annotated=bgr_frame.copy())
            return self.last_result

        annotated = bgr_frame.copy()

        # 3. Handle no-hand-detected frames safely (no crash, clear state)
        if not result.hand_landmarks:
            self._handle_no_hand(annotated=annotated)
            return self.last_result

        # 4. Hand re-acquisition check:
        # If the hand disappeared for multiple frames and just returned, reset
        # the smoothing filter so the cursor snaps immediately to the new hand
        # position rather than violently lagging/dragging across the canvas.
        if self._frames_without_hand >= self.reacquire_reset_frames:
            self.smoother.reset()
        self._frames_without_hand = 0

        landmarks = result.hand_landmarks[0]
        h, w = bgr_frame.shape[:2]

        # 5. Extract raw normalized [0, 1] landmark coordinates
        norm_x, norm_y = self._extract_landmark_norm(landmarks)

        # Raw camera pixel coordinates
        raw_cam_x = float(norm_x * w)
        raw_cam_y = float(norm_y * h)

        # 6. Raw game canvas coordinates (mirror X so right movement -> moves right)
        mirrored_nx = 1.0 - norm_x
        raw_gx = float(np.clip(mirrored_nx * float(self.canvas_w), 0.0, float(self.canvas_w - 1)))
        raw_gy = float(np.clip(norm_y * float(self.canvas_h), 0.0, float(self.canvas_h - 1)))
        self.raw_coords = (raw_gx, raw_gy)

        # 7. Coordinate smoothing stage (OneEuroFilter2D / EMA)
        sx, sy = self.smoother.update(raw_gx, raw_gy, dt=0.033)
        smoothed_gx = float(np.clip(sx, 0.0, float(self.canvas_w - 1)))
        smoothed_gy = float(np.clip(sy, 0.0, float(self.canvas_h - 1)))
        self.smoothed_coords = (smoothed_gx, smoothed_gy)

        # 8. Render visual marker on detected landmark
        target_pt = (int(round(raw_cam_x)), int(round(raw_cam_y)))
        self._draw_skeleton_with_marker(annotated, landmarks, target_pt)
        self.annotated_frame = annotated

        # 9. Build structured result keeping raw and smoothed strictly separate
        self.last_result = HandTrackingResult(
            has_hand=True,
            raw_camera_coords=(raw_cam_x, raw_cam_y),
            normalized_coords=(norm_x, norm_y),
            raw_game_coords=(raw_gx, raw_gy),
            smoothed_game_coords=(smoothed_gx, smoothed_gy),
            landmark_id=self.landmark_id,
            landmark_name=self.landmark_name,
        )
        return self.last_result

    # -------------------------------------------------------------------------
    #  Internal Helpers
    # -------------------------------------------------------------------------

    def _handle_no_hand(self, annotated: Optional[np.ndarray]) -> None:
        """Update internal state when no hand is present in the frame."""
        self._frames_without_hand += 1
        self.raw_coords      = None
        self.smoothed_coords = None
        self.annotated_frame = annotated
        self.last_result = HandTrackingResult(
            has_hand=False,
            raw_camera_coords=None,
            normalized_coords=None,
            raw_game_coords=None,
            smoothed_game_coords=None,
            landmark_id=self.landmark_id,
            landmark_name=self.landmark_name,
        )

    def _extract_landmark_norm(self, landmarks: list) -> Tuple[float, float]:
        """Extract normalized (x, y) in [0, 1] for the configured landmark."""
        if self.landmark_id == self.PALM_CENTROID or self.landmark_id in _PALM_LANDMARKS:
            xs = [landmarks[i].x for i in _PALM_LANDMARKS]
            ys = [landmarks[i].y for i in _PALM_LANDMARKS]
            return float(np.mean(xs)), float(np.mean(ys))

        idx = max(0, min(self.landmark_id, len(landmarks) - 1))
        lm = landmarks[idx]
        return float(lm.x), float(lm.y)

    def _landmark_to_canvas(self, landmarks: list) -> Tuple[float, float]:
        """Convert normalized landmark directly to canvas-pixel coordinates."""
        nx, ny = self._extract_landmark_norm(landmarks)
        mirrored_nx = 1.0 - nx
        return mirrored_nx * float(self.canvas_w), ny * float(self.canvas_h)

    def _draw_skeleton_with_marker(
        self,
        bgr_img:   np.ndarray,
        landmarks: list,
        target_pt: Tuple[int, int],
    ) -> None:
        """
        Draw the hand skeleton and a distinct visual marker on the tracked landmark.
        """
        h, w = bgr_img.shape[:2]
        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Skeleton bone lines
        for a, b in _HAND_CONNECTIONS:
            cv2.line(bgr_img, pts[a], pts[b], (40, 180, 80), 1, cv2.LINE_AA)

        # Standard landmark joints
        for i, pt in enumerate(pts):
            if i in _PALM_LANDMARKS:
                cv2.circle(bgr_img, pt, 3, (80, 220, 120), -1, cv2.LINE_AA)
            else:
                cv2.circle(bgr_img, pt, 2, (180, 180, 180), -1, cv2.LINE_AA)

        # --- Distinct Visual Targeting Reticle on Tracked Landmark ---
        tx, ty = target_pt

        # 1. Outer target ring (cyan)
        cv2.circle(bgr_img, (tx, ty), 10, (255, 220, 0), 2, cv2.LINE_AA)
        # 2. Inner targeting dot (yellow/gold)
        cv2.circle(bgr_img, (tx, ty), 4, (0, 255, 255), -1, cv2.LINE_AA)
        # 3. Crosshair tick marks (+)
        cv2.line(bgr_img, (tx - 14, ty), (tx - 10, ty), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(bgr_img, (tx + 10, ty), (tx + 14, ty), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(bgr_img, (tx, ty - 14), (tx, ty - 10), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(bgr_img, (tx, ty + 10), (tx, ty + 14), (255, 220, 0), 1, cv2.LINE_AA)

        # 4. Text label badge
        label = self.landmark_name
        lx = tx + 14
        ly = ty - 8
        # Background badge for high legibility
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(bgr_img, (lx - 2, ly - lh - 2), (lx + lw + 2, ly + 2), (20, 20, 30), -1)
        cv2.putText(bgr_img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
