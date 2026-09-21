"""
vision/hand_tracker.py — MediaPipe Hand Tracking Wrapper (Tasks API >= 0.10)

Wraps the MediaPipe HandLandmarker Tasks API to deliver a smoothed
(cx, cy) cursor position in game-canvas pixel coordinates each frame.

Features:
  - Monotonic VIDEO-mode timestamps for detect_for_video()
  - Configurable landmark selection (index tip, palm centroid, wrist)
  - EMA smoothing via vision.smoothing.ExponentialMovingAverage
  - Horizontal mirror so right-hand movement -> cursor moves right
  - Auto-download of the hand_landmarker.task model on first run
  - OpenCV skeleton annotation on the webcam frame (for PiP)

Dependency chain:
    vision.hand_tracker -> vision.smoothing, config
"""

from __future__ import annotations
import logging
import os
import urllib.request
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

import config
from vision.smoothing import ExponentialMovingAverage

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

# Palm landmark group for centroid-mode cursor
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


class HandTracker:
    """
    Delivers a smoothed (cx, cy) cursor in game-canvas pixel space.

    Usage:
        tracker = HandTracker(canvas_w=1000, canvas_h=700)
        cursor  = tracker.process(bgr_frame)   # None if no hand visible
        pip     = tracker.annotated_frame       # frame with skeleton drawn
    """

    # Public landmark index constants
    WRIST            = 0
    INDEX_FINGER_TIP = 8
    MIDDLE_MCP       = 9

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
        self.canvas_w    = canvas_w
        self.canvas_h    = canvas_h
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
        self._ema         = ExponentialMovingAverage(alpha=smoothing)

        # Last annotated webcam frame (for PiP overlay)
        self.annotated_frame: Optional[np.ndarray] = None

        log.info(
            f"HandTracker ready | landmark={landmark_id} "
            f"smoothing={smoothing:.2f} canvas={canvas_w}x{canvas_h}"
        )

    # -------------------------------------------------------------------------
    #  Public API
    # -------------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect hand in bgr_frame and return a smoothed, continuous canvas cursor.

        Mapping Pipeline:
            1. Camera hand position -> MediaPipe normalized landmark (nx, ny) in [0, 1]
            2. Horizontal mirror: nx_mirrored = 1.0 - nx (so right-hand movement moves right)
            3. Direct mapping to game canvas: (x_game, y_game) = (nx_mirrored * W, ny * H)
            4. Exponential Moving Average (EMA) smoothing for tremor reduction
            5. Return continuous 2D float coordinates (no grid or cell snapping)

        Args:
            bgr_frame: Raw BGR frame from cv2.VideoCapture.

        Returns:
            (cx, cy) continuous float coordinates in game space, or None if no hand detected.
        """
        self._frame_ts_ms += 33   # ~30 fps; must be monotonically increasing

        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        annotated = bgr_frame.copy()

        if not result.hand_landmarks:
            self.annotated_frame = annotated
            return None

        landmarks = result.hand_landmarks[0]
        self._draw_skeleton(annotated, landmarks)
        self.annotated_frame = annotated

        # Extract landmark normalized coordinates [0, 1] -> canvas game space
        raw_x, raw_y = self._landmark_to_canvas(landmarks)

        # EMA smoothing across frames for continuous, smooth movement
        sx, sy = self._ema.update(raw_x, raw_y)

        # Clamp continuously to canvas boundaries
        cx = float(np.clip(sx, 0.0, float(self.canvas_w - 1)))
        cy = float(np.clip(sy, 0.0, float(self.canvas_h - 1)))
        return cx, cy

    def reset_smoothing(self) -> None:
        """Clear EMA state — call on game restart to avoid position lag."""
        self._ema.reset()

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        log.info("HandTracker closed.")

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------

    def _landmark_to_canvas(self, landmarks: list) -> Tuple[float, float]:
        """
        Convert normalized MediaPipe landmark [0, 1] to game-canvas coordinates.
        Applies horizontal mirroring so physical hand movements map naturally.
        """
        if self.landmark_id in _PALM_LANDMARKS:
            xs = [landmarks[i].x for i in _PALM_LANDMARKS]
            ys = [landmarks[i].y for i in _PALM_LANDMARKS]
            nx = float(np.mean(xs))
            ny = float(np.mean(ys))
        else:
            lm = landmarks[self.landmark_id]
            nx, ny = float(lm.x), float(lm.y)

        # Mirror X: camera mirror so user hand moving right moves right in game
        mirrored_nx = 1.0 - nx

        # Direct linear mapping to game-space continuous 2D coordinates
        x_game = mirrored_nx * float(self.canvas_w)
        y_game = ny * float(self.canvas_h)
        return x_game, y_game

    def _draw_skeleton(self, bgr_img: np.ndarray, landmarks: list) -> None:
        """Draw the hand skeleton onto a BGR ndarray using OpenCV."""
        h, w = bgr_img.shape[:2]
        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in _HAND_CONNECTIONS:
            cv2.line(bgr_img, pts[a], pts[b], (0, 200, 100), 1, cv2.LINE_AA)

        for i, pt in enumerate(pts):
            if i == self.landmark_id:
                cv2.circle(bgr_img, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            elif i in _PALM_LANDMARKS:
                cv2.circle(bgr_img, pt, 4, (100, 255, 100), -1, cv2.LINE_AA)
            else:
                cv2.circle(bgr_img, pt, 3, (200, 200, 200), -1, cv2.LINE_AA)
