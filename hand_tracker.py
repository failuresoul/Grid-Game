"""
hand_tracker.py — MediaPipe Hand Tracking Wrapper (Tasks API — MediaPipe ≥ 0.10)

Provides a thin, convenient wrapper around the MediaPipe HandLandmarker Tasks API
that:
  - Processes webcam frames and extracts the chosen landmark as (x, y)
  - Applies Exponential Moving Average (EMA) smoothing to reduce jitter
    (critical for tremor patients — reduces high-frequency noise in the cursor)
  - Maps normalised MediaPipe coordinates → pixel coordinates on the game canvas
  - Supports configurable landmark selection (index tip, palm centre, wrist)
  - Optionally annotates the webcam frame for the PiP overlay

Model file: hand_landmarker.task  (≈7.8 MB, downloaded automatically on first run)
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

log = logging.getLogger(__name__)

# ─── Model file ──────────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model() -> str:
    """Download the model file if it doesn't exist yet."""
    if not os.path.isfile(_MODEL_FILE):
        log.info(f"Downloading hand landmarker model → {_MODEL_FILE}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
        log.info(f"Model downloaded ({os.path.getsize(_MODEL_FILE)} bytes)")
    return _MODEL_FILE


# ─── Drawing helpers (Tasks API has its own drawing utils) ────────────────────
_drawing_utils  = mp_vision.drawing_utils
_drawing_styles = mp_vision.drawing_styles

# Hand connections constant lives here in Tasks API
_HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandTracker:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) and returns a smoothed
    (cx, cy) cursor position in game-canvas pixel space each frame.

    Usage:
        tracker = HandTracker(canvas_w=1000, canvas_h=700)
        while True:
            ok, frame = cap.read()
            cursor = tracker.process(frame)   # None if no hand
    """

    # ─── Landmark indices ────────────────────────────────────────────────────
    WRIST             = 0
    INDEX_FINGER_TIP  = 8
    MIDDLE_MCP        = 9    # stable palm-centre approximation

    # Landmarks that form the palm centroid (average for a very stable cursor)
    PALM_LANDMARKS = (0, 5, 9, 13, 17)   # wrist + 4 MCP joints

    def __init__(
        self,
        canvas_w: int,
        canvas_h: int,
        smoothing: float = config.CURSOR_SMOOTHING,
        landmark_id: int = config.CURSOR_LANDMARK,
        max_hands: int = 1,
        detection_confidence: float = 0.70,
        tracking_confidence: float  = 0.60,
    ) -> None:
        self.canvas_w    = canvas_w
        self.canvas_h    = canvas_h
        self.smoothing   = smoothing          # EMA α  (lower = smoother)
        self.landmark_id = landmark_id

        model_path = _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,          # VIDEO = frame-by-frame with timestamps
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0   # monotonically increasing timestamp for VIDEO mode

        # EMA state — None until first detection
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Last annotated frame (webcam image with skeleton drawn)
        self.annotated_frame: Optional[np.ndarray] = None

        log.info(
            f"HandTracker ready | landmark={landmark_id} "
            f"smoothing={smoothing} canvas={canvas_w}×{canvas_h}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect a hand in `bgr_frame` and return a smoothed cursor position.

        Args:
            bgr_frame: Raw BGR frame from cv2.VideoCapture.

        Returns:
            (cx, cy) in game-canvas pixel coordinates, or None if no hand found.
        """
        # Advance the video timestamp (33 ms ≈ 30 fps; must be monotonically increasing)
        self._frame_ts_ms += 33

        # MediaPipe Tasks API uses mp.Image (RGB)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        # Build annotated frame for PiP
        annotated = bgr_frame.copy()

        if not result.hand_landmarks:
            self.annotated_frame = annotated
            return None

        # ── Draw skeleton on annotated frame ─────────────────────────────────
        # Tasks API: draw_landmarks expects a mp.Image, not ndarray
        annotated_rgb = rgb.copy()
        hand_proto = result.hand_landmarks[0]

        # Use the protobuf-style NormalizedLandmarkList for drawing
        # The Tasks API result has hand_landmarks as list[list[NormalizedLandmark]]
        self._draw_landmarks_on(annotated, hand_proto)
        self.annotated_frame = annotated

        # ── Extract raw landmark position ─────────────────────────────────────
        raw_x, raw_y = self._extract_position(hand_proto)

        # ── Mirror X: webcam is front-facing; mirror so right-hand movement
        #    moves cursor right (natural mapping)
        raw_x = self.canvas_w - raw_x

        # ── EMA smoothing ─────────────────────────────────────────────────────
        if self._smooth_x is None:
            self._smooth_x = float(raw_x)
            self._smooth_y = float(raw_y)
        else:
            α = self.smoothing
            self._smooth_x = α * raw_x + (1 - α) * self._smooth_x
            self._smooth_y = α * raw_y + (1 - α) * self._smooth_y

        # Clamp to canvas bounds
        cx = int(np.clip(self._smooth_x, 0, self.canvas_w  - 1))
        cy = int(np.clip(self._smooth_y, 0, self.canvas_h  - 1))

        return cx, cy

    def reset_smoothing(self) -> None:
        """Clear EMA state (call on game restart so cursor snaps to hand)."""
        self._smooth_x = None
        self._smooth_y = None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        log.info("HandTracker closed.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_position(self, landmarks: list) -> Tuple[float, float]:
        """
        Convert the chosen MediaPipe landmark to canvas-pixel coordinates.
        landmarks is a list of NormalizedLandmark (Tasks API).
        """
        if self.landmark_id in self.PALM_LANDMARKS:
            xs = [landmarks[i].x for i in self.PALM_LANDMARKS]
            ys = [landmarks[i].y for i in self.PALM_LANDMARKS]
            nx = float(np.mean(xs))
            ny = float(np.mean(ys))
        else:
            lm = landmarks[self.landmark_id]
            nx, ny = lm.x, lm.y

        cx = nx * self.canvas_w
        cy = ny * self.canvas_h
        return cx, cy

    def _draw_landmarks_on(self, bgr_img: np.ndarray, landmarks: list) -> None:
        """
        Draw hand landmark dots and connections directly onto a BGR ndarray.
        The Tasks API doesn't provide a direct ndarray drawing path, so we
        draw manually using OpenCV for simplicity.
        """
        h, w = bgr_img.shape[:2]

        # Connection pairs (Tasks API — same indices as classic MP)
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),        # thumb
            (0,5),(5,6),(6,7),(7,8),         # index
            (0,9),(9,10),(10,11),(11,12),    # middle
            (0,13),(13,14),(14,15),(15,16),  # ring
            (0,17),(17,18),(18,19),(19,20),  # pinky
            (5,9),(9,13),(13,17),            # palm
        ]

        # Map to pixel coordinates
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw connections
        for a, b in CONNECTIONS:
            cv2.line(bgr_img, pts[a], pts[b], (0, 200, 100), 1, cv2.LINE_AA)

        # Draw landmark dots; highlight the active cursor landmark
        for i, pt in enumerate(pts):
            if i == self.landmark_id:
                cv2.circle(bgr_img, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            elif i in self.PALM_LANDMARKS:
                cv2.circle(bgr_img, pt, 4, (100, 255, 100), -1, cv2.LINE_AA)
            else:
                cv2.circle(bgr_img, pt, 3, (200, 200, 200), -1, cv2.LINE_AA)
