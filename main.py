"""
main.py — Entry Point: Stroke Rehabilitation Maze Game

Run with:
    python main.py

Keyboard controls (game window must be focused):
    1 / 2 / 3     Select difficulty (Easy / Medium / Hard)
    ENTER / SPACE Start game from selection screen
    R             Restart current level
    N             Next level (same difficulty)
    P             Pause / Resume
    C             Toggle camera PiP
    ESC           Quit

Architecture:
    main.py
        vision.hand_tracker  →  (cx, cy) cursor
        game.maze            →  Level layout
        game.game_engine     →  State machine + collision
        metrics.performance  →  Clinical metrics + CSV
        ui.renderer          →  OpenCV frame composition
        emg.emg_interface    →  (disabled stub, imported only if EMG_ENABLED)

EMG:
    Completely isolated in the emg/ package.
    Set config.EMG_ENABLED = True and configure EMG_PORT / EMG_DEVICE
    to re-activate hardware support.  See emg/emg_interface.py for details.
"""

from __future__ import annotations

# ── Suppress TensorFlow / MediaPipe / Google log noise ───────────────────────
# These must be set BEFORE cv2, mediapipe, or tensorflow are imported.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",   "3")   # Hide TF info/warning/error logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",  "0")   # Silence oneDNN floating-point notices
os.environ.setdefault("GLOG_minloglevel",        "3")   # Silence Google-glog (MediaPipe internals)
os.environ.setdefault("GLOG_logtostderr",        "0")   # Don't copy glog to stderr
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU",   "1")   # Avoid GPU-init messages on CPU systems

import logging
import math
import sys
import time
from typing import List, Optional, Tuple


import cv2
import numpy as np

import config
from vision.hand_tracker   import HandTracker
from game.maze             import MazeGenerator
from game.game_engine      import GameEngine, GameState
from metrics.performance   import MetricsCollector
from ui.renderer           import Renderer

# ── EMG: imported only when explicitly enabled ────────────────────────────────
if config.EMG_ENABLED:
    from emg.emg_interface import EMGInterface


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture Selector  — hover + hold to activate any button without keyboard
# ─────────────────────────────────────────────────────────────────────────────

class GestureSelector:
    """
    Tracks cursor position against a list of named button rects and fires
    a callback after the cursor has hovered inside the same button for
    HOLD_DURATION seconds without leaving.

    Each button entry is:  (button_id, (x, y, w, h), label, key)
    — matching the format returned by get_results_button_rects() etc.

    Usage each frame:
        fired_id = gesture.update(cursor, buttons, dt)
        if fired_id:
            handle(fired_id)

    Also exposes .hover_id and .hover_progress (0–1) for rendering feedback.
    """

    HOLD_DURATION: float = 1.5   # seconds to hold before firing

    def __init__(self) -> None:
        self.hover_id:       Optional[str]   = None
        self.hover_timer:    float            = 0.0
        self.hover_progress: float            = 0.0   # 0.0 – 1.0

    def reset(self) -> None:
        """Clear hover state (call on screen transitions)."""
        self.hover_id       = None
        self.hover_timer    = 0.0
        self.hover_progress = 0.0

    def update(
        self,
        cursor:  Optional[Tuple[float, float]],
        buttons: list,   # list of (b_id, (x,y,w,h), label, key)
        dt:      float,
    ) -> Optional[str]:
        """
        Advance hover timer.  Returns fired button_id on completion, else None.
        """
        if cursor is None:
            self.reset()
            return None

        cx, cy = float(cursor[0]), float(cursor[1])
        hit_id: Optional[str] = None

        for entry in buttons:
            b_id, (bx, by, bw, bh), _label, _key = entry
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                hit_id = b_id
                break

        if hit_id != self.hover_id:
            # Cursor moved to a different button — reset timer
            self.hover_id    = hit_id
            self.hover_timer = 0.0

        if self.hover_id is not None:
            self.hover_timer    += dt
            self.hover_progress  = min(1.0, self.hover_timer / self.HOLD_DURATION)
            if self.hover_timer >= self.HOLD_DURATION:
                fired = self.hover_id
                self.reset()
                return fired
        else:
            self.hover_progress = 0.0

        return None


def _draw_gesture_progress(
    canvas: np.ndarray,
    buttons: list,
    hover_id: Optional[str],
    hover_progress: float,
) -> None:
    """
    Draw a semi-transparent fill bar inside the hovered button to show
    gesture-hold progress.  Called after the button backgrounds are drawn.
    """
    if hover_id is None or hover_progress <= 0.0:
        return
    for entry in buttons:
        b_id, (bx, by, bw, bh), _label, _key = entry
        if b_id != hover_id:
            continue
        fill_w = int(bw * hover_progress)
        if fill_w <= 0:
            break
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + fill_w, by + bh), (0, 200, 120), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        # Thin progress border
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 140), 2,
                      cv2.LINE_AA)
        break


# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class RehabGame:
    """Top-level application — owns the main loop and wires all subsystems."""

    WINDOW_NAME = "Hand Rehabilitation Maze"

    def __init__(self) -> None:
        self.difficulty:   int  = config.DEFAULT_DIFFICULTY
        self.level_index:  int  = 0          # 0-based index within the difficulty pool
        # ── Application State Machine ─────────────────────────────────────────
        self.app_state: GameState = GameState.MENU

        self.cap:      cv2.VideoCapture | None = None
        self.tracker:  HandTracker      | None = None
        self.maze_gen: MazeGenerator           = MazeGenerator()
        self.renderer: Renderer                = Renderer()
        self.engine:   GameEngine       | None = None

        # ── Mouse fallback for development/testing ───────────────────────────
        self.mouse_fallback_enabled: bool = config.MOUSE_FALLBACK_ENABLED
        self._mouse_pos: Optional[Tuple[float, float]] = None
        self._using_mouse: bool = False

        # ── Real-time coordinate telemetry debug mode ────────────────────────
        self.debug_mode: bool = getattr(config, "DEBUG_COORDINATES", False)

        # ── Progress & History state ─────────────────────────────────────────
        self.history_filter: str = "ALL"
        self.history_page: int = 0
        self.history_sessions: list = []

        # ── Gesture Selector (hover + hold for all UI screens) ───────────────
        self.gesture = GestureSelector()

        # ── Moveable PiP Camera state ─────────────────────────────────────────
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        self.pip_pos: List[int] = [config.CANVAS_WIDTH - pw - 12, 12]
        self.pip_dragging: bool = False
        self.pip_drag_offset: Tuple[int, int] = (0, 0)
        self.pip_corner_idx: int = 1
        self.pip_user_moved: bool = False

        # ── EMG Sensor (Future Integration) ──────────────────────────────────
        # Default: config.EMG_ENABLED = False.
        # No hardware required; game runs completely with camera hand tracking.
        # When physical hardware is connected in the future, set config.EMG_ENABLED = True.
        if getattr(config, "EMG_ENABLED", False):
            from emg.emg_interface import EMGInterface, initialize_emg
            self.emg = EMGInterface()
            initialize_emg()
        else:
            self.emg = None


    @property
    def _start_screen(self) -> bool:
        """Backward compatibility helper."""
        return self.app_state == GameState.MENU

    @_start_screen.setter
    def _start_screen(self, val: bool) -> None:
        self.app_state = GameState.MENU if val else GameState.READY

    # ─────────────────────────────────────────────────────────────────────────
    #  Setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _cycle_pip_corner(self) -> None:
        """Cycle camera PiP between candidate screen corners."""
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        W, H = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
        margin = 12
        corners = [
            [margin, margin],                         # 0: Top-Left
            [W - pw - margin, margin],                # 1: Top-Right
            [W - pw - margin, H - ph - margin - 22],  # 2: Bottom-Right
            [margin, H - ph - margin - 22],           # 3: Bottom-Left
        ]
        self.pip_corner_idx = (self.pip_corner_idx + 1) % len(corners)
        self.pip_pos = list(corners[self.pip_corner_idx])
        self.pip_user_moved = True
        log.info(f"Camera PiP moved to corner {self.pip_corner_idx}: {self.pip_pos}")

    def _auto_place_pip_for_level(self, level) -> None:
        """Place camera PiP in the cleanest corner if user hasn't explicitly placed it."""
        if self.pip_user_moved:
            return
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        W, H = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
        margin = 12
        corners = [
            (margin, margin),                          # 0: Top-Left
            (W - pw - margin, margin),                 # 1: Top-Right
            (W - pw - margin, H - ph - margin - 22),   # 2: Bottom-Right
            (margin, H - ph - margin - 22),            # 3: Bottom-Left
        ]
        sx, sy = float(level.start[0]), float(level.start[1])
        ex, ey = float(level.end[0]), float(level.end[1])
        sr = float(getattr(level, "start_r", 35))
        er = float(getattr(level, "end_r", 35))

        best_corner = corners[1]
        best_score = -1e9
        pip_radius = math.hypot(pw / 2, ph / 2)
        for cx, cy in corners:
            d_start = math.hypot((cx + pw / 2) - sx, (cy + ph / 2) - sy)
            d_end = math.hypot((cx + pw / 2) - ex, (cy + ph / 2) - ey)

            collides_target = (d_start < (sr + pip_radius + 20)) or (d_end < (er + pip_radius + 20))
            collides_wall = False
            for obs in getattr(level, "walls", []):
                if hasattr(obs, "as_rect"):
                    ox, oy, ow, oh = obs.as_rect()
                elif isinstance(obs, (tuple, list)) and len(obs) == 4:
                    ox, oy, ow, oh = obs
                else:
                    continue
                if not (cx + pw < ox or cx > ox + ow or cy + ph + 18 < oy or cy > oy + oh):
                    collides_wall = True
                    break

            overlaps_header = (cx < 260 and cy < 95)
            score = min(d_start, d_end)
            if collides_target:
                score -= 30000.0
            if collides_wall:
                score -= 15000.0
            if overlaps_header:
                score -= 5000.0

            if score > best_score:
                best_score = score
                best_corner = (cx, cy)
        self.pip_pos = list(best_corner)

    def _open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            log.error("No webcam detected.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.FPS)
        log.info(f"Camera opened (index={config.CAMERA_INDEX})")
        return True

    def _build_tracker(self) -> None:
        diff_cfg = config.DIFFICULTIES[self.difficulty]
        if self.tracker is not None:
            self.tracker.close()
        self.tracker = HandTracker(
            canvas_w=config.CANVAS_WIDTH,
            canvas_h=config.CANVAS_HEIGHT,
            smoothing=diff_cfg.cursor_smoothing,
            landmark_id=config.CURSOR_LANDMARK,
        )

    def _new_game(self) -> None:
        diff_cfg = config.DIFFICULTIES[self.difficulty]
        level    = self.maze_gen.get_level(self.difficulty, self.level_index)
        self._auto_place_pip_for_level(level)
        metrics  = MetricsCollector(
            start=level.start,
            end=level.end,
            min_path_distance=level.min_path_distance,
            optimal_waypoints=getattr(level, "optimal_waypoints", None),
        )
        self.engine = GameEngine(level=level, difficulty_cfg=diff_cfg,
                                 metrics_collector=metrics)
        self.engine.require_start_dwell = True
        self.app_state = GameState.READY
        if self.tracker:
            self.tracker.reset_smoothing()
        log.info(f"New game: {diff_cfg.name}  |  {level.name} (READY)")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Track mouse position and handle interactive UI button clicks and PiP dragging."""
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self._mouse_pos = (float(x), float(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = self.pip_pos
            if px <= x <= px + pw and py <= y <= py + ph:
                self.pip_dragging = True
                self.pip_user_moved = True
                self.pip_drag_offset = (int(x - px), int(y - py))
                return
            if self.app_state in (GameState.COMPLETED, GameState.RESULTS):
                self._handle_results_click(float(x), float(y))
            elif self.app_state == GameState.HISTORY:
                self._handle_history_click(float(x), float(y))

        elif event == cv2.EVENT_MOUSEMOVE and self.pip_dragging:
            nx = int(x - self.pip_drag_offset[0])
            ny = int(y - self.pip_drag_offset[1])
            self.pip_pos[0] = max(5, min(config.CANVAS_WIDTH - pw - 5, nx))
            self.pip_pos[1] = max(5, min(config.CANVAS_HEIGHT - ph - 25, ny))

        elif event == cv2.EVENT_LBUTTONUP:
            self.pip_dragging = False

    def _handle_history_click(self, x: float, y: float) -> bool:
        """Handle mouse clicks on history screen buttons and filter tabs."""
        from ui.screens import get_history_button_rects
        from metrics.history_reader import filter_sessions
        filtered = filter_sessions(self.history_sessions, self.history_filter)
        total_pages = max(1, (len(filtered) + 6) // 7)
        buttons = get_history_button_rects(
            config.CANVAS_WIDTH, config.CANVAS_HEIGHT,
            current_filter=self.history_filter,
            page=self.history_page,
            total_pages=total_pages,
        )
        for b_id, (bx, by, bw, bh), label, key in buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                log.info(f"History button clicked: {label} ({b_id})")
                if b_id == "FILTER_ALL":
                    self.history_filter = "ALL"
                    self.history_page = 0
                elif b_id == "FILTER_EASY":
                    self.history_filter = "EASY"
                    self.history_page = 0
                elif b_id == "FILTER_MEDIUM":
                    self.history_filter = "MEDIUM"
                    self.history_page = 0
                elif b_id == "FILTER_HARD":
                    self.history_filter = "HARD"
                    self.history_page = 0
                elif b_id == "PREV_PAGE":
                    self.history_page = max(0, self.history_page - 1)
                elif b_id == "NEXT_PAGE":
                    self.history_page = min(total_pages - 1, self.history_page + 1)
                elif b_id == "MAIN_MENU":
                    self.app_state = GameState.MENU
                elif b_id == "LEVEL_SELECT":
                    self.app_state = GameState.LEVEL_SELECT
                elif b_id == "EXIT":
                    sys.exit(0)
                return True
        return False

    def _handle_results_click(self, x: float, y: float) -> bool:
        """Handle mouse clicks on results screen action buttons."""
        from ui.screens import get_results_button_rects
        buttons = get_results_button_rects(config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
        for b_id, (bx, by, bw, bh), label, key in buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                log.info(f"Results button clicked: {label} ({b_id})")
                self._handle_results_action(b_id)
                return True
        return False

    def _handle_results_action(self, b_id: str) -> None:
        """Execute a results-screen button action (called by mouse OR gesture)."""
        if b_id == "PLAY_AGAIN":
            # Restart the SAME level from READY (don't advance index)
            if self.engine:
                self.engine.restart()
                self.app_state = GameState.READY
            else:
                self._new_game()
            self.gesture.reset()
        elif b_id == "NEXT_LEVEL":
            total = self.maze_gen.level_count(self.difficulty)
            self.level_index = (self.level_index + 1) % total
            self._new_game()
            self.gesture.reset()
        elif b_id == "LEVEL_SELECT":
            self.app_state = GameState.LEVEL_SELECT
            self.gesture.reset()
        elif b_id == "MAIN_MENU":
            self.app_state = GameState.MENU
            self.gesture.reset()
        elif b_id == "EXIT":
            sys.exit(0)

    def _handle_history_action(self, b_id: str) -> None:
        """Execute a history-screen button action (called by mouse OR gesture)."""
        from metrics.history_reader import filter_sessions
        filtered = filter_sessions(self.history_sessions, self.history_filter)
        total_pages = max(1, (len(filtered) + 6) // 7)
        if b_id == "FILTER_ALL":
            self.history_filter = "ALL";  self.history_page = 0
        elif b_id == "FILTER_EASY":
            self.history_filter = "EASY"; self.history_page = 0
        elif b_id == "FILTER_MEDIUM":
            self.history_filter = "MEDIUM"; self.history_page = 0
        elif b_id == "FILTER_HARD":
            self.history_filter = "HARD"; self.history_page = 0
        elif b_id == "PREV_PAGE":
            self.history_page = max(0, self.history_page - 1)
        elif b_id == "NEXT_PAGE":
            self.history_page = min(total_pages - 1, self.history_page + 1)
        elif b_id == "MAIN_MENU":
            self.app_state = GameState.MENU;  self.gesture.reset()
        elif b_id == "LEVEL_SELECT":
            self.app_state = GameState.LEVEL_SELECT; self.gesture.reset()
        elif b_id == "EXIT":
            sys.exit(0)


    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("=== Rehab Maze Game starting ===")

        has_camera = self._open_camera()
        if not has_camera:
            if self.mouse_fallback_enabled:
                log.warning("No camera detected, but MOUSE_FALLBACK_ENABLED is True. Running in mouse dev mode.")
            else:
                self._no_camera_screen()
                return

        if has_camera:
            self._build_tracker()

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        prev_time = time.perf_counter()

        try:
            while True:
                now = time.perf_counter()
                dt  = min(now - prev_time, config.FRAME_DT_MAX)
                prev_time = now

                cam_frame = None
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if ok:
                        cam_frame = frame
                    else:
                        time.sleep(0.01)

                # ── Hand tracking (primary) + Mouse fallback ──────────────────
                cursor    = None
                pip_frame = None
                if self.tracker and cam_frame is not None:
                    cursor    = self.tracker.process(cam_frame, dt=max(0.001, dt))
                    pip_frame = self.tracker.annotated_frame

                is_mouse = False
                if cursor is None and self.mouse_fallback_enabled and self._mouse_pos is not None:
                    cursor = self._mouse_pos
                    is_mouse = True
                self._using_mouse = is_mouse

                # ── 1. MENU State ─────────────────────────────────────────────
                pip_rect = (int(self.pip_pos[0]), int(self.pip_pos[1]), config.PIP_WIDTH, 22)

                if self.app_state == GameState.MENU:
                    from ui.screens import get_menu_button_rects
                    canvas    = self.renderer.draw_start_screen(self.difficulty)
                    menu_btns = get_menu_button_rects(config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
                    menu_btns.append(("MOVE_CAMERA", pip_rect, "MOVE CAM", "V"))
                    fired = self.gesture.update(cursor, menu_btns, dt)
                    _draw_gesture_progress(canvas, menu_btns, self.gesture.hover_id, self.gesture.hover_progress)
                    if fired:
                        if fired == 'DIFF_1':
                            self.difficulty = 1; self.gesture.reset()
                        elif fired == 'DIFF_2':
                            self.difficulty = 2; self.gesture.reset()
                        elif fired == 'DIFF_3':
                            self.difficulty = 3; self.gesture.reset()
                        elif fired == 'START':
                            self.app_state = GameState.LEVEL_SELECT
                            self.gesture.reset()
                            log.info('Gesture: MENU -> LEVEL_SELECT')
                        elif fired == 'HISTORY':
                            from metrics.history_reader import load_all_sessions
                            self.history_sessions = load_all_sessions()
                            self.history_filter   = 'ALL'
                            self.history_page     = 0
                            self.app_state        = GameState.HISTORY
                            self.gesture.reset()
                        elif fired == 'MOVE_CAMERA':
                            self._cycle_pip_corner()
                            self.gesture.reset()
                    self._composite_pip(canvas, pip_frame)
                    cv2.imshow(self.WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_menu_key(key):
                        break
                    if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                # ── 1b. HISTORY State ─────────────────────────────────────────
                if self.app_state == GameState.HISTORY:
                    from ui.screens import get_history_button_rects
                    from metrics.history_reader import filter_sessions
                    filtered_s = filter_sessions(self.history_sessions, self.history_filter)
                    total_pgs  = max(1, (len(filtered_s) + 6) // 7)
                    canvas, _  = self.renderer.draw_history_screen(
                        sessions=self.history_sessions,
                        current_filter=self.history_filter,
                        page=self.history_page,
                        mouse_pos=self._mouse_pos,
                    )
                    hist_btns = get_history_button_rects(
                        config.CANVAS_WIDTH, config.CANVAS_HEIGHT,
                        current_filter=self.history_filter,
                        page=self.history_page,
                        total_pages=total_pgs,
                    )
                    hist_btns.append(("MOVE_CAMERA", pip_rect, "MOVE CAM", "V"))
                    fired = self.gesture.update(cursor, hist_btns, dt)
                    _draw_gesture_progress(
                        canvas, hist_btns,
                        self.gesture.hover_id, self.gesture.hover_progress,
                    )
                    if fired == 'MOVE_CAMERA':
                        self._cycle_pip_corner()
                        self.gesture.reset()
                    elif fired:
                        self._handle_history_action(fired)
                    self._composite_pip(canvas, pip_frame)
                    cv2.imshow(self.WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_history_key(key):
                        break
                    if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                # ── 2. LEVEL_SELECT State ─────────────────────────────────────
                if self.app_state == GameState.LEVEL_SELECT:
                    from ui.screens import get_level_select_button_rects
                    diff_cfg     = config.DIFFICULTIES[self.difficulty]
                    total_levels = self.maze_gen.level_count(self.difficulty)
                    prev_lvl     = self.maze_gen.get_level(self.difficulty, self.level_index)
                    canvas = self.renderer.draw_level_select_screen(
                        difficulty_name=diff_cfg.name,
                        level_index=self.level_index,
                        total_levels=total_levels,
                        level_name=prev_lvl.name,
                        min_path_distance=prev_lvl.minimum_path_distance,
                        wall_count=len(prev_lvl.walls),
                    )
                    lvl_btns = get_level_select_button_rects(config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
                    lvl_btns.append(("MOVE_CAMERA", pip_rect, "MOVE CAM", "V"))
                    fired = self.gesture.update(cursor, lvl_btns, dt)
                    _draw_gesture_progress(canvas, lvl_btns, self.gesture.hover_id, self.gesture.hover_progress)
                    if fired:
                        total = self.maze_gen.level_count(self.difficulty)
                        if fired == 'PREV_LEVEL':
                            self.level_index = (self.level_index - 1) % total
                            self.gesture.reset()
                        elif fired == 'NEXT_LEVEL':
                            self.level_index = (self.level_index + 1) % total
                            self.gesture.reset()
                        elif fired == 'CONFIRM':
                            if self.cap and self.cap.isOpened():
                                self._build_tracker()
                            self._new_game()
                            self.app_state = GameState.READY
                            self.gesture.reset()
                            log.info('Gesture: LEVEL_SELECT -> READY')
                        elif fired == 'BACK':
                            self.app_state = GameState.MENU
                            self.gesture.reset()
                        elif fired == 'MOVE_CAMERA':
                            self._cycle_pip_corner()
                            self.gesture.reset()
                    self._composite_pip(canvas, pip_frame, level=prev_lvl)
                    cv2.imshow(self.WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_level_select_key(key):
                        break
                    if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                # ── 3, 4, 5, 6. READY / PLAYING / COMPLETED / RESULTS ─────────
                if self.engine is None:
                    self._new_game()

                self.engine.update(cursor, dt)
                self.app_state = self.engine.state

                landmark_name = self.tracker.landmark_name if self.tracker else "INDEX_TIP"
                raw_coords = self.tracker.raw_coords if self.tracker else None
                algo_name = getattr(self.tracker, "smoothing_algo_name", "ONE_EURO") if self.tracker else ""

                canvas = self.renderer.draw(
                    self.engine, dt, pip_frame,
                    is_mouse_fallback=self._using_mouse,
                    landmark_name=landmark_name,
                    raw_coords=raw_coords,
                    debug_mode=self.debug_mode,
                    algo_name=algo_name,
                    mouse_pos=self._mouse_pos,
                    pip_pos=tuple(self.pip_pos),
                )

                # ── Gesture control per active gameplay state ─────────────────
                pip_rect = (int(self.pip_pos[0]), int(self.pip_pos[1]), config.PIP_WIDTH, 22)

                if self.app_state in (GameState.READY, GameState.WAITING):
                    # 1. Start beacon target area
                    sx, sy = int(self.engine.level.start[0]), int(self.engine.level.start[1])
                    sr = int(self.engine.level.start_r + self.engine.radius + 15)
                    start_beacon_rect = (sx - sr, sy - sr, sr * 2, sr * 2)

                    # 2. Bottom prominent PROCEED button card
                    proceed_bx = config.CANVAS_WIDTH // 2 - 130
                    proceed_by = config.CANVAS_HEIGHT - 60
                    proceed_rect = (proceed_bx, proceed_by, 260, 48)

                    cv2.rectangle(canvas, (proceed_bx - 2, proceed_by - 2),
                                  (proceed_bx + 262, proceed_by + 50), (0, 255, 140), 2, cv2.LINE_AA)
                    overlay_p = canvas.copy()
                    cv2.rectangle(overlay_p, (proceed_bx, proceed_by),
                                  (proceed_bx + 260, proceed_by + 48), (0, 160, 85), -1)
                    cv2.addWeighted(overlay_p, 0.85, canvas, 0.15, 0, canvas)
                    p_txt = "START MAZE  [ENTER]"
                    pts = cv2.getTextSize(p_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
                    cv2.putText(canvas, p_txt, (proceed_bx + (260 - pts[0]) // 2, proceed_by + (48 + pts[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

                    ready_btns = [
                        ("START_ZONE", start_beacon_rect, "START", "ENTER"),
                        ("PROCEED", proceed_rect, "START MAZE", "ENTER"),
                        ("BACK", (15, 8, 120, 36), "< LEVELS", "ESC"),
                        ("MOVE_CAMERA", (config.CANVAS_WIDTH - 155, 8, 140, 36), "MOVE CAM [V]", "V"),
                        ("CAM_HEADER", pip_rect, "MOVE CAM", "V"),
                    ]
                    # Avoid overlapping PiP if located top-right
                    if self.pip_pos[0] > config.CANVAS_WIDTH - config.PIP_WIDTH - 40 and self.pip_pos[1] < 70:
                        ready_btns[3] = ("MOVE_CAMERA", (config.CANVAS_WIDTH - 155, config.CANVAS_HEIGHT - 45, 140, 36), "MOVE CAM [V]", "V")

                    fired = self.gesture.update(cursor, ready_btns, dt)
                    _draw_gesture_progress(canvas, ready_btns, self.gesture.hover_id, self.gesture.hover_progress)
                    if fired in ('START_ZONE', 'PROCEED'):
                        self.engine.start_playing()
                        self.app_state = GameState.PLAYING
                        self.gesture.reset()
                        log.info("Gesture: READY -> PLAYING (dwell activated)")
                    elif fired == 'BACK':
                        self.app_state = GameState.LEVEL_SELECT
                        self.gesture.reset()
                    elif fired in ('MOVE_CAMERA', 'CAM_HEADER'):
                        self._cycle_pip_corner()
                        self.gesture.reset()

                elif self.app_state in (GameState.PLAYING, GameState.RUNNING):
                    play_btns = [
                        ("PAUSE", (config.CANVAS_WIDTH // 2 - 55, 6, 110, 34), "PAUSE", "P"),
                        ("CAM_HEADER", pip_rect, "MOVE CAM", "V"),
                    ]
                    cv2.rectangle(canvas, (config.CANVAS_WIDTH // 2 - 55, 6),
                                  (config.CANVAS_WIDTH // 2 + 55, 40), (24, 32, 46), -1)
                    cv2.rectangle(canvas, (config.CANVAS_WIDTH // 2 - 55, 6),
                                  (config.CANVAS_WIDTH // 2 + 55, 40), (60, 100, 150), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "PAUSE [P]", (config.CANVAS_WIDTH // 2 - 42, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 215, 245), 1, cv2.LINE_AA)
                    fired = self.gesture.update(cursor, play_btns, dt)
                    _draw_gesture_progress(canvas, play_btns, self.gesture.hover_id, self.gesture.hover_progress)
                    if fired == 'PAUSE':
                        if self.engine:
                            self.engine.toggle_pause()
                            self.app_state = self.engine.state
                        self.gesture.reset()
                    elif fired == 'CAM_HEADER':
                        self._cycle_pip_corner()
                        self.gesture.reset()

                elif self.app_state == GameState.PAUSED:
                    from ui.screens import get_paused_button_rects
                    p_btns = get_paused_button_rects(config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
                    p_btns.append(("CAM_HEADER", pip_rect, "MOVE CAM", "V"))
                    fired = self.gesture.update(cursor, p_btns, dt)
                    _draw_gesture_progress(canvas, p_btns, self.gesture.hover_id, self.gesture.hover_progress)
                    if fired == 'RESUME':
                        if self.engine:
                            self.engine.toggle_pause()
                            self.app_state = self.engine.state
                        self.gesture.reset()
                    elif fired == 'MAIN_MENU':
                        self.app_state = GameState.MENU
                        self.gesture.reset()
                    elif fired == 'CAM_HEADER':
                        self._cycle_pip_corner()
                        self.gesture.reset()

                elif self.app_state in (GameState.COMPLETED, GameState.RESULTS,
                                        GameState.WIN, GameState.TIMEOUT):
                    from ui.screens import get_results_button_rects
                    res_btns = get_results_button_rects(
                        config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
                    res_btns.append(("CAM_HEADER", pip_rect, "MOVE CAM", "V"))
                    fired = self.gesture.update(cursor, res_btns, dt)
                    _draw_gesture_progress(
                        canvas, res_btns,
                        self.gesture.hover_id,
                        self.gesture.hover_progress,
                    )
                    if fired == 'CAM_HEADER':
                        self._cycle_pip_corner()
                        self.gesture.reset()
                    elif fired:
                        log.info(f"Gesture activated: {fired}")
                        self._handle_results_action(fired)
                else:
                    self.gesture.reset()

                cv2.imshow(self.WINDOW_NAME, canvas)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_game_key(key):
                    break
                if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            self._cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    #  Keyboard handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _cycle_landmark(self) -> None:
        """Cycle through common tracked landmarks: Index -> Palm -> Middle -> Thumb -> Wrist."""
        if not self.tracker:
            return
        cycle = [
            HandTracker.INDEX_FINGER_TIP,
            HandTracker.MIDDLE_MCP,
            HandTracker.MIDDLE_FINGER_TIP,
            HandTracker.THUMB_TIP,
            HandTracker.WRIST,
        ]
        curr = self.tracker.landmark_id
        idx = cycle.index(curr) if curr in cycle else 0
        next_id = cycle[(idx + 1) % len(cycle)]
        name = self.tracker.set_landmark(next_id)
        log.info(f"Switched tracked landmark to: {name} (id={next_id})")

    def _handle_menu_key(self, key: int) -> bool:
        """Handle keyboard input on MENU screen."""
        if key == 27:
            return True
        if key in (ord('1'), ord('2'), ord('3')):
            self.difficulty = int(chr(key))
            self.level_index = 0
        elif key in (ord('v'), ord('V')):
            self._cycle_pip_corner()
        elif key in (ord('l'), ord('L')):
            self._cycle_landmark()
        elif key in (ord('d'), ord('D')):
            self.debug_mode = not self.debug_mode
            log.info(f"Debug coordinates overlay toggled: {self.debug_mode}")
        elif key in (ord('m'), ord('M')):
            self.mouse_fallback_enabled = not self.mouse_fallback_enabled
            log.info(f"Mouse fallback toggled: {self.mouse_fallback_enabled}")
        elif key in (ord('h'), ord('H')):
            from metrics.history_reader import load_all_sessions
            self.history_sessions = load_all_sessions()
            self.history_filter = "ALL"
            self.history_page = 0
            self.app_state = GameState.HISTORY
            log.info("State transition: MENU -> HISTORY")
        elif key in (13, 32):   # ENTER or SPACE -> Proceed to LEVEL_SELECT
            self.app_state = GameState.LEVEL_SELECT
            log.info("State transition: MENU -> LEVEL_SELECT")
        return False

    def _handle_history_key(self, key: int) -> bool:
        """Handle keyboard input on HISTORY screen."""
        if key == 27 or key in (ord('m'), ord('M')):  # ESC or M -> return to MENU
            self.app_state = GameState.MENU
            log.info("State transition: HISTORY -> MENU")
            return False
        if key in (ord('v'), ord('V')):
            self._cycle_pip_corner()
            return False
        if key in (ord('l'), ord('L')):
            self.app_state = GameState.LEVEL_SELECT
            log.info("State transition: HISTORY -> LEVEL_SELECT")
            return False
        if key == ord('1'):
            self.history_filter = "ALL"
            self.history_page = 0
        elif key == ord('2'):
            self.history_filter = "EASY"
            self.history_page = 0
        elif key == ord('3'):
            self.history_filter = "MEDIUM"
            self.history_page = 0
        elif key == ord('4'):
            self.history_filter = "HARD"
            self.history_page = 0
        elif key in (ord('p'), ord('P')):
            self.history_page = max(0, self.history_page - 1)
        elif key in (ord('n'), ord('N')):
            from metrics.history_reader import filter_sessions
            filtered = filter_sessions(self.history_sessions, self.history_filter)
            total_pages = max(1, (len(filtered) + 6) // 7)
            self.history_page = min(total_pages - 1, self.history_page + 1)
        return False

    def _handle_level_select_key(self, key: int) -> bool:
        """Handle keyboard input on LEVEL_SELECT screen."""
        if key == 27:  # ESC -> return to MENU
            self.app_state = GameState.MENU
            log.info("State transition: LEVEL_SELECT -> MENU")
            return False
        if key in (ord('m'), ord('M')):
            self.app_state = GameState.MENU
            log.info("State transition: LEVEL_SELECT -> MENU")
            return False
        if key in (ord('v'), ord('V')):
            self._cycle_pip_corner()
            return False
        total = self.maze_gen.level_count(self.difficulty)
        if key in (ord('n'), ord('N'), 83, 2555904):  # N or Right Arrow
            self.level_index = (self.level_index + 1) % total
        elif key in (ord('p'), ord('P'), 81, 2424832):  # P or Left Arrow
            self.level_index = (self.level_index - 1) % total
        elif key in (ord('1'), ord('2'), ord('3')):
            self.difficulty = int(chr(key))
            self.level_index = 0
        elif key in (13, 32):  # ENTER or SPACE -> Confirm selection, load level -> READY
            if self.cap and self.cap.isOpened():
                self._build_tracker()
            self._new_game()
            self.app_state = GameState.READY
            log.info("State transition: LEVEL_SELECT -> READY")
        return False

    def _handle_start_key(self, key: int) -> bool:
        """Backward-compatible alias for _handle_menu_key."""
        return self._handle_menu_key(key)

    def _handle_game_key(self, key: int) -> bool:
        """Handle keyboard input during READY, PLAYING, COMPLETED, or RESULTS."""
        if key == 27:
            return True
        elif key in (ord('r'), ord('R')):
            # Restart current level cleanly back to READY
            if self.engine:
                self.engine.restart()
                self.app_state = GameState.READY
            else:
                self._new_game()
        elif key in (ord('v'), ord('V')):
            self._cycle_pip_corner()
        elif key in (13, 32):  # ENTER or SPACE
            if self.app_state in (GameState.READY, GameState.WAITING):
                # ── Explicit start: begin timer + gameplay NOW ──────────────
                if self.engine and self.engine.state in (GameState.READY, GameState.WAITING):
                    self.engine.start_playing()
                    self.app_state = GameState.PLAYING
                    log.info("State transition: READY -> PLAYING (explicit SPACE/ENTER start)")
            elif self.app_state in (GameState.COMPLETED, GameState.RESULTS):
                # Advance to next level after completion
                total = self.maze_gen.level_count(self.difficulty)
                self.level_index = (self.level_index + 1) % total
                self._new_game()
        elif key in (ord('n'), ord('N')):
            # Advance to next level
            total = self.maze_gen.level_count(self.difficulty)
            self.level_index = (self.level_index + 1) % total
            self._new_game()
        elif key in (ord('m'), ord('M')):
            # Return to main MENU
            self.app_state = GameState.MENU
            log.info("Returning to MENU")
        elif key in (ord('p'), ord('P')):
            if self.engine:
                self.engine.toggle_pause()
                self.app_state = self.engine.state
        elif key in (ord('l'), ord('L')):
            if self.app_state in (GameState.COMPLETED, GameState.RESULTS):
                self.app_state = GameState.LEVEL_SELECT
                log.info("Results navigation -> LEVEL_SELECT")
            else:
                self._cycle_landmark()
        elif key in (ord('d'), ord('D')):
            self.debug_mode = not self.debug_mode
            log.info(f"Debug coordinates overlay toggled: {self.debug_mode}")
        elif key in (ord('1'), ord('2'), ord('3')):
            new_d = int(chr(key))
            if new_d != self.difficulty:
                self.difficulty  = new_d
                self.level_index = 0
                if self.cap and self.cap.isOpened():
                    self._build_tracker()
                self._new_game()
        elif key in (ord('c'), ord('C')):
            config.CAMERA_PIP_ENABLED = not config.CAMERA_PIP_ENABLED
        elif key in (ord('h'), ord('H')) and self.app_state in (GameState.COMPLETED, GameState.RESULTS):
            from metrics.history_reader import load_all_sessions
            self.history_sessions = load_all_sessions()
            self.history_filter = "ALL"
            self.history_page = 0
            self.app_state = GameState.HISTORY
            log.info("State transition: RESULTS -> HISTORY")
        return False




    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_save_metrics(self) -> None:
        """Save session CSV exactly once after the session ends."""
        if not (self.engine and self.engine.is_finished):
            return
        if self.engine._metrics_saved:
            return

        won      = self.engine.state in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN)
        should   = (won and config.SAVE_METRICS_ON_WIN) or \
                   (not won and config.SAVE_METRICS_ON_TIMEOUT)

        if should:
            self.engine.save_session_metrics()

    def _composite_pip(
        self,
        canvas:    np.ndarray,
        pip_frame: np.ndarray | None,
        level=None,           # pass current level for smart corner selection
    ) -> None:
        """Blit moveable PiP thumbnail onto any canvas."""
        if pip_frame is None or not config.CAMERA_PIP_ENABLED:
            return
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        H, W   = canvas.shape[:2]

        try:
            thumb = cv2.resize(pip_frame, (pw, ph))
        except Exception:
            return

        x1 = max(5, min(W - pw - 5, int(self.pip_pos[0])))
        y1 = max(5, min(H - ph - 25, int(self.pip_pos[1])))

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x1 + pw + 3, y1 + ph + 3), (20, 26, 38), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        canvas[y1:y1 + ph, x1:x1 + pw] = thumb
        cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x1 + pw + 1, y1 + ph + 1),
                      (80, 140, 220), 1, cv2.LINE_AA)

        # Header handle
        cv2.rectangle(canvas, (x1, y1), (x1 + pw, y1 + 16), (18, 25, 36), -1)
        cv2.putText(canvas, "CAM [DRAG / MOVE]", (x1 + 6, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 215, 245), 1, cv2.LINE_AA)

    def _no_camera_screen(self) -> None:
        W, H   = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        msgs   = [
            ("ERROR: No camera detected",              0.9, (60, 60, 220), -20),
            ("Connect a webcam and restart the game.", 0.65, (180, 180, 220), 20),
            ("Press any key to exit.",                 0.55, (140, 140, 180), 60),
        ]
        for text, scale, color, dy in msgs:
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            cv2.putText(canvas, text, (W // 2 - ts[0] // 2, H // 2 + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _cleanup(self) -> None:
        log.info("Cleaning up resources.")
        if self.cap:
            self.cap.release()
        if self.tracker:
            self.tracker.close()
        if self.emg is not None:
            try:
                from emg.emg_interface import close_emg
                close_emg()
            except Exception as err:
                log.warning(f"Error during EMG cleanup: {err}")
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RehabGame().run()
