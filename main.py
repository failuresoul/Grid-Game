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
import logging
import sys
import time
from typing import Optional, Tuple


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
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class RehabGame:
    """Top-level application — owns the main loop and wires all subsystems."""

    WINDOW_NAME = "Hand Rehabilitation Maze"

    def __init__(self) -> None:
        self.difficulty:   int  = config.DEFAULT_DIFFICULTY
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

        # ── EMG (disabled) ────────────────────────────────────────────────────
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
        metrics  = MetricsCollector(
            start=level.start,
            end=level.end,
            min_path_distance=level.min_path_distance,
            optimal_waypoints=getattr(level, "optimal_waypoints", None),
        )
        self.engine = GameEngine(level=level, difficulty_cfg=diff_cfg,
                                 metrics_collector=metrics)
        self.app_state = GameState.READY
        if self.tracker:
            self.tracker.reset_smoothing()
        log.info(f"New game: {diff_cfg.name}  |  {level.name} (READY)")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Track mouse position and handle interactive UI button clicks."""
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self._mouse_pos = (float(x), float(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.app_state in (GameState.COMPLETED, GameState.RESULTS):
                self._handle_results_click(float(x), float(y))
            elif self.app_state == GameState.HISTORY:
                self._handle_history_click(float(x), float(y))

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
                if b_id == "PLAY_AGAIN":
                    if self.engine:
                        self.engine.restart()
                        self.app_state = GameState.READY
                    else:
                        self._new_game()
                elif b_id == "NEXT_LEVEL":
                    total = self.maze_gen.level_count(self.difficulty)
                    self.level_index = (self.level_index + 1) % total
                    self._new_game()
                elif b_id == "LEVEL_SELECT":
                    self.app_state = GameState.LEVEL_SELECT
                elif b_id == "MAIN_MENU":
                    self.app_state = GameState.MENU
                elif b_id == "EXIT":
                    sys.exit(0)
                return True
        return False

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
                    cursor    = self.tracker.process(cam_frame)
                    pip_frame = self.tracker.annotated_frame

                is_mouse = False
                if cursor is None and self.mouse_fallback_enabled and self._mouse_pos is not None:
                    cursor = self._mouse_pos
                    is_mouse = True
                self._using_mouse = is_mouse

                # ── 1. MENU State ─────────────────────────────────────────────
                if self.app_state == GameState.MENU:
                    canvas = self.renderer.draw_start_screen(self.difficulty)
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
                    canvas, _ = self.renderer.draw_history_screen(
                        sessions=self.history_sessions,
                        current_filter=self.history_filter,
                        page=self.history_page,
                        mouse_pos=self._mouse_pos,
                    )
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
                    diff_cfg = config.DIFFICULTIES[self.difficulty]
                    total_levels = self.maze_gen.level_count(self.difficulty)
                    prev_lvl = self.maze_gen.get_level(self.difficulty, self.level_index)
                    canvas = self.renderer.draw_level_select_screen(
                        difficulty_name=diff_cfg.name,
                        level_index=self.level_index,
                        total_levels=total_levels,
                        level_name=prev_lvl.name,
                        min_path_distance=prev_lvl.minimum_path_distance,
                        wall_count=len(prev_lvl.walls),
                    )
                    self._composite_pip(canvas, pip_frame)
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
                )
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
        elif key in (13, 32) and self.app_state in (GameState.COMPLETED, GameState.RESULTS):
            # Advance from results to next level
            total = self.maze_gen.level_count(self.difficulty)
            self.level_index = (self.level_index + 1) % total
            self._new_game()
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
    ) -> None:
        """Blit a PiP thumbnail onto the start screen canvas."""
        if pip_frame is None or not config.CAMERA_PIP_ENABLED:
            return
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        try:
            thumb = cv2.resize(pip_frame, (pw, ph))
            x1    = config.CANVAS_WIDTH  - pw - 10
            y1    = 10
            canvas[y1:y1 + ph, x1:x1 + pw] = thumb
            cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x1 + pw + 1, y1 + ph + 1),
                          (80, 130, 200), 1, cv2.LINE_AA)
        except Exception:
            pass

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
        # if self.emg:
        #     self.emg.disconnect()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RehabGame().run()
