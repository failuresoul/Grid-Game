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
        self.level_index:  int  = 0
        self._start_screen: bool = True

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

        # ── EMG (disabled) ────────────────────────────────────────────────────
        self.emg = None
        # if config.EMG_ENABLED:
        #     self.emg = EMGInterface(
        #         port=config.EMG_PORT,
        #         device=config.EMG_DEVICE,
        #         sample_rate=config.EMG_SAMPLE_RATE,
        #         channels=config.EMG_CHANNELS,
        #     )
        #     if not self.emg.connect():
        #         log.warning("EMG device failed to connect — continuing without EMG.")
        #         self.emg = None

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
        metrics  = MetricsCollector(start=level.start, end=level.end, min_path_distance=level.min_path_distance)
        self.engine = GameEngine(level=level, difficulty_cfg=diff_cfg,
                                 metrics_collector=metrics)
        if self.tracker:
            self.tracker.reset_smoothing()
        log.info(f"New game: {diff_cfg.name}  |  {level.name}")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Track mouse position for development / testing fallback."""
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self._mouse_pos = (float(x), float(y))

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

                # ── Start screen ──────────────────────────────────────────────
                if self._start_screen:
                    canvas = self.renderer.draw_start_screen(self.difficulty)
                    self._composite_pip(canvas, pip_frame)
                    cv2.imshow(self.WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_start_key(key):
                        break
                    if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                # ── Gameplay ──────────────────────────────────────────────────
                if self.engine is None:
                    self._new_game()

                self.engine.update(cursor, dt)

                # ── EMG integration hook (disabled) ───────────────────────────
                # if self.emg and self.emg.is_connected:
                #     activation = self.emg.get_muscle_activation()
                #     # TODO: map to game mechanic (speed boost, gate unlock, etc.)

                self._maybe_save_metrics()

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

    def _handle_start_key(self, key: int) -> bool:
        """Returns True if the application should quit."""
        if key == 27:
            return True
        if key in (ord('1'), ord('2'), ord('3')):
            self.difficulty = int(chr(key))
        elif key in (ord('l'), ord('L')):
            self._cycle_landmark()
        elif key in (ord('d'), ord('D')):
            self.debug_mode = not self.debug_mode
            log.info(f"Debug coordinates overlay toggled: {self.debug_mode}")
        elif key in (ord('m'), ord('M')):
            self.mouse_fallback_enabled = not self.mouse_fallback_enabled
            log.info(f"Mouse fallback toggled: {self.mouse_fallback_enabled}")
        elif key in (13, 32):   # ENTER or SPACE
            self._start_screen = False
            if self.cap and self.cap.isOpened():
                self._build_tracker()
            self._new_game()
        return False

    def _handle_game_key(self, key: int) -> bool:
        """Returns True if the application should quit."""
        if key == 27:
            return True
        elif key in (ord('r'), ord('R')):
            self._new_game()
        elif key in (ord('n'), ord('N')):
            total = self.maze_gen.level_count(self.difficulty)
            self.level_index = (self.level_index + 1) % total
            self._new_game()
        elif key in (ord('p'), ord('P')):
            if self.engine:
                self.engine.toggle_pause()
        elif key in (ord('l'), ord('L')):
            self._cycle_landmark()
        elif key in (ord('d'), ord('D')):
            self.debug_mode = not self.debug_mode
            log.info(f"Debug coordinates overlay toggled: {self.debug_mode}")
        elif key in (ord('m'), ord('M')):
            self.mouse_fallback_enabled = not self.mouse_fallback_enabled
            log.info(f"Mouse fallback toggled: {self.mouse_fallback_enabled}")
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
        elif key in (13, 32):
            self._start_screen = True
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

        won      = self.engine.state == GameState.WIN
        should   = (won and config.SAVE_METRICS_ON_WIN) or \
                   (not won and config.SAVE_METRICS_ON_TIMEOUT)

        if should and self.engine.metrics and self.engine.final_metrics:
            diff_cfg = config.DIFFICULTIES[self.difficulty]
            self.engine.metrics.save_csv(
                self.engine.final_metrics,
                difficulty=diff_cfg.name,
                level_name=self.engine.level.name,
                maze_seed=self.engine.level.seed,
            )
        self.engine._metrics_saved = True

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
