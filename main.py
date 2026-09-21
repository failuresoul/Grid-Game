"""
main.py — Entry Point: Stroke Rehabilitation Maze Game

Run with:
    python main.py

Keyboard controls (while window is focused):
    1 / 2 / 3     Select difficulty (Easy / Medium / Hard)
    ENTER / SPACE Start game (from start screen)
    R             Restart current level
    N             Next level (same difficulty)
    P             Pause / Resume
    C             Toggle camera PiP
    ESC           Quit

Architecture:
    main.py → HandTracker → (cx,cy)
                          → GameState.update() → collision / win detection
                          → MetricsCollector.record()
              Renderer.draw(game, dt, pip_frame) → canvas
              cv2.imshow("Rehab Game", canvas)

EMG:
    Disabled (config.EMG_ENABLED = False).
    See emg_interface.py for re-activation instructions.
"""

from __future__ import annotations
import logging
import time
import sys
import os

import cv2
import numpy as np

import config
from hand_tracker   import HandTracker
from maze_generator import MazeGenerator
from game_state     import GameState, State
from renderer       import Renderer
from metrics        import MetricsCollector

# ── Optional EMG (disabled by default) ───────────────────────────────────────
if config.EMG_ENABLED:
    from emg_interface import EMGInterface


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
#  Game Application
# ─────────────────────────────────────────────────────────────────────────────

class RehabGame:
    """
    Top-level application class.  Owns the main loop and wires all subsystems.
    """

    WINDOW_NAME = "Hand Rehabilitation Maze"

    def __init__(self) -> None:
        # ── Selected difficulty and level index ───────────────────────────────
        self.difficulty: int  = config.DEFAULT_DIFFICULTY
        self.level_index: int = 0

        # ── Game screen (START_SCREEN shown first) ─────────────────────────────
        self._on_start_screen: bool = True

        # ── Subsystems (created in _setup) ────────────────────────────────────
        self.cap:       cv2.VideoCapture | None = None
        self.tracker:   HandTracker      | None = None
        self.maze_gen:  MazeGenerator           = MazeGenerator()
        self.renderer:  Renderer                = Renderer()
        self.game:      GameState        | None = None

        # ── EMG (disabled) ────────────────────────────────────────────────────
        self.emg = None
        # if config.EMG_ENABLED:
        #     self.emg = EMGInterface(
        #         port=config.EMG_PORT,
        #         device=config.EMG_DEVICE,
        #         sample_rate=config.EMG_SAMPLE_RATE,
        #         channels=config.EMG_CHANNELS,
        #     )
        #     connected = self.emg.connect()
        #     if not connected:
        #         log.warning("EMG device failed to connect — continuing without EMG.")
        #         self.emg = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Setup
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_camera(self) -> bool:
        """Open webcam and set resolution.  Returns False if unavailable."""
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback: try without backend hint
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            log.error("Could not open webcam — no camera detected.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        log.info(f"Camera opened (index={config.CAMERA_INDEX})")
        return True

    def _setup_tracker(self) -> None:
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
        """Create a fresh GameState for the current difficulty + level."""
        diff_cfg = config.DIFFICULTIES[self.difficulty]
        level    = self.maze_gen.get_level(self.difficulty, self.level_index)

        metrics = MetricsCollector(start=level.start, end=level.end)

        self.game = GameState(
            level=level,
            difficulty_cfg=diff_cfg,
            metrics_collector=metrics,
        )

        # Reset tracker smoothing so cursor snaps to hand on restart
        if self.tracker:
            self.tracker.reset_smoothing()

        log.info(f"New game: diff={diff_cfg.name}  level={level.name}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the application and block until the user quits."""
        log.info("=== Rehab Maze Game starting ===")

        if not self._setup_camera():
            self._run_no_camera()
            return

        self._setup_tracker()

        # Create OpenCV window (resizable for different monitor sizes)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, config.CANVAS_WIDTH, config.CANVAS_HEIGHT)

        prev_time = time.perf_counter()

        try:
            while True:
                # ── Frame timing ──────────────────────────────────────────────
                now = time.perf_counter()
                dt  = min(now - prev_time, 0.1)   # cap dt to 100 ms
                prev_time = now

                # ── Read webcam frame ─────────────────────────────────────────
                ok, cam_frame = self.cap.read()
                if not ok:
                    log.warning("Empty camera frame — retrying.")
                    time.sleep(0.01)
                    continue

                # Flip horizontally so it's a mirror image for PiP
                cam_frame_display = cv2.flip(cam_frame, 1)

                # ── Hand tracking ─────────────────────────────────────────────
                pip_frame: np.ndarray | None = None
                cursor = None

                if self.tracker is not None:
                    cursor = self.tracker.process(cam_frame)
                    pip_frame = self.tracker.annotated_frame

                # ── Start screen ──────────────────────────────────────────────
                if self._on_start_screen:
                    canvas = self.renderer.draw_start_screen(self.difficulty)
                    if pip_frame is not None and config.CAMERA_PIP_ENABLED:
                        from renderer import _draw_text
                        # Draw PiP manually on start screen
                        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
                        try:
                            thumb = cv2.resize(pip_frame, (pw, ph))
                            x1 = config.CANVAS_WIDTH - pw - 10
                            y1 = 10
                            canvas[y1:y1+ph, x1:x1+pw] = thumb
                            cv2.rectangle(canvas,
                                          (x1-1, y1-1), (x1+pw+1, y1+ph+1),
                                          (80, 130, 200), 1, cv2.LINE_AA)
                        except Exception:
                            pass
                    cv2.imshow(self.WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key_start_screen(key)
                    continue

                # ── Active gameplay ───────────────────────────────────────────
                if self.game is None:
                    self._new_game()

                assert self.game is not None

                # Update game state
                self.game.update(cursor, dt)

                # ── EMG integration point (disabled) ──────────────────────────
                # if self.emg and self.emg.is_connected:
                #     activation = self.emg.get_muscle_activation()
                #     # TODO: map activation to game mechanic (e.g. speed boost)
                #     pass

                # ── Auto-save metrics on session end ──────────────────────────
                self._maybe_save_metrics()

                # ── Render ────────────────────────────────────────────────────
                canvas = self.renderer.draw(self.game, dt, pip_frame)
                cv2.imshow(self.WINDOW_NAME, canvas)

                # ── Keyboard input ────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                should_quit = self._handle_key_game(key)
                if should_quit:
                    break

                # ── Window closed by user (X button) ─────────────────────────
                if cv2.getWindowProperty(
                    self.WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

        finally:
            self._cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    #  Keyboard handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_key_start_screen(self, key: int) -> None:
        if key == 27:   # ESC
            self._quit()
        elif key in (ord('1'), ord('2'), ord('3')):
            self.difficulty = int(chr(key))
            log.info(f"Difficulty selected: {config.DIFFICULTIES[self.difficulty].name}")
        elif key in (13, 32):   # ENTER or SPACE
            self._on_start_screen = False
            self._setup_tracker()
            self._new_game()

    def _handle_key_game(self, key: int) -> bool:
        """Returns True if the game should quit."""
        if key == 27:   # ESC → quit
            return True

        elif key == ord('r') or key == ord('R'):
            # Restart same level
            self._new_game()

        elif key == ord('n') or key == ord('N'):
            # Next level in the same difficulty
            total = self.maze_gen.level_count(self.difficulty)
            self.level_index = (self.level_index + 1) % total
            self._new_game()
            log.info(f"Next level: index={self.level_index}")

        elif key == ord('p') or key == ord('P'):
            if self.game:
                self.game.toggle_pause()

        elif key in (ord('1'), ord('2'), ord('3')):
            new_diff = int(chr(key))
            if new_diff != self.difficulty:
                self.difficulty  = new_diff
                self.level_index = 0
                self._setup_tracker()   # update smoothing for new difficulty
                self._new_game()
                log.info(f"Switched difficulty: {config.DIFFICULTIES[self.difficulty].name}")

        elif key == ord('c') or key == ord('C'):
            # Toggle camera PiP
            config.CAMERA_PIP_ENABLED = not config.CAMERA_PIP_ENABLED
            log.info(f"Camera PiP: {config.CAMERA_PIP_ENABLED}")

        elif key == 13 or key == 32:   # ENTER / SPACE — go back to start screen
            self._on_start_screen = True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_save_metrics(self) -> None:
        """Save session CSV exactly once when the game ends."""
        if self.game is None or not self.game.is_finished:
            return
        if self.game.final_metrics is None:
            return

        # Use a flag to prevent saving every frame after session ends
        if getattr(self.game, "_metrics_saved", False):
            return

        diff_cfg = config.DIFFICULTIES[self.difficulty]
        won = self.game.state == State.WIN

        should_save = (
            (won and config.SAVE_METRICS_ON_WIN) or
            (not won and config.SAVE_METRICS_ON_TIMEOUT)
        )

        if should_save and self.game.metrics:
            path = self.game.metrics.save_csv(
                self.game.final_metrics,
                difficulty=diff_cfg.name,
                level_name=self.game.level.name,
            )
            if path:
                log.info(f"Session saved → {path}")

        self.game._metrics_saved = True   # type: ignore[attr-defined]

    def _run_no_camera(self) -> None:
        """Fallback mode: show a message if the camera can't be opened."""
        log.warning("Running in NO-CAMERA mode — showing error screen.")
        canvas = np.zeros((config.CANVAS_HEIGHT, config.CANVAS_WIDTH, 3), dtype=np.uint8)
        cv2.putText(canvas, "ERROR: No camera detected",
                    (config.CANVAS_WIDTH//2 - 220, config.CANVAS_HEIGHT//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Connect a webcam and restart the game.",
                    (config.CANVAS_WIDTH//2 - 270, config.CANVAS_HEIGHT//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press any key to exit.",
                    (config.CANVAS_WIDTH//2 - 120, config.CANVAS_HEIGHT//2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 180), 1, cv2.LINE_AA)
        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _quit(self) -> None:
        self._cleanup()
        sys.exit(0)

    def _cleanup(self) -> None:
        log.info("Cleaning up resources.")
        if self.cap is not None:
            self.cap.release()
        if self.tracker is not None:
            self.tracker.close()
        # if self.emg is not None:
        #     self.emg.disconnect()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = RehabGame()
    app.run()
