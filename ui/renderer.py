"""
ui/renderer.py — OpenCV Rendering Engine

Composes every visual layer into a single BGR frame each call to draw().
Delegates all full-canvas screen and overlay drawing to ui.screens.

Layer order (bottom → top):
  1. Background gradient
  2. Hint path          (Easy mode dashed line)
  3. Obstacle walls     (glow border + solid body)
  4. Path trail         (fading polyline)
  5. Start / End zones  (circle + pulsing star)
  6. Player cursor      (speed-coloured glow circle)
  7. HUD                (time, difficulty, distance, wall hits)
  8. State overlay      (WAITING / PAUSED / WIN / TIMEOUT)
  9. Camera PiP         (annotated webcam thumbnail)

Dependency chain:
    ui.renderer → ui.screens, config
    (GameEngine passed as runtime argument — not imported at module level)
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

import config
from ui.screens import (
    draw_text, glow_circle, draw_star,
    draw_start_screen,
    draw_waiting_overlay, draw_paused_overlay,
    draw_win_overlay, draw_timeout_overlay,
)

if TYPE_CHECKING:
    from game.game_engine import GameEngine, GameState

log = logging.getLogger(__name__)


class Renderer:
    """
    Produces one complete rendered frame per call to draw().

    Usage:
        renderer = Renderer()
        canvas   = renderer.draw(engine, dt, pip_frame)
        cv2.imshow("Rehab Game", canvas)
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.W = canvas_w
        self.H = canvas_h

        # Pre-bake the gradient background (expensive to compute every frame)
        self._bg = self._make_gradient(
            config.BG_TOP_COLOR,
            config.BG_BOTTOM_COLOR,
        ).astype(np.uint8)

        self._t: float = 0.0   # animation clock (seconds)

        log.info(f"Renderer ready  {canvas_w}×{canvas_h}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def draw(
        self,
        engine:    "GameEngine",
        dt:        float,
        pip_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a complete gameplay frame.

        Args:
            engine:    Current GameEngine.
            dt:        Seconds since the last frame (drives animations).
            pip_frame: Optional annotated webcam frame for PiP thumbnail.

        Returns:
            BGR ndarray (H × W × 3) ready for cv2.imshow().
        """
        from game.game_engine import GameState   # local import avoids circular at module level

        self._t += dt
        canvas = self._bg.copy()
        level  = engine.level

        if engine.difficulty_cfg.show_path_hint:
            self._draw_hint_path(canvas, level)

        self._draw_walls(canvas, level)
        self._draw_trail(canvas, engine.trail)
        self._draw_start_end(canvas, level)
        self._draw_player(canvas, engine)
        self._draw_hud(canvas, engine)

        # State overlays
        state = engine.state
        if state == GameState.WAITING:
            draw_waiting_overlay(canvas, self.W, self.H, self._t)
        elif state == GameState.PAUSED:
            draw_paused_overlay(canvas, self.W, self.H)
        elif state == GameState.WIN:
            draw_win_overlay(
                canvas, self.W, self.H, self._t,
                engine.final_metrics or {},
                engine.wall_hit_count,
                engine.elapsed_time,
            )
        elif state == GameState.TIMEOUT:
            draw_timeout_overlay(
                canvas, self.W, self.H,
                engine.elapsed_time,
                engine.wall_hit_count,
            )

        if pip_frame is not None and config.CAMERA_PIP_ENABLED:
            self._draw_pip(canvas, pip_frame)

        return canvas

    def draw_start_screen(self, selected_difficulty: int) -> np.ndarray:
        """Return a rendered start / difficulty-selection screen."""
        canvas = self._bg.copy()
        draw_start_screen(canvas, selected_difficulty, self.W, self.H)
        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  Layer drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_walls(self, canvas: np.ndarray, level) -> None:
        for (wx, wy, ww, wh) in level.walls:
            # Glow border
            overlay = canvas.copy()
            cv2.rectangle(overlay, (wx - 3, wy - 3), (wx + ww + 3, wy + wh + 3),
                          config.WALL_BORDER_COLOR, 3)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            # Solid fill
            cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh),
                          config.WALL_COLOR, -1, cv2.LINE_AA)
            # 3-D highlight (top & left edges)
            bright = tuple(min(255, c + 60) for c in config.WALL_COLOR)
            cv2.line(canvas, (wx, wy),      (wx + ww, wy),      bright, 2, cv2.LINE_AA)
            cv2.line(canvas, (wx, wy),      (wx, wy + wh),      bright, 2, cv2.LINE_AA)

    def _draw_trail(self, canvas: np.ndarray, trail: list) -> None:
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha     = i / n
            intensity = int(80 + 120 * alpha)
            color     = (intensity, intensity + 40, 255)
            thickness = max(1, int(2 * alpha))
            cv2.line(canvas, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    def _draw_start_end(self, canvas: np.ndarray, level) -> None:
        # Start zone
        sc, sr = level.start, level.start_r
        cv2.circle(canvas, sc, sr, config.START_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, sc, sr, (255, 255, 255), 1, cv2.LINE_AA)
        sl = "START"
        slw = cv2.getTextSize(sl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        draw_text(canvas, sl, (sc[0] - slw[0] // 2, sc[1] + sr + 16),
                  font_scale=0.45, color=config.START_COLOR)

        # End zone — pulsing star
        ec, er = level.end, level.end_r
        pulse  = int(4 * math.sin(self._t * 3) + 4)
        draw_star(canvas, ec, er + pulse, config.END_COLOR)
        cv2.circle(canvas, ec, er + pulse + 8, config.END_COLOR, 1, cv2.LINE_AA)
        el = "GOAL"
        elw = cv2.getTextSize(el, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        draw_text(canvas, el, (ec[0] - elw[0] // 2, ec[1] + er + pulse + 22),
                  font_scale=0.45, color=config.END_COLOR)

    def _draw_hint_path(self, canvas: np.ndarray, level) -> None:
        sx, sy = level.start
        ex, ey = level.end
        total  = math.dist((sx, sy), (ex, ey))
        if total < 1:
            return
        steps = int(total // 30)
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 0.5) / steps
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            p2 = (int(sx + (ex - sx) * t2), int(sy + (ey - sy) * t2))
            cv2.line(canvas, p1, p2, (50, 100, 50), 1, cv2.LINE_AA)

    def _draw_player(self, canvas: np.ndarray, engine: "GameEngine") -> None:
        from game.game_engine import GameState

        speed = engine.metrics.live_speed() if engine.metrics else 0.0
        speed_norm = min(speed / 300.0, 1.0)
        slow  = np.array(config.PLAYER_COLOR_SLOW, dtype=float)
        fast  = np.array(config.PLAYER_COLOR_FAST, dtype=float)
        color = tuple(int(c) for c in (slow + (fast - slow) * speed_norm))

        pos = engine.player_pos
        if engine.state == GameState.WAITING:
            r = engine.radius + int(3 * math.sin(self._t * 4))
            glow_circle(canvas, pos, r, color, glow_layers=2)
        else:
            glow_circle(canvas, pos, engine.radius, color, glow_layers=3)

    def _draw_hud(self, canvas: np.ndarray, engine: "GameEngine") -> None:
        W, H = self.W, self.H
        y    = 24

        draw_text(canvas, f"DIFFICULTY: {engine.difficulty_cfg.name.upper()}",
                  (12, y), font_scale=0.55, color=config.HUD_LABEL_COLOR)
        draw_text(canvas, engine.level.name,
                  (12, y + 26), font_scale=0.45, color=config.HUD_LABEL_COLOR)

        # Elapsed time (right-aligned)
        t        = engine.elapsed_time
        time_str = f"{int(t // 60):02d}:{t % 60:05.2f}"
        ts       = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        draw_text(canvas, time_str, (W - ts[0] - 12, y),
                  font_scale=0.75, color=config.HUD_TEXT_COLOR, thickness=2)

        dist_str = f"  {int(engine.distance_to_end)} px to GOAL"
        draw_text(canvas, dist_str, (W - 200, y + 30),
                  font_scale=0.45, color=config.HUD_LABEL_COLOR)

        draw_text(canvas, f"Wall hits: {engine.wall_hit_count}",
                  (12, H - 16), font_scale=0.45, color=config.HUD_LABEL_COLOR)

        hint = "R=Restart  P=Pause  ESC=Quit"
        hs   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        draw_text(canvas, hint, (W - hs[0] - 8, H - 10),
                  font_scale=0.4, color=(80, 100, 130))

    def _draw_pip(self, canvas: np.ndarray, cam_frame: np.ndarray) -> None:
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        try:
            thumb = cv2.resize(cam_frame, (pw, ph))
        except Exception:
            return
        x1, y1 = self.W - pw - 10, 10
        x2, y2 = x1 + pw, y1 + ph

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (30, 40, 60), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        canvas[y1:y2, x1:x2] = thumb
        cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                      (80, 130, 200), 1, cv2.LINE_AA)
        draw_text(canvas, "Camera", (x1, y2 + 14),
                  font_scale=0.38, color=(100, 140, 180))

    # ─────────────────────────────────────────────────────────────────────────
    #  Background gradient
    # ─────────────────────────────────────────────────────────────────────────

    def _make_gradient(
        self,
        top:    Tuple[int, int, int],
        bottom: Tuple[int, int, int],
    ) -> np.ndarray:
        gradient = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for y in range(self.H):
            t = y / self.H
            for c in range(3):
                gradient[y, :, c] = top[c] * (1 - t) + bottom[c] * t
        return gradient
