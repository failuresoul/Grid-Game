"""
renderer.py — OpenCV Rendering Engine

Responsible for all visual output.  Draws each frame in layers:
  1. Background gradient
  2. Obstacle walls (with glow border)
  3. Ideal-path hint line  (Easy mode only)
  4. Trail polyline  (fading history of player path)
  5. Start / End zone markers
  6. Player cursor  (glowing circle, colour reflects speed)
  7. HUD  (time, score, speed, distance)
  8. Pause / Win / Timeout overlays
  9. Camera PiP thumbnail  (top-right corner)

Renders onto a fixed-size NumPy/OpenCV canvas; caller displays with imshow().
No pygame dependency — pure OpenCV.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from game_state import GameState, State
from maze_generator import Level, Rect

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blend(canvas: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    """In-place alpha-blend overlay onto canvas."""
    np.multiply(canvas, 1 - alpha, out=canvas, casting="unsafe")
    np.multiply(overlay, alpha, out=overlay, casting="unsafe")
    np.add(canvas, overlay, out=canvas, casting="unsafe")


def _draw_text(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (220, 240, 255),
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Draw text with an optional dark shadow for legibility on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (origin[0]+1, origin[1]+1),
                    font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def _glow_circle(
    img: np.ndarray,
    centre: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    glow_layers: int = 3,
) -> None:
    """Draw a circle with multiple translucent outer rings to simulate a glow."""
    for i in range(glow_layers, 0, -1):
        glow_r = radius + i * 5
        alpha = 0.15 / i
        overlay = img.copy()
        cv2.circle(overlay, centre, glow_r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, centre, radius, color, -1, cv2.LINE_AA)
    # Bright inner highlight
    hi_r = max(radius // 3, 2)
    hi_c = (centre[0] - radius // 4, centre[1] - radius // 4)
    bright = tuple(min(255, int(c * 1.5)) for c in color)
    cv2.circle(img, hi_c, hi_r, bright, -1, cv2.LINE_AA)


def _draw_star(
    img: np.ndarray,
    centre: Tuple[int, int],
    outer_r: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a 5-pointed star polygon (for the End marker)."""
    points = []
    inner_r = outer_r // 2
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = outer_r if i % 2 == 0 else inner_r
        x = int(centre[0] + r * math.cos(angle))
        y = int(centre[1] + r * math.sin(angle))
        points.append([x, y])
    pts = np.array([points], dtype=np.int32)
    cv2.fillPoly(img, pts, color, cv2.LINE_AA)
    cv2.polylines(img, pts, True, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Renderer class
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    """
    Produces one complete rendered frame per call to `draw()`.
    Returns a NumPy BGR image ready for cv2.imshow().
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.W = canvas_w
        self.H = canvas_h

        # Pre-compute vertical background gradient (dark purple → dark teal)
        self._bg = self._make_gradient(
            config.BG_TOP_COLOR,
            config.BG_BOTTOM_COLOR,
        ).astype(np.uint8)

        # Animated time counter (for pulsing effects)
        self._t: float = 0.0

        log.info(f"Renderer ready  {canvas_w}×{canvas_h}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def draw(
        self,
        game: GameState,
        dt: float,
        pip_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a complete frame.

        Args:
            game:      Current GameState
            dt:        Seconds since last frame (drives animations)
            pip_frame: Optional annotated webcam frame for PiP thumbnail

        Returns:
            BGR image (H × W × 3) ready for cv2.imshow()
        """
        self._t += dt
        canvas = self._bg.copy()

        level = game.level

        # 1. Hint path (Easy mode)
        if game.difficulty_cfg.show_path_hint:
            self._draw_hint_path(canvas, level)

        # 2. Walls
        self._draw_walls(canvas, level)

        # 3. Trail
        self._draw_trail(canvas, game.trail)

        # 4. Start / End markers
        self._draw_start_end(canvas, level)

        # 5. Player
        self._draw_player(canvas, game)

        # 6. HUD
        self._draw_hud(canvas, game)

        # 7. State overlays
        if game.state == State.WAITING:
            self._draw_waiting_overlay(canvas)
        elif game.state == State.PAUSED:
            self._draw_paused_overlay(canvas)
        elif game.state == State.WIN:
            self._draw_win_overlay(canvas, game)
        elif game.state == State.TIMEOUT:
            self._draw_timeout_overlay(canvas, game)

        # 8. Camera PiP
        if pip_frame is not None and config.CAMERA_PIP_ENABLED:
            self._draw_pip(canvas, pip_frame)

        return canvas

    def draw_start_screen(self, selected_difficulty: int) -> np.ndarray:
        """Render the game start / difficulty selection screen."""
        canvas = self._bg.copy()

        W, H = self.W, self.H

        # Title
        title = "Hand Rehabilitation Maze"
        ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
        tx = (W - ts[0]) // 2
        _draw_text(canvas, title, (tx, 100), font_scale=1.4,
                   color=(120, 220, 255), thickness=2)

        subtitle = "Move your hand to control the cursor"
        ss = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        _draw_text(canvas, subtitle, ((W - ss[0]) // 2, 145),
                   font_scale=0.65, color=(160, 200, 230))

        # Difficulty buttons
        labels = {1: ("1  EASY",   config.EASY),
                  2: ("2  MEDIUM", config.MEDIUM),
                  3: ("3  HARD",   config.HARD)}
        colors = {1: (50, 200, 80), 2: (30, 180, 255), 3: (60, 60, 220)}

        for diff, (label, cfg) in labels.items():
            bx = W // 2 - 150
            by = 220 + (diff - 1) * 100
            bw, bh = 300, 60

            # Highlight selected
            if diff == selected_difficulty:
                cv2.rectangle(canvas, (bx-4, by-4), (bx+bw+4, by+bh+4),
                              colors[diff], 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh),
                              colors[diff], -1)
                text_col = (10, 10, 10)
            else:
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh),
                              (40, 40, 60), -1)
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh),
                              colors[diff], 1, cv2.LINE_AA)
                text_col = colors[diff]

            ts2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            _draw_text(canvas, label,
                       (bx + (bw - ts2[0]) // 2, by + (bh + ts2[1]) // 2),
                       font_scale=0.8, color=text_col, thickness=2, shadow=False)

        # Instructions
        instructions = [
            "ENTER / SPACE  →  Start game",
            "P              →  Pause",
            "R              →  Restart",
            "ESC            →  Quit",
            "1 / 2 / 3      →  Change difficulty",
        ]
        for i, line in enumerate(instructions):
            _draw_text(canvas, line, (W // 2 - 200, 560 + i * 30),
                       font_scale=0.5, color=(140, 180, 210))

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  Layer drawing methods
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_walls(self, canvas: np.ndarray, level: Level) -> None:
        """Draw all obstacle walls with a glow border."""
        for (wx, wy, ww, wh) in level.walls:
            # Glow border (blurred semi-transparent)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (wx-3, wy-3), (wx+ww+3, wy+wh+3),
                          config.WALL_BORDER_COLOR, 3)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            # Solid wall body
            cv2.rectangle(canvas, (wx, wy), (wx+ww, wy+wh),
                          config.WALL_COLOR, -1, cv2.LINE_AA)
            # Bright top/left edge for 3-D effect
            bright = tuple(min(255, c + 60) for c in config.WALL_COLOR)
            cv2.line(canvas, (wx, wy), (wx+ww, wy), bright, 2, cv2.LINE_AA)
            cv2.line(canvas, (wx, wy), (wx, wy+wh), bright, 2, cv2.LINE_AA)

    def _draw_trail(
        self,
        canvas: np.ndarray,
        trail: list,
    ) -> None:
        """Draw the player's path history as a fading polyline."""
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = i / n   # older = more transparent
            intensity = int(80 + 120 * alpha)
            color = (intensity, intensity + 40, 255)
            thickness = max(1, int(2 * alpha))
            cv2.line(canvas, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

    def _draw_start_end(self, canvas: np.ndarray, level: Level) -> None:
        """Draw Start (green circle) and End (gold star with pulsing ring)."""
        # Start
        sc = level.start
        sr = level.start_r
        cv2.circle(canvas, sc, sr, config.START_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, sc, sr, (255, 255, 255), 1, cv2.LINE_AA)
        s_label = "START"
        sl_size = cv2.getTextSize(s_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        _draw_text(canvas, s_label,
                   (sc[0] - sl_size[0]//2, sc[1] + sr + 16),
                   font_scale=0.45, color=config.START_COLOR)

        # End — pulsing ring
        ec = level.end
        er = level.end_r
        pulse = int(4 * math.sin(self._t * 3) + 4)
        _draw_star(canvas, ec, er + pulse, config.END_COLOR)
        ring_r = er + pulse + 8
        cv2.circle(canvas, ec, ring_r, config.END_COLOR, 1, cv2.LINE_AA)
        e_label = "GOAL"
        el_size = cv2.getTextSize(e_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        _draw_text(canvas, e_label,
                   (ec[0] - el_size[0]//2, ec[1] + er + pulse + 22),
                   font_scale=0.45, color=config.END_COLOR)

    def _draw_hint_path(self, canvas: np.ndarray, level: Level) -> None:
        """Draw the direct start→end ideal path as a subtle dashed ghost line."""
        sx, sy = level.start
        ex, ey = level.end
        # Dashed line: draw segments
        total = math.dist((sx, sy), (ex, ey))
        if total < 1:
            return
        steps = int(total // 30)
        for i in range(steps):
            t1, t2 = i / steps, (i + 0.5) / steps
            p1 = (int(sx + (ex-sx)*t1), int(sy + (ey-sy)*t1))
            p2 = (int(sx + (ex-sx)*t2), int(sy + (ey-sy)*t2))
            cv2.line(canvas, p1, p2, (50, 100, 50), 1, cv2.LINE_AA)

    def _draw_player(self, canvas: np.ndarray, game: GameState) -> None:
        """Draw the player cursor with glow; colour reflects speed."""
        if game.metrics is not None:
            speed = game.metrics.live_speed()
        else:
            speed = 0.0

        # Lerp colour between slow (teal) and fast (blue-white) based on speed
        speed_norm = min(speed / 300.0, 1.0)
        slow = np.array(config.PLAYER_COLOR_SLOW, dtype=float)
        fast = np.array(config.PLAYER_COLOR_FAST, dtype=float)
        color = tuple(int(c) for c in (slow + (fast - slow) * speed_norm))

        pos = game.player_pos
        if game.state == State.WAITING:
            # Pulse while waiting
            pulse_r = game.radius + int(3 * math.sin(self._t * 4))
            _glow_circle(canvas, pos, pulse_r, color, glow_layers=2)
        else:
            _glow_circle(canvas, pos, game.radius, color, glow_layers=3)

    def _draw_hud(self, canvas: np.ndarray, game: GameState) -> None:
        """Draw the heads-up display: time, difficulty, speed, distance."""
        W = self.W
        y = 24
        # Difficulty badge
        diff_name = game.difficulty_cfg.name.upper()
        _draw_text(canvas, f"DIFFICULTY: {diff_name}", (12, y),
                   font_scale=0.55, color=config.HUD_LABEL_COLOR)

        # Level name
        _draw_text(canvas, game.level.name, (12, y + 26),
                   font_scale=0.45, color=config.HUD_LABEL_COLOR)

        # Elapsed time (right-aligned)
        t = game.elapsed_time
        time_str = f"{int(t//60):02d}:{t%60:05.2f}"
        ts = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        _draw_text(canvas, time_str, (W - ts[0] - 12, y),
                   font_scale=0.75, color=config.HUD_TEXT_COLOR, thickness=2)

        # Distance to goal
        dist = game.distance_to_end
        dist_str = f"  {int(dist)} px to GOAL"
        _draw_text(canvas, dist_str, (W - 200, y + 30),
                   font_scale=0.45, color=config.HUD_LABEL_COLOR)

        # Wall hit counter
        hits_str = f"Wall hits: {game.wall_hit_count}"
        _draw_text(canvas, hits_str, (12, self.H - 16),
                   font_scale=0.45, color=config.HUD_LABEL_COLOR)

        # Controls hint (bottom-right)
        hint = "R=Restart  P=Pause  ESC=Quit"
        hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        _draw_text(canvas, hint, (W - hs[0] - 8, self.H - 10),
                   font_scale=0.4, color=(80, 100, 130))

    # ─────────────────────────────────────────────────────────────────────────
    #  Overlay screens
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_waiting_overlay(self, canvas: np.ndarray) -> None:
        """Semi-transparent overlay shown before the player starts moving."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (10, 15, 30), -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        W, H = self.W, self.H
        pulse = 0.7 + 0.3 * math.sin(self._t * 3)
        color = tuple(int(c * pulse) for c in (100, 220, 255))
        _draw_text(canvas, "SHOW YOUR HAND", (W//2 - 160, H//2 - 30),
                   font_scale=1.0, color=color, thickness=2)
        _draw_text(canvas, "Move away from START to begin", (W//2 - 200, H//2 + 20),
                   font_scale=0.6, color=(160, 200, 230))

    def _draw_paused_overlay(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (10, 15, 30), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        W, H = self.W, self.H
        _draw_text(canvas, "PAUSED", (W//2 - 80, H//2),
                   font_scale=1.4, color=(200, 220, 255), thickness=3)
        _draw_text(canvas, "Press P to continue", (W//2 - 130, H//2 + 50),
                   font_scale=0.65, color=(140, 180, 220))

    def _draw_win_overlay(self, canvas: np.ndarray, game: GameState) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (5, 25, 15), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        W, H = self.W, self.H
        pulse = 0.8 + 0.2 * math.sin(self._t * 5)
        color = tuple(int(c * pulse) for c in config.WIN_COLOR)

        _draw_text(canvas, "GOAL REACHED!", (W//2 - 160, H//2 - 110),
                   font_scale=1.3, color=color, thickness=3)

        # Metrics table
        m = game.final_metrics or {}
        rows = [
            ("Time",              f"{m.get('completion_time_s', 0):.2f} s"),
            ("Path efficiency",   f"{m.get('path_efficiency', 0)*100:.1f}%"),
            ("Norm. jerk score",  f"{m.get('normalised_jerk', 0):.2f}"),
            ("Tremor index",      f"{m.get('tremor_index', 0)*100:.1f}%"),
            ("Peak speed",        f"{m.get('peak_speed_px_s', 0):.0f} px/s"),
            ("ROM  W × H",        f"{m.get('rom_width_px',0):.0f} × {m.get('rom_height_px',0):.0f} px"),
            ("Wall hits",         str(game.wall_hit_count)),
        ]
        box_x, box_y = W//2 - 220, H//2 - 80
        cv2.rectangle(canvas, (box_x-10, box_y-10),
                      (box_x + 440, box_y + len(rows)*34 + 10),
                      (20, 40, 30), -1)
        cv2.rectangle(canvas, (box_x-10, box_y-10),
                      (box_x + 440, box_y + len(rows)*34 + 10),
                      config.WIN_COLOR, 1)
        for i, (label, val) in enumerate(rows):
            ry = box_y + i * 34 + 24
            _draw_text(canvas, label, (box_x, ry),
                       font_scale=0.55, color=config.HUD_LABEL_COLOR)
            _draw_text(canvas, val, (box_x + 280, ry),
                       font_scale=0.6, color=config.HUD_TEXT_COLOR, thickness=1)

        _draw_text(canvas, "Press R to restart  |  ESC to quit",
                   (W//2 - 200, H - 60),
                   font_scale=0.55, color=(140, 200, 160))

    def _draw_timeout_overlay(self, canvas: np.ndarray, game: GameState) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (20, 5, 5), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        W, H = self.W, self.H
        _draw_text(canvas, "TIME'S UP!", (W//2 - 120, H//2 - 30),
                   font_scale=1.3, color=config.FAIL_COLOR, thickness=3)
        _draw_text(canvas, f"Time: {game.elapsed_time:.1f}s   Wall hits: {game.wall_hit_count}",
                   (W//2 - 180, H//2 + 20), font_scale=0.65, color=(180, 140, 200))
        _draw_text(canvas, "Press R to restart  |  ESC to quit",
                   (W//2 - 200, H - 60), font_scale=0.55, color=(140, 120, 180))

    # ─────────────────────────────────────────────────────────────────────────
    #  Camera PiP
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_pip(self, canvas: np.ndarray, cam_frame: np.ndarray) -> None:
        """Composite a small webcam thumbnail into the top-right corner."""
        pw, ph = config.PIP_WIDTH, config.PIP_HEIGHT
        try:
            thumb = cv2.resize(cam_frame, (pw, ph))
        except Exception:
            return

        x1 = self.W - pw - 10
        y1 = 10
        x2 = x1 + pw
        y2 = y1 + ph

        # Semi-transparent backing
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1-3, y1-3), (x2+3, y2+3), (30, 40, 60), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        canvas[y1:y2, x1:x2] = thumb
        cv2.rectangle(canvas, (x1-1, y1-1), (x2+1, y2+1),
                      (80, 130, 200), 1, cv2.LINE_AA)
        _draw_text(canvas, "Camera", (x1, y2 + 14),
                   font_scale=0.38, color=(100, 140, 180))

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _make_gradient(
        self,
        top_color: Tuple[int, int, int],
        bottom_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Generate a vertical linear gradient background image."""
        gradient = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for y in range(self.H):
            t = y / self.H
            for c in range(3):
                gradient[y, :, c] = top_color[c] * (1 - t) + bottom_color[c] * t
        return gradient
