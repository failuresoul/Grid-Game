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
    draw_start_beacon, draw_target_rings,
    draw_start_screen, draw_level_select_screen,
    draw_waiting_overlay, draw_ready_overlay, draw_paused_overlay,
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
        engine:            "GameEngine",
        dt:                float,
        pip_frame:         Optional[np.ndarray] = None,
        is_mouse_fallback: bool = False,
        landmark_name:     str  = "",
        raw_coords:        Optional[Tuple[float, float]] = None,
        debug_mode:        bool = False,
        algo_name:         str  = "",
        mouse_pos:         Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Render a complete gameplay frame.

        Args:
            engine:            Current GameEngine.
            dt:                Seconds since the last frame (drives animations).
            pip_frame:         Optional annotated webcam frame for PiP thumbnail.
            is_mouse_fallback: True if mouse cursor fallback is actively controlling player.
            landmark_name:     Name of currently tracked hand landmark (e.g. INDEX_TIP).
            raw_coords:        Raw unfiltered canvas coordinates before smoothing.
            debug_mode:        Whether to draw the real-time telemetry debug overlay.
            algo_name:         Name of active smoothing algorithm (e.g. ONE_EURO / EMA).

        Returns:
            BGR ndarray (H × W × 3) ready for cv2.imshow().
        """
        from game.game_engine import GameState   # local import avoids circular at module level

        self._t += dt
        canvas = self._bg.copy()
        level  = engine.level

        # Minimum path displayed visually strictly in debug mode
        if debug_mode:
            self._draw_minimum_path(canvas, level)

        self._draw_walls(canvas, level)
        self._draw_trail(canvas, engine.trail)
        self._draw_start_end(canvas, level)
        self._draw_player(canvas, engine)
        # Only render gameplay HUD when active, not during results screen
        state = engine.state
        if state not in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN):
            self._draw_hud(canvas, engine, is_mouse_fallback, landmark_name)

        # Real-time telemetry debug overlay (Raw vs. Smoothed X,Y)
        if debug_mode:
            self._draw_debug_overlay(canvas, engine, raw_coords, algo_name)

        # State overlays
        if state in (GameState.READY, GameState.WAITING):
            draw_waiting_overlay(canvas, self.W, self.H, self._t)
        elif state == GameState.PAUSED:
            draw_paused_overlay(canvas, self.W, self.H)
        elif state in (GameState.COMPLETED, GameState.RESULTS, GameState.WIN):
            draw_win_overlay(
                canvas=canvas,
                W=self.W,
                H=self.H,
                anim_t=self._t,
                final_metrics=engine.final_metrics or {},
                wall_hits=engine.wall_hit_count,
                elapsed=engine.elapsed_time,
                level=engine.level,
                trajectory=engine.full_trajectory,
                difficulty_name=getattr(engine.difficulty_cfg, "name", "EASY"),
                mouse_pos=mouse_pos,
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
        """Return a rendered start / difficulty-selection screen (MENU state)."""
        canvas = self._bg.copy()
        draw_start_screen(canvas, selected_difficulty, self.W, self.H)
        return canvas

    def draw_level_select_screen(
        self,
        difficulty_name:   str,
        level_index:       int,
        total_levels:      int,
        level_name:        str,
        min_path_distance: float,
        wall_count:        int,
    ) -> np.ndarray:
        """Return a rendered level preview / selection screen (LEVEL_SELECT state)."""
        canvas = self._bg.copy()
        draw_level_select_screen(
            canvas=canvas,
            difficulty_name=difficulty_name,
            level_index=level_index,
            total_levels=total_levels,
            level_name=level_name,
            min_path_distance=min_path_distance,
            wall_count=wall_count,
            anim_t=self._t,
        )
        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  Layer drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_walls(self, canvas: np.ndarray, level) -> None:
        """Draw clean, high-contrast geometric architectural obstacles with precision borders."""
        for obs in level.walls:
            if hasattr(obs, "is_polygon") and obs.is_polygon:
                pts = np.array(obs.points, dtype=np.int32).reshape((-1, 1, 2))
                # Solid fill
                cv2.fillPoly(canvas, [pts], color=config.WALL_COLOR, lineType=cv2.LINE_AA)
                # Precision hairline border
                cv2.polylines(canvas, [pts], isClosed=True, color=config.WALL_BORDER_COLOR,
                              thickness=1, lineType=cv2.LINE_AA)
            else:
                if hasattr(obs, "as_rect"):
                    wx, wy, ww, wh = obs.as_rect()
                elif isinstance(obs, (tuple, list)) and len(obs) == 4:
                    wx, wy, ww, wh = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])
                else:
                    wx, wy, ww, wh = int(obs.x), int(obs.y), int(obs.w), int(obs.h)

                # Solid fill
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh),
                              config.WALL_COLOR, -1, cv2.LINE_AA)
                # Precision hairline border
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh),
                              config.WALL_BORDER_COLOR, 1, cv2.LINE_AA)

    def _draw_trail(self, canvas: np.ndarray, trail: list) -> None:
        """Draw smooth continuous trajectory line showing patient movement path."""
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = i / n
            # Subtle gradient from muted slate-blue to clinical sky-blue
            color = (
                int(140 + 75 * alpha),
                int(160 + 20 * alpha),
                int(190 - 105 * alpha),
            )
            p1 = (int(round(trail[i - 1][0])), int(round(trail[i - 1][1])))
            p2 = (int(round(trail[i][0])), int(round(trail[i][1])))
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    def _draw_start_end(self, canvas: np.ndarray, level) -> None:
        """Render clinical START beacon and END target bullseye."""
        # 1. Start point
        sc, sr = level.start, level.start_r
        draw_start_beacon(canvas, sc, sr, config.START_COLOR)
        sl = "START"
        slw = cv2.getTextSize(sl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        draw_text(canvas, sl, (sc[0] - slw[0] // 2, sc[1] + sr + 15),
                  font_scale=0.42, color=(160, 220, 180), thickness=1)

        # 2. End point (target bullseye with gentle, calm breathing pulse)
        ec, er = level.end, level.end_r
        pulse = int(round(1.5 * math.sin(self._t * 2.5)))
        draw_target_rings(canvas, ec, er, config.END_COLOR, pulse_r=pulse)
        el = "END"
        elw = cv2.getTextSize(el, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        draw_text(canvas, el, (ec[0] - elw[0] // 2, ec[1] + er + pulse + 17),
                  font_scale=0.42, color=(140, 210, 255), thickness=1)

    def _draw_minimum_path(self, canvas: np.ndarray, level) -> None:
        """Render the collision-free minimum path with waypoints strictly in debug mode."""
        pts = level.optimal_waypoints if getattr(level, "optimal_waypoints", None) else [level.start, level.end]
        if len(pts) < 2:
            return

        # Connecting dashed lines
        for i in range(len(pts) - 1):
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            p2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
            cv2.line(canvas, p1, p2, (0, 220, 160), 1, cv2.LINE_AA)

        # Waypoint nodes
        for idx, pt in enumerate(pts):
            c_pt = (int(round(pt[0])), int(round(pt[1])))
            if idx == 0 or idx == len(pts) - 1:
                cv2.circle(canvas, c_pt, 4, (0, 255, 120), -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, c_pt, 3, (0, 200, 240), -1, cv2.LINE_AA)

        if len(pts) >= 2:
            mid_pt = pts[len(pts) // 2]
            label = f"MIN PATH: {getattr(level, 'min_path_distance', 0.0):.1f} px"
            lx = max(20, min(self.W - 200, int(round(mid_pt[0])) + 10))
            ly = max(30, min(self.H - 30, int(round(mid_pt[1])) - 10))
            draw_text(canvas, label, (lx, ly), font_scale=0.38, color=(0, 240, 180), thickness=1)

    def _draw_player(self, canvas: np.ndarray, engine: "GameEngine") -> None:
        """Render player circular cursor with precision focal dot and subtle collision feedback."""
        pos = engine.player_pos
        r = engine.radius

        is_colliding = getattr(engine.player, "is_colliding", False) or (getattr(engine.player, "collision_flash_timer", 0.0) > 0.0)

        # Base cursor colors
        if is_colliding:
            fill_col = (40, 55, 230)      # Alert coral/red
            border_col = (60, 90, 255)
        else:
            fill_col = config.PLAYER_COLOR  # Calm medical mint (80, 205, 140)
            border_col = (220, 250, 235)    # Crisp hairline edge

        # Subtle collision pulse halo (non-intrusive 1-px pulse ring)
        if is_colliding:
            cv2.circle(canvas, pos, r + 4, (45, 75, 250), 1, cv2.LINE_AA)

        # Core cursor disc
        cv2.circle(canvas, pos, r, fill_col, -1, cv2.LINE_AA)
        # Precision perimeter ring
        cv2.circle(canvas, pos, r, border_col, 1, cv2.LINE_AA)
        # Center precision micro-dot for motor alignment
        cv2.circle(canvas, pos, 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        engine: "GameEngine",
        is_mouse_fallback: bool = False,
        landmark_name: str = "",
    ) -> None:
        """
        Render clean clinical rehabilitation HUD:
        Top-Left:
            STROKE REHABILITATION
            Difficulty: EASY / MEDIUM / HARD
        Top-Right (Telemetry Card):
            Time:       00.0 s
            Distance:   0000 px
            Collisions: 0
            Efficiency: ---
        """
        W, H = self.W, self.H

        # ── 1. Top-Left Header: Title & Difficulty ───────────────────────────
        draw_text(canvas, "STROKE REHABILITATION", (20, 32),
                  font_scale=0.66, color=(245, 248, 252), thickness=2)

        diff_name = (getattr(engine.level, "difficulty_tag", None) or engine.difficulty_cfg.name).upper()
        if "EASY" in diff_name:
            diff_color = (100, 215, 140)  # Medical mint
        elif "HARD" in diff_name:
            diff_color = (90, 100, 245)   # Clinical coral
        else:
            diff_color = (60, 190, 250)   # Medical amber/gold

        draw_text(canvas, "Difficulty: ", (20, 56),
                  font_scale=0.50, color=(140, 165, 190), thickness=1)
        ds = cv2.getTextSize("Difficulty: ", cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
        draw_text(canvas, diff_name, (20 + ds[0], 56),
                  font_scale=0.52, color=diff_color, thickness=2)

        # ── 2. Top-Right Telemetry Card ──────────────────────────────────────
        # Format metrics exactly as specified:
        # Time: 00.0 s
        # Distance: 0000 px
        # Collisions: 0
        # Efficiency: ---
        t = engine.elapsed_time
        time_val = f"{t:04.1f} s"

        dist_val = f"{int(round(engine.actual_distance)):04d} px"
        col_val  = str(engine.wall_hit_count)

        if engine.actual_distance < 10.0 or engine.path_efficiency <= 0.0:
            eff_val = "---"
        else:
            eff_val = f"{engine.path_efficiency:4.1f}%"

        cw, ch = 215, 98
        cx1 = W - cw - 20
        cy1 = 14

        # Sleek, semi-transparent dark clinical card
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + cw, cy1 + ch), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + cw, cy1 + ch), (55, 72, 90), 1, cv2.LINE_AA)

        is_colliding = getattr(engine.player, "is_colliding", False) or (getattr(engine.player, "collision_flash_timer", 0.0) > 0.0)
        col_color = (60, 80, 245) if is_colliding else (240, 245, 250)

        rows = [
            ("Time:",       time_val, (240, 245, 250)),
            ("Distance:",   dist_val, (240, 245, 250)),
            ("Collisions:", col_val,  col_color),
            ("Efficiency:", eff_val,  (240, 245, 250)),
        ]

        row_y = cy1 + 20
        for label, val, val_col in rows:
            draw_text(canvas, label, (cx1 + 14, row_y),
                      font_scale=0.46, color=(140, 165, 190), thickness=1)
            vs = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            draw_text(canvas, val, (cx1 + cw - 14 - vs[0], row_y),
                      font_scale=0.48, color=val_col, thickness=1)
            row_y += 22

        # ── 3. Bottom Footer (Level name & subtle shortcuts) ─────────────────
        lvl_name = getattr(engine.level, "name", "Level")
        seed_val = getattr(engine.level, "seed", None)
        if seed_val is not None:
            left_info = f"Level: {lvl_name}  [Seed: {seed_val}]"
        else:
            left_info = f"Level: {lvl_name}"
        draw_text(canvas, left_info, (20, H - 14),
                  font_scale=0.44, color=(120, 140, 165))

        if not engine.is_finished:
            hint = "[R] Restart   [P] Pause   [M] Menu   [ESC] Exit"
            hs   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            draw_text(canvas, hint, (W - hs[0] - 20, H - 14),
                      font_scale=0.42, color=(95, 115, 135))

    def _draw_debug_overlay(
        self,
        canvas: np.ndarray,
        engine: "GameEngine",
        raw_coords: Optional[Tuple[float, float]],
        algo_name: str,
    ) -> None:
        """Render real-time telemetry overlay showing Raw vs. Smoothed coordinates and optimal path."""
        bx, by, bw, bh = 12, 82, 245, 126
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 220, 160), 1, cv2.LINE_AA)

        px, py = engine.player.px, engine.player.py
        if raw_coords is not None:
            rx, ry = raw_coords
            jitter = math.hypot(rx - px, ry - py)
            raw_str = f"RAW:    X={rx:5.1f}  Y={ry:5.1f}"
            jit_str = f"JITTER: D={jitter:5.2f} px"
        else:
            raw_str = "RAW:    NO SIGNAL"
            jit_str = "JITTER: N/A"

        smooth_str = f"SMOOTH: X={px:5.1f}  Y={py:5.1f}"
        act_dist_str = f"ACTUAL: D={engine.actual_distance:5.1f} px"
        filter_str = f"FILTER: {algo_name or 'ONE_EURO'}"

        pts = getattr(engine.level, "optimal_waypoints", [])
        min_p = getattr(engine.level, "min_path_distance", 0.0)
        path_str = f"PATH:   D={min_p:.1f}px ({len(pts)} pts)"

        draw_text(canvas, "[DEBUG TELEMETRY]", (bx + 8, by + 18),
                  font_scale=0.42, color=(0, 255, 180), thickness=1)
        draw_text(canvas, raw_str, (bx + 8, by + 34),
                  font_scale=0.36, color=(60, 220, 255))
        draw_text(canvas, smooth_str, (bx + 8, by + 50),
                  font_scale=0.36, color=(80, 255, 120))
        draw_text(canvas, act_dist_str, (bx + 8, by + 66),
                  font_scale=0.36, color=(255, 220, 100))
        draw_text(canvas, jit_str, (bx + 8, by + 82),
                  font_scale=0.36, color=(240, 240, 255))
        draw_text(canvas, path_str, (bx + 8, by + 98),
                  font_scale=0.36, color=(0, 240, 200))
        draw_text(canvas, filter_str, (bx + 8, by + 114),
                  font_scale=0.33, color=(160, 160, 220))

        # Canvas visual marker: draw raw crosshair and connecting line to smoothed player
        if raw_coords is not None:
            ipt = (int(round(raw_coords[0])), int(round(raw_coords[1])))
            spt = engine.player.position
            cv2.line(canvas, ipt, spt, (100, 100, 220), 1, cv2.LINE_AA)
            cv2.circle(canvas, ipt, 4, (60, 220, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, ipt, (60, 220, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)


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
