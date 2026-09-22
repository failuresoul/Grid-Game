"""
ui/screens.py — Screen and Overlay Drawing Functions

Contains all full-screen or overlay drawing routines that are shown
during non-gameplay states:
  - draw_start_screen()    (difficulty selection)
  - draw_waiting_overlay() (show-your-hand prompt)
  - draw_paused_overlay()
  - draw_win_overlay()     (metrics table)
  - draw_timeout_overlay()

Also re-exports the shared drawing primitives used by ui.renderer:
  draw_text(), glow_circle(), draw_star()

Dependency chain:
    ui.screens → config
    (GameEngine is passed as a plain argument — not imported at module level)
"""

from __future__ import annotations
import math
from typing import Tuple

import cv2
import numpy as np

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Shared drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def draw_text(
    img:        np.ndarray,
    text:       str,
    origin:     Tuple[int, int],
    font_scale: float = 0.6,
    color:      Tuple[int, int, int] = (220, 240, 255),
    thickness:  int = 1,
    shadow:     bool = True,
) -> None:
    """Draw text with an optional 1-px dark shadow for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (origin[0] + 1, origin[1] + 1),
                    font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def glow_circle(
    img:         np.ndarray,
    centre:      Tuple[int, int],
    radius:      int,
    color:       Tuple[int, int, int],
    glow_layers: int = 2,
) -> None:
    """Draw a clean circular beacon with subtle precision boundary."""
    cx, cy = int(centre[0]), int(centre[1])
    r = max(2, int(radius))
    # Subtle soft outer edge
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r + 2, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    # Core circle
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (240, 250, 255), 1, cv2.LINE_AA)


def draw_start_beacon(
    img:    np.ndarray,
    centre: Tuple[int, int],
    radius: int,
    color:  Tuple[int, int, int] = (70, 195, 110),
) -> None:
    """
    Draw a clean clinical start point beacon.
    - Soft translucent inner fill
    - Crisp boundary ring
    - Concentric anchor ring & center micro-dot
    """
    cx, cy = int(centre[0]), int(centre[1])
    r = max(4, int(radius))

    # Translucent soft fill
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Precision outer ring
    cv2.circle(img, (cx, cy), r, (180, 245, 205), 1, cv2.LINE_AA)
    # Inner calibration ring
    cv2.circle(img, (cx, cy), max(2, int(r * 0.45)), color, 1, cv2.LINE_AA)
    # Center anchor dot
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_target_rings(
    img:     np.ndarray,
    centre:  Tuple[int, int],
    outer_r: int,
    color:   Tuple[int, int, int] = (60, 185, 240),
    pulse_r: int = 0,
) -> None:
    """
    Draw a professional clinical target beacon (precision concentric bullseye).
    - Precision outer target ring
    - Mid-calibration reference ring
    - Solid bullseye core disc with bright center dot
    - 4 cardinal crosshair tick notches
    """
    cx, cy = int(centre[0]), int(centre[1])
    r_out = max(6, int(outer_r + pulse_r))
    r_mid = max(4, int(r_out * 0.60))
    r_core = max(2, int(r_out * 0.28))

    # Soft translucent outer disc
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r_out, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Outer precision boundary ring
    cv2.circle(img, (cx, cy), r_out, color, 2, cv2.LINE_AA)
    # Mid reference ring
    cv2.circle(img, (cx, cy), r_mid, (220, 240, 255), 1, cv2.LINE_AA)
    # Core bullseye
    cv2.circle(img, (cx, cy), r_core, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 4 Cardinal crosshair tick marks (precision targeting aesthetic)
    t_len = 5
    cv2.line(img, (cx, cy - r_out - t_len), (cx, cy - r_out + 2), (220, 240, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + r_out - 2), (cx, cy + r_out + t_len), (220, 240, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx - r_out - t_len, cy), (cx - r_out + 2, cy), (220, 240, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx + r_out - 2, cy), (cx + r_out + t_len, cy), (220, 240, 255), 1, cv2.LINE_AA)


def draw_star(
    img:     np.ndarray,
    centre:  Tuple[int, int],
    outer_r: int,
    color:   Tuple[int, int, int],
) -> None:
    """Compatibility wrapper: renders clinical target rings."""
    draw_target_rings(img, centre, outer_r, color, pulse_r=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Full-canvas screens
# ─────────────────────────────────────────────────────────────────────────────

def draw_start_screen(
    canvas:             np.ndarray,
    selected_difficulty: int,
    W:                  int,
    H:                  int,
) -> None:
    """Difficulty-selection start screen (drawn onto `canvas` in-place)."""

    # Title
    title = "Hand Rehabilitation Maze"
    ts    = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
    draw_text(canvas, title, ((W - ts[0]) // 2, 100),
              font_scale=1.4, color=(120, 220, 255), thickness=2)

    sub = "Move your hand to control the cursor"
    ss  = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
    draw_text(canvas, sub, ((W - ss[0]) // 2, 145),
              font_scale=0.65, color=(160, 200, 230))

    try:
        from emg.emg_interface import get_emg_status_string
        emg_status = get_emg_status_string()
    except Exception:
        emg_status = "EMG: Not Connected / Disabled"
    sensor_info = f"Tracking: Camera (OpenCV)  |  {emg_status}"
    sis = cv2.getTextSize(sensor_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
    draw_text(canvas, sensor_info, ((W - sis[0]) // 2, 175),
              font_scale=0.40, color=(125, 150, 175))

    # Difficulty buttons (1: Easy, 2: Medium, 3: Hard)
    labels = {
        1: ("1  EASY",   (50, 200, 80)),
        2: ("2  MEDIUM", (30, 180, 255)),
        3: ("3  HARD",   (70, 80, 235)),
    }

    bw, bh = 320, 50
    bx = W // 2 - bw // 2
    for diff, (label, col) in labels.items():
        by = 195 + (diff - 1) * 58
        is_sel = (diff == selected_difficulty)
        if is_sel:
            cv2.rectangle(canvas, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), col, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1)
            text_col = (15, 20, 25)
            # Checkmark indicator
            cv2.putText(canvas, "[ACTIVE]", (bx + bw - 85, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (30, 38, 52), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, 1, cv2.LINE_AA)
            text_col = col

        ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        draw_text(canvas, label, (bx + 25, by + (bh + ls[1]) // 2),
                  font_scale=0.72, color=text_col, thickness=2, shadow=False)

    # Prominent START GAME button
    start_by = 385
    start_bh = 58
    cv2.rectangle(canvas, (bx - 2, start_by - 2), (bx + bw + 2, start_by + start_bh + 2), (0, 255, 140), 2, cv2.LINE_AA)
    overlay_btn = canvas.copy()
    cv2.rectangle(overlay_btn, (bx, start_by), (bx + bw, start_by + start_bh), (0, 180, 100), -1)
    cv2.addWeighted(overlay_btn, 0.85, canvas, 0.15, 0, canvas)
    start_lbl = "START GAME  [ENTER]"
    sls = cv2.getTextSize(start_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    draw_text(canvas, start_lbl, (bx + (bw - sls[0]) // 2, start_by + (start_bh + sls[1]) // 2),
              font_scale=0.75, color=(255, 255, 255), thickness=2, shadow=True)

    # PROGRESS & HISTORY button
    hist_by = 455
    hist_bh = 46
    cv2.rectangle(canvas, (bx, hist_by), (bx + bw, hist_by + hist_bh), (35, 45, 65), -1)
    cv2.rectangle(canvas, (bx, hist_by), (bx + bw, hist_by + hist_bh), (60, 160, 230), 1, cv2.LINE_AA)
    hist_lbl = "PROGRESS & HISTORY  [H]"
    hls = cv2.getTextSize(hist_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
    draw_text(canvas, hist_lbl, (bx + (bw - hls[0]) // 2, hist_by + (hist_bh + hls[1]) // 2),
              font_scale=0.58, color=(200, 230, 255), thickness=1, shadow=False)

    # MOVE CAMERA button
    cam_by = 512
    cam_bh = 42
    cv2.rectangle(canvas, (bx, cam_by), (bx + bw, cam_by + cam_bh), (28, 36, 48), -1)
    cv2.rectangle(canvas, (bx, cam_by), (bx + bw, cam_by + cam_bh), (100, 130, 170), 1, cv2.LINE_AA)
    cam_lbl = "MOVE CAMERA CORNER  [V]"
    cls = cv2.getTextSize(cam_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    draw_text(canvas, cam_lbl, (bx + (bw - cls[0]) // 2, cam_by + (cam_bh + cls[1]) // 2),
              font_scale=0.50, color=(160, 195, 225), thickness=1, shadow=False)

    # Instruction footnote
    foot = "Hover hand cursor over any button for 1.5s to select  |  Full Hand Gesture Control"
    fs = cv2.getTextSize(foot, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
    draw_text(canvas, foot, ((W - fs[0]) // 2, 590),
              font_scale=0.46, color=(120, 165, 195))


def get_menu_button_rects(
    W: int,
    H: int,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """
    Return gesture-activatable button rects for the MENU screen.
    Format: (button_id, (x, y, w, h), label, shortcut_key)
    """
    bw = 320
    bx = W // 2 - bw // 2
    return [
        ("DIFF_1",      (bx, 195, bw, 50), "1  EASY",                 "1"),
        ("DIFF_2",      (bx, 253, bw, 50), "2  MEDIUM",               "2"),
        ("DIFF_3",      (bx, 311, bw, 50), "3  HARD",                 "3"),
        ("START",       (bx, 385, bw, 58), "START GAME",              "ENTER"),
        ("HISTORY",     (bx, 455, bw, 46), "PROGRESS & HISTORY",      "H"),
        ("MOVE_CAMERA", (bx, 512, bw, 42), "MOVE CAMERA CORNER",       "V"),
    ]


def get_level_select_button_rects(
    W: int,
    H: int,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """
    Return gesture-activatable button rects for the LEVEL_SELECT screen.
    """
    return [
        ("PREV_LEVEL",  (W // 2 - 270, 420, 120, 52), "< PREV",            "P"),
        ("CONFIRM",     (W // 2 - 135, 420, 270, 52), "CONFIRM & LOAD",     "ENTER"),
        ("NEXT_LEVEL",  (W // 2 + 150, 420, 120, 52), "NEXT >",            "N"),
        ("BACK",        (W // 2 - 135, 485, 270, 44), "BACK TO MENU",       "ESC"),
        ("MOVE_CAMERA", (W // 2 - 135, 540, 270, 38), "MOVE CAMERA CORNER", "V"),
    ]


def draw_level_select_screen(
    canvas:            np.ndarray,
    difficulty_name:   str,
    level_index:       int,
    total_levels:      int,
    level_name:        str,
    min_path_distance: float,
    wall_count:        int,
    anim_t:            float = 0.0,
) -> None:
    """
    Level Selection & Preview screen (LEVEL_SELECT state).
    Displays level card, difficulty info, minimal path length, and obstacle complexity.
    """
    W, H = canvas.shape[1], canvas.shape[0]
    canvas[:] = getattr(config, "CANVAS_BG_COLOR", getattr(config, "BACKGROUND_COLOR", (19, 29, 51)))

    # Title
    title = "SELECT MAZE LEVEL"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 2)[0]
    draw_text(canvas, title, ((W - ts[0]) // 2, 65),
              font_scale=1.25, color=(120, 220, 255), thickness=2)

    diff_label = f"DIFFICULTY: {difficulty_name.upper()}  (Hover buttons or use hand gestures)"
    ds = cv2.getTextSize(diff_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    draw_text(canvas, diff_label, ((W - ds[0]) // 2, 102),
              font_scale=0.52, color=(160, 200, 230))

    # Level Card Box
    bx, by, bw, bh = W // 2 - 270, 125, 540, 275
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (22, 32, 48), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 160, 220), 2)

    # Card content
    header = f"Level {level_index + 1} of {total_levels}"
    draw_text(canvas, header, (bx + 35, by + 40), font_scale=0.62, color=(255, 200, 80), thickness=2)

    lvl_title = f"{level_name}"
    draw_text(canvas, lvl_title, (bx + 35, by + 82), font_scale=0.82, color=(230, 240, 255), thickness=2)

    # Details
    draw_text(canvas, f"Optimal Minimum Path:   {min_path_distance:.1f} px",
              (bx + 35, by + 130), font_scale=0.56, color=(180, 210, 235))
    draw_text(canvas, f"Geometric Obstacles:    {wall_count} barriers",
              (bx + 35, by + 168), font_scale=0.56, color=(180, 210, 235))
    draw_text(canvas, f"Movement Space:         Continuous 2D free navigation",
              (bx + 35, by + 206), font_scale=0.52, color=(140, 185, 215))
    draw_text(canvas, f"Status:                 READY to load",
              (bx + 35, by + 244), font_scale=0.52, color=(100, 230, 160))

    # Action Buttons:
    # 1. PREV
    p_bx, p_by, p_bw, p_bh = W // 2 - 270, 420, 120, 52
    cv2.rectangle(canvas, (p_bx, p_by), (p_bx + p_bw, p_by + p_bh), (28, 38, 54), -1)
    cv2.rectangle(canvas, (p_bx, p_by), (p_bx + p_bw, p_by + p_bh), (65, 120, 180), 1, cv2.LINE_AA)
    p_txt = "< PREV [P]"
    pts = cv2.getTextSize(p_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    draw_text(canvas, p_txt, (p_bx + (p_bw - pts[0]) // 2, p_by + (p_bh + pts[1]) // 2),
              font_scale=0.48, color=(200, 230, 255))

    # 2. CONFIRM & LOAD (Prominent Green)
    c_bx, c_by, c_bw, c_bh = W // 2 - 135, 420, 270, 52
    cv2.rectangle(canvas, (c_bx - 2, c_by - 2), (c_bx + c_bw + 2, c_by + c_bh + 2), (0, 255, 140), 2, cv2.LINE_AA)
    overlay_c = canvas.copy()
    cv2.rectangle(overlay_c, (c_bx, c_by), (c_bx + c_bw, c_by + c_bh), (0, 175, 95), -1)
    cv2.addWeighted(overlay_c, 0.85, canvas, 0.15, 0, canvas)
    c_txt = "LOAD & PLAY  [ENTER]"
    cts = cv2.getTextSize(c_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    draw_text(canvas, c_txt, (c_bx + (c_bw - cts[0]) // 2, c_by + (c_bh + cts[1]) // 2),
              font_scale=0.65, color=(255, 255, 255), thickness=2)

    # 3. NEXT
    n_bx, n_by, n_bw, n_bh = W // 2 + 150, 420, 120, 52
    cv2.rectangle(canvas, (n_bx, n_by), (n_bx + n_bw, n_by + n_bh), (28, 38, 54), -1)
    cv2.rectangle(canvas, (n_bx, n_by), (n_bx + n_bw, n_by + n_bh), (65, 120, 180), 1, cv2.LINE_AA)
    n_txt = "NEXT > [N]"
    nts = cv2.getTextSize(n_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    draw_text(canvas, n_txt, (n_bx + (n_bw - nts[0]) // 2, n_by + (n_bh + nts[1]) // 2),
              font_scale=0.48, color=(200, 230, 255))

    # 4. BACK TO MENU
    b_bx, b_by, b_bw, b_bh = W // 2 - 135, 485, 270, 44
    cv2.rectangle(canvas, (b_bx, b_by), (b_bx + b_bw, b_by + b_bh), (25, 34, 48), -1)
    cv2.rectangle(canvas, (b_bx, b_by), (b_bx + b_bw, b_by + b_bh), (60, 95, 140), 1, cv2.LINE_AA)
    b_txt = "< BACK TO MENU  [ESC]"
    bts = cv2.getTextSize(b_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    draw_text(canvas, b_txt, (b_bx + (b_bw - bts[0]) // 2, b_by + (b_bh + bts[1]) // 2),
              font_scale=0.50, color=(170, 205, 235))

    # 5. MOVE CAMERA CORNER
    m_bx, m_by, m_bw, m_bh = W // 2 - 135, 540, 270, 38
    cv2.rectangle(canvas, (m_bx, m_by), (m_bx + m_bw, m_by + m_bh), (22, 30, 42), -1)
    cv2.rectangle(canvas, (m_bx, m_by), (m_bx + m_bw, m_by + m_bh), (80, 110, 150), 1, cv2.LINE_AA)
    m_txt = "MOVE CAMERA CORNER  [V]"
    mts = cv2.getTextSize(m_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    draw_text(canvas, m_txt, (m_bx + (m_bw - mts[0]) // 2, m_by + (m_bh + mts[1]) // 2),
              font_scale=0.45, color=(150, 185, 215))

    # Navigation instructions
    foot = "Hold hand cursor over any button for 1.5s to select  |  Full Hand Gesture Control"
    fs = cv2.getTextSize(foot, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
    draw_text(canvas, foot, ((W - fs[0]) // 2, 605),
              font_scale=0.44, color=(120, 160, 190))


# ─────────────────────────────────────────────────────────────────────────────
#  In-game overlays
# ─────────────────────────────────────────────────────────────────────────────

def draw_waiting_overlay(canvas: np.ndarray, W: int, H: int, anim_t: float) -> None:
    """Shown before player begins moving (READY state). Clean, non-intrusive clinical prompt."""
    bw, bh = 500, 115
    bx, by = (W - bw) // 2, (H - bh) // 2 - 20

    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (65, 88, 115), 1, cv2.LINE_AA)

    title = "REHABILITATION SESSION READY"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    draw_text(canvas, title, (bx + (bw - ts[0]) // 2, by + 32),
              font_scale=0.72, color=(240, 248, 255), thickness=2)

    sub = "Place cursor inside START and move to begin."
    ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    draw_text(canvas, sub, (bx + (bw - ss[0]) // 2, by + 65),
              font_scale=0.48, color=(160, 205, 235))

    note = "Timer and kinematic tracking activate on movement."
    ns = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    draw_text(canvas, note, (bx + (bw - ns[0]) // 2, by + 94),
              font_scale=0.42, color=(110, 150, 185))


# Alias for backward compatibility
draw_ready_overlay = draw_waiting_overlay


def get_paused_button_rects(
    W: int,
    H: int,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """Return gesture buttons for the PAUSED overlay."""
    return [
        ("RESUME",    (W // 2 - 150, H // 2 + 10, 140, 48), "RESUME",        "P"),
        ("MAIN_MENU", (W // 2 + 10,  H // 2 + 10, 140, 48), "QUIT TO MENU",  "M"),
    ]


def draw_paused_overlay(canvas: np.ndarray, W: int, H: int) -> None:
    """Clinical paused state overlay with interactive gesture buttons."""
    bw, bh = 360, 150
    bx, by = (W - bw) // 2, (H - bh) // 2
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (70, 110, 155), 1, cv2.LINE_AA)

    pts = cv2.getTextSize("SESSION PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
    draw_text(canvas, "SESSION PAUSED", (bx + (bw - pts[0]) // 2, by + 40),
              font_scale=0.85, color=(240, 245, 250), thickness=2)

    # 1. RESUME button
    r_bx, r_by, r_bw, r_bh = W // 2 - 150, H // 2 + 10, 140, 48
    cv2.rectangle(canvas, (r_bx, r_by), (r_bx + r_bw, r_by + r_bh), (0, 160, 90), -1)
    cv2.rectangle(canvas, (r_bx, r_by), (r_bx + r_bw, r_by + r_bh), (0, 240, 130), 1, cv2.LINE_AA)
    r_txt = "RESUME [P]"
    rts = cv2.getTextSize(r_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    draw_text(canvas, r_txt, (r_bx + (r_bw - rts[0]) // 2, r_by + (r_bh + rts[1]) // 2),
              font_scale=0.50, color=(255, 255, 255))

    # 2. QUIT TO MENU button
    q_bx, q_by, q_bw, q_bh = W // 2 + 10, H // 2 + 10, 140, 48
    cv2.rectangle(canvas, (q_bx, q_by), (q_bx + q_bw, q_by + q_bh), (40, 45, 60), -1)
    cv2.rectangle(canvas, (q_bx, q_by), (q_bx + q_bw, q_by + q_bh), (120, 140, 175), 1, cv2.LINE_AA)
    q_txt = "MENU [M]"
    qts = cv2.getTextSize(q_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    draw_text(canvas, q_txt, (q_bx + (q_bw - qts[0]) // 2, q_by + (q_bh + qts[1]) // 2),
              font_scale=0.50, color=(210, 230, 250))


def get_results_button_rects(W: int = 800, H: int = 600) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """
    Returns list of (button_id, (bx, by, bw, bh), label, key_shortcut) for the results screen:
        1. PLAY AGAIN   [R]
        2. NEXT LEVEL   [N]
        3. LEVEL SELECT [L]
        4. MAIN MENU    [M]
        5. EXIT         [ESC]
    """
    btn_defs = [
        ("PLAY_AGAIN",   "PLAY AGAIN",   "R"),
        ("NEXT_LEVEL",   "NEXT LEVEL",   "N"),
        ("LEVEL_SELECT", "LEVEL SELECT", "L"),
        ("MAIN_MENU",    "MAIN MENU",    "M"),
        ("EXIT",         "EXIT",         "ESC"),
    ]
    margin = 20
    available_w = W - 2 * margin
    n = len(btn_defs)
    gap = 12
    bw = (available_w - (n - 1) * gap) // n
    bh = 42
    by = H - 54

    result = []
    for i, (b_id, label, key) in enumerate(btn_defs):
        bx = margin + i * (bw + gap)
        result.append((b_id, (bx, by, bw, bh), label, key))
    return result


def draw_results_screen(
    canvas:          np.ndarray,
    W: int, H:       int,
    anim_t:          float = 0.0,
    final_metrics:   dict = None,
    wall_hits:       int = 0,
    elapsed:         float = 0.0,
    level:           Optional[Any] = None,
    trajectory:      Optional[List[Tuple[float, float]]] = None,
    difficulty_name: str = "EASY",
    mouse_pos:       Optional[Tuple[float, float]] = None,
    adaptive_recommendation: Optional[Any] = None,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """
    Professional results screen after completing a maze:
    - Header: REHABILITATION RESULTS (Difficulty, Completion Status)
    - Left Card: Game Performance Metrics (Time, Actual Dist, Min Dist, Efficiency, Accuracy, Smoothness, Collisions, Deviations)
    - Right Card: Trajectory Visualization (Optimal/Minimum path vs. Patient actual trajectory)
    - Bottom Buttons: PLAY AGAIN, NEXT LEVEL, LEVEL SELECT, MAIN MENU, EXIT
    - Clear non-diagnostic framing: 'Game Performance Metrics' (not a medical diagnosis)
    """
    m = final_metrics or {}
    raw_eff = m.get("path_efficiency", 0.0)
    eff_pct = raw_eff * 100.0 if 0.0 < raw_eff <= 1.0 else raw_eff

    min_dist = m.get("minimum_distance", m.get("min_path_distance_px", 0.0))
    act_dist = m.get("actual_distance", m.get("actual_distance_px", 0.0))
    traj_acc = m.get("trajectory_accuracy", 100.0 if not m else 0.0)
    mean_dev = m.get("mean_path_deviation_px", 0.0)
    dev_events = m.get("deviation_events", 0)
    smoothness = m.get("smoothness_score", 100.0)
    t_outside = m.get("time_outside_route_s", 0.0)
    compl_time = m.get("completion_time_s", elapsed)

    # ── Adaptive difficulty recommendation resolution ────────────────────────
    rec = adaptive_recommendation
    if rec is None and getattr(config, "ADAPTIVE_DIFFICULTY", True):
        raw_rec = m.get("adaptive_recommendation")
        if isinstance(raw_rec, dict):
            from metrics.adaptive import AdaptiveRecommendation
            act = raw_rec.get("action", "MAINTAIN")
            col = (90, 225, 140) if act == "INCREASE" else ((80, 95, 245) if act == "DECREASE" else (215, 190, 90))
            rec = AdaptiveRecommendation(
                action=act,
                current_difficulty=raw_rec.get("current_difficulty", difficulty_name),
                target_difficulty=raw_rec.get("target_difficulty", difficulty_name),
                badge_title=raw_rec.get("badge_title", "PRACTICE"),
                headline=raw_rec.get("headline", ""),
                rationale=raw_rec.get("rationale", ""),
                color=col,
                criteria_met=raw_rec.get("criteria_met", {}),
            )
        else:
            try:
                from metrics.adaptive import evaluate_adaptive_difficulty
                rec = evaluate_adaptive_difficulty(
                    current_metrics=m,
                    current_difficulty=difficulty_name,
                    wall_hits=wall_hits,
                    completion_status="COMPLETED",
                )
            except Exception:
                rec = None

    # ── 1. Semi-translucent dark slate backdrop ──────────────────────────────
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (14, 17, 23), -1)
    cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)

    # ── 2. Header Bar: Title, Subtitle & Badges ──────────────────────────────
    draw_text(canvas, "REHABILITATION RESULTS", (24, 38),
              font_scale=0.82, color=(245, 250, 255), thickness=2)
    draw_text(canvas, "Game Performance Metrics", (24, 62),
              font_scale=0.48, color=(120, 210, 160), thickness=1)

    # Status & Difficulty Badges (Top Right)
    diff_tag = difficulty_name.upper()
    if "EASY" in diff_tag:
        diff_col = (100, 215, 140)
    elif "HARD" in diff_tag:
        diff_col = (90, 100, 245)
    else:
        diff_col = (60, 190, 250)

    # Status badge
    status_label = "Status: COMPLETED"
    draw_text(canvas, status_label, (W - 300, 24),
              font_scale=0.46, color=(90, 220, 140), thickness=1)

    # Difficulty badge
    draw_text(canvas, f"Difficulty: {diff_tag}", (W - 300, 44),
              font_scale=0.46, color=diff_col, thickness=1)

    # EMG sensor status badge (guaranteed to show 'EMG: Not Connected / Disabled' when disabled)
    try:
        from emg.emg_interface import get_emg_status_string
        emg_status_label = get_emg_status_string()
    except Exception:
        emg_status_label = "EMG: Not Connected / Disabled"
    draw_text(canvas, emg_status_label, (W - 300, 64),
              font_scale=0.40, color=(120, 145, 170), thickness=1)

    # ── 3. Layout Grid ───────────────────────────────────────────────────────
    card_top = 78
    card_bot = H - 68
    card_h = card_bot - card_top
    card_gap = 18
    card_w = (W - 2 * 20 - card_gap) // 2

    c1_x = 20
    c2_x = c1_x + card_w + card_gap

    # ─────────────────────────────────────────────────────────────────────────
    #  LEFT CARD: Game Performance Metrics
    # ─────────────────────────────────────────────────────────────────────────
    # Card container
    cv2.rectangle(canvas, (c1_x, card_top), (c1_x + card_w, card_top + card_h),
                  (18, 22, 30), -1)
    cv2.rectangle(canvas, (c1_x, card_top), (c1_x + card_w, card_top + card_h),
                  (55, 75, 95), 1, cv2.LINE_AA)

    # Card Title
    draw_text(canvas, "GAME PERFORMANCE METRICS", (c1_x + 16, card_top + 26),
              font_scale=0.52, color=(140, 195, 240), thickness=2)

    # Subtitle note
    draw_text(canvas, "Session Kinematics & Corridor Adherence", (c1_x + 16, card_top + 46),
              font_scale=0.38, color=(120, 145, 170))

    # Divider line
    cv2.line(canvas, (c1_x + 16, card_top + 56), (c1_x + card_w - 16, card_top + 56),
             (45, 60, 75), 1, cv2.LINE_AA)

    # Formatted Metric Rows
    metric_rows = [
        ("Completion Time",        f"{compl_time:.2f} s"),
        ("Actual Distance",        f"{act_dist:.1f} px"),
        ("Minimum Distance",       f"{min_dist:.1f} px"),
        ("Path Efficiency",        f"{eff_pct:.1f}%"),
        ("Accuracy",               f"{traj_acc:.1f}%"),
        ("Movement Smoothness",    f"{smoothness:.1f}/100"),
        ("Wall Collisions",        str(wall_hits)),
        ("Wrong/Deviation Events", str(dev_events)),
        # Secondary Kinematic Reference
        ("Mean Path Deviation",    f"{mean_dev:.1f} px"),
        ("Time Outside Route",     f"{t_outside:.2f} s"),
    ]

    row_start_y = card_top + 80
    row_h = 28
    for i, (label, val) in enumerate(metric_rows):
        ry = row_start_y + i * row_h
        # Alternating subtle row band
        if i % 2 == 1:
            overlay_row = canvas.copy()
            cv2.rectangle(overlay_row, (c1_x + 12, ry - 18), (c1_x + card_w - 12, ry + 6),
                          (24, 30, 40), -1)
            cv2.addWeighted(overlay_row, 0.45, canvas, 0.55, 0, canvas)

        draw_text(canvas, label, (c1_x + 18, ry),
                  font_scale=0.45, color=(145, 170, 195))
        vs = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        draw_text(canvas, val, (c1_x + card_w - 18 - vs[0], ry),
                  font_scale=0.48, color=(240, 248, 255), thickness=1)

    # ── Adaptive Difficulty Recommendation Card ──────────────────────────────
    if rec is not None and getattr(config, "ADAPTIVE_DIFFICULTY", True):
        rec_box_y = card_top + card_h - 116
        rec_box_h = 88
        rec_box_w = card_w - 24
        rec_box_x = c1_x + 12
        rec_col = getattr(rec, "color", (100, 220, 140))

        cv2.rectangle(canvas, (rec_box_x, rec_box_y), (rec_box_x + rec_box_w, rec_box_y + rec_box_h),
                      (16, 22, 30), -1)
        cv2.rectangle(canvas, (rec_box_x, rec_box_y), (rec_box_x + rec_box_w, rec_box_y + rec_box_h),
                      rec_col, 1, cv2.LINE_AA)

        rec_badge = f"ADAPTIVE RECOMMENDATION: {getattr(rec, 'badge_title', 'RECOMMENDATION')}"
        draw_text(canvas, rec_badge, (rec_box_x + 10, rec_box_y + 18),
                  font_scale=0.38, color=rec_col, thickness=1)

        headline = getattr(rec, "headline", "")
        draw_text(canvas, headline, (rec_box_x + 10, rec_box_y + 40),
                  font_scale=0.36, color=(240, 248, 255), thickness=1)

        rationale = getattr(rec, "rationale", "")
        draw_text(canvas, rationale, (rec_box_x + 10, rec_box_y + 64),
                  font_scale=0.32, color=(145, 170, 195))

    # Non-diagnostic disclaimer footnote at bottom of left card
    footnote_y = card_top + card_h - 12
    draw_text(canvas, "* Evaluates game movement performance only. * Not a medical diagnosis.",
              (c1_x + 16, footnote_y), font_scale=0.32, color=(105, 125, 145))

    # ─────────────────────────────────────────────────────────────────────────
    #  RIGHT CARD: Trajectory Visualization (Optimal vs. Patient Path)
    # ─────────────────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (c2_x, card_top), (c2_x + card_w, card_top + card_h),
                  (18, 22, 30), -1)
    cv2.rectangle(canvas, (c2_x, card_top), (c2_x + card_w, card_top + card_h),
                  (55, 75, 95), 1, cv2.LINE_AA)

    # Card Title
    draw_text(canvas, "TRAJECTORY VISUALIZATION", (c2_x + 16, card_top + 26),
              font_scale=0.52, color=(140, 195, 240), thickness=2)

    # Subtitle / Route comparison legend
    draw_text(canvas, "Route Comparison: Optimal vs. Actual", (c2_x + 16, card_top + 46),
              font_scale=0.38, color=(120, 145, 170))

    cv2.line(canvas, (c2_x + 16, card_top + 56), (c2_x + card_w - 16, card_top + 56),
             (45, 60, 75), 1, cv2.LINE_AA)

    # Legend indicator chips
    # 1. Optimal path legend
    cv2.line(canvas, (c2_x + 20, card_top + 72), (c2_x + 40, card_top + 72), (0, 230, 150), 2, cv2.LINE_AA)
    cv2.circle(canvas, (c2_x + 30, card_top + 72), 3, (0, 230, 150), -1, cv2.LINE_AA)
    draw_text(canvas, "Optimal Route", (c2_x + 46, card_top + 75),
              font_scale=0.40, color=(160, 230, 190))

    # 2. Patient path legend
    cv2.line(canvas, (c2_x + 185, card_top + 72), (c2_x + 210, card_top + 72), (220, 185, 85), 2, cv2.LINE_AA)
    draw_text(canvas, "Patient Trajectory", (c2_x + 216, card_top + 75),
              font_scale=0.40, color=(220, 205, 160))

    # Viewport container box
    vx = c2_x + 14
    vy = card_top + 88
    vw = card_w - 28
    vh = card_h - 106

    cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (11, 14, 18), -1)
    cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (45, 58, 72), 1, cv2.LINE_AA)

    # Scale level geometry and trajectories into viewport
    if level is not None:
        orig_w = float(getattr(level, "width", 800))
        orig_h = float(getattr(level, "height", 600))
        pad = 12
        scale = min((vw - 2 * pad) / orig_w, (vh - 2 * pad) / orig_h)
        ox = vx + (vw - orig_w * scale) / 2.0
        oy = vy + (vh - orig_h * scale) / 2.0

        def tx(px: float, py: float) -> Tuple[int, int]:
            return int(round(ox + float(px) * scale)), int(round(oy + float(py) * scale))

        # 1. Draw obstacles
        walls = getattr(level, "walls", [])
        for obs in walls:
            if hasattr(obs, "is_polygon") and obs.is_polygon:
                pts_t = np.array([tx(p[0], p[1]) for p in obs.points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [pts_t], color=(38, 44, 54), lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [pts_t], True, (70, 90, 115), 1, cv2.LINE_AA)
            else:
                if hasattr(obs, "as_rect"):
                    rx, ry, rw, rh = obs.as_rect()
                elif isinstance(obs, (tuple, list)) and len(obs) == 4:
                    rx, ry, rw, rh = obs
                else:
                    rx, ry, rw, rh = obs.x, obs.y, obs.w, obs.h
                p1 = tx(rx, ry)
                p2 = tx(rx + rw, ry + rh)
                cv2.rectangle(canvas, p1, p2, (38, 44, 54), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, p1, p2, (70, 90, 115), 1, cv2.LINE_AA)

        # 2. Draw Optimal / Minimum Path
        wps = getattr(level, "optimal_waypoints", None) or [level.start, level.end]
        if len(wps) >= 2:
            for i in range(len(wps) - 1):
                p1 = tx(wps[i][0], wps[i][1])
                p2 = tx(wps[i + 1][0], wps[i + 1][1])
                cv2.line(canvas, p1, p2, (0, 230, 150), 2, cv2.LINE_AA)
            for pt in wps:
                c_pt = tx(pt[0], pt[1])
                cv2.circle(canvas, c_pt, 3, (0, 230, 150), -1, cv2.LINE_AA)

        # 3. Draw Patient Actual Trajectory
        traj = trajectory or []
        if len(traj) >= 2:
            for i in range(len(traj) - 1):
                p1 = tx(traj[i][0], traj[i][1])
                p2 = tx(traj[i + 1][0], traj[i + 1][1])
                cv2.line(canvas, p1, p2, (220, 185, 85), 2, cv2.LINE_AA)

        # 4. Start Point Beacon
        sp = tx(level.start[0], level.start[1])
        cv2.circle(canvas, sp, 7, (70, 195, 110), -1, cv2.LINE_AA)
        cv2.circle(canvas, sp, 7, (180, 245, 205), 1, cv2.LINE_AA)
        cv2.circle(canvas, sp, 2, (255, 255, 255), -1, cv2.LINE_AA)
        draw_text(canvas, "START", (sp[0] - 14, sp[1] + 16),
                  font_scale=0.34, color=(160, 220, 180))

        # 5. End Target Bullseye
        ep = tx(level.end[0], level.end[1])
        cv2.circle(canvas, ep, 7, (60, 185, 240), 2, cv2.LINE_AA)
        cv2.circle(canvas, ep, 4, (220, 240, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, ep, 2, (60, 185, 240), -1, cv2.LINE_AA)
        draw_text(canvas, "END", (ep[0] - 10, ep[1] + 16),
                  font_scale=0.34, color=(140, 210, 255))
    else:
        # Fallback if level geometry not passed
        draw_text(canvas, "Optimal vs. Patient Trajectory", (vx + 45, vy + vh // 2 - 10),
                  font_scale=0.48, color=(140, 175, 205))
        draw_text(canvas, "Trajectory map loaded during active maze session", (vx + 20, vy + vh // 2 + 15),
                  font_scale=0.38, color=(100, 130, 160))

    # ── 4. Bottom Action Buttons: PLAY AGAIN, NEXT LEVEL, LEVEL SELECT, MAIN MENU, EXIT ──
    button_rects = get_results_button_rects(W, H)
    for b_id, (bx, by, bw, bh), label, key in button_rects:
        # Hover detection
        is_hover = False
        if mouse_pos is not None:
            mx, my = mouse_pos
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                is_hover = True

        bg_col = (35, 48, 64) if is_hover else (22, 28, 38)
        border_col = (100, 225, 170) if is_hover else (65, 88, 115)
        txt_col = (255, 255, 255) if is_hover else (230, 240, 250)

        # Button card
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)

        # Text with keyboard shortcut badge
        full_text = f"{label} [{key}]"
        ts = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        tx_pos = bx + (bw - ts[0]) // 2
        ty_pos = by + (bh + ts[1]) // 2
        draw_text(canvas, full_text, (tx_pos, ty_pos),
                  font_scale=0.44, color=txt_col, thickness=1, shadow=False)

    return button_rects


# Backward-compatible alias so existing callers seamlessly route to draw_results_screen
def draw_win_overlay(
    canvas:        np.ndarray,
    W: int, H:     int,
    anim_t:        float = 0.0,
    final_metrics: dict = None,
    wall_hits:     int = 0,
    elapsed:       float = 0.0,
    level:         Optional[Any] = None,
    trajectory:    Optional[List[Tuple[float, float]]] = None,
    difficulty_name: str = "EASY",
    mouse_pos:     Optional[Tuple[float, float]] = None,
    adaptive_recommendation: Optional[Any] = None,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """Backward-compatible wrapper routing to the professional draw_results_screen."""
    return draw_results_screen(
        canvas=canvas,
        W=W,
        H=H,
        anim_t=anim_t,
        final_metrics=final_metrics,
        wall_hits=wall_hits,
        elapsed=elapsed,
        level=level,
        trajectory=trajectory,
        difficulty_name=difficulty_name,
        mouse_pos=mouse_pos,
        adaptive_recommendation=adaptive_recommendation,
    )


def draw_timeout_overlay(
    canvas:    np.ndarray,
    W: int, H: int,
    elapsed:   float,
    wall_hits: int,
) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (20, 5, 5), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    draw_text(canvas, "TIME'S UP!", (W // 2 - 120, H // 2 - 30),
              font_scale=1.3, color=config.FAIL_COLOR, thickness=3)
    draw_text(canvas,
              f"Time: {elapsed:.1f}s   Wall hits: {wall_hits}",
              (W // 2 - 180, H // 2 + 20),
              font_scale=0.65, color=(180, 140, 200))
    draw_text(canvas, "Press R to restart  |  ESC to quit",
              (W // 2 - 200, H - 60), font_scale=0.55, color=(140, 120, 180))


# =============================================================================
#  7. Progress & History Screen
# =============================================================================

def _draw_trend_sparkline(
    canvas: np.ndarray,
    x: int, y: int, w: int, h: int,
    title: str,
    values: Sequence[float],
    unit: str = "",
    color: Tuple[int, int, int] = (100, 220, 140),
) -> None:
    """Render a compact clinical trend sparkline on OpenCV canvas."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 20, 26), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (42, 54, 68), 1, cv2.LINE_AA)

    # Title & latest value header
    draw_text(canvas, title, (x + 8, y + 16), font_scale=0.36, color=(180, 200, 220), thickness=1)

    if not values:
        draw_text(canvas, "--", (x + w - 30, y + 16), font_scale=0.36, color=(110, 130, 150))
        draw_text(canvas, "No data", (x + w // 2 - 24, y + h // 2 + 6), font_scale=0.36, color=(90, 105, 125))
        return

    latest_val = values[-1]
    latest_str = f"{latest_val:.1f}{unit}" if isinstance(latest_val, float) else f"{latest_val}{unit}"
    draw_text(canvas, latest_str, (x + w - 70, y + 16), font_scale=0.38, color=color, thickness=1)

    # Plot area bounds
    plot_x = x + 8
    plot_y = y + 24
    plot_w = w - 16
    plot_h = h - 30

    # Grid / baseline guideline
    mid_y = plot_y + plot_h // 2
    cv2.line(canvas, (plot_x, mid_y), (plot_x + plot_w, mid_y), (28, 36, 46), 1, cv2.LINE_AA)

    n = len(values)
    if n == 1:
        pt = (plot_x + plot_w // 2, mid_y)
        cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, color, 1, cv2.LINE_AA)
        return

    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin if vmax > vmin else 1.0

    pts: List[Tuple[int, int]] = []
    for idx, v in enumerate(values):
        px = int(round(plot_x + (float(idx) / (n - 1)) * plot_w))
        py = int(round(plot_y + plot_h - ((float(v) - vmin) / vrange) * plot_h))
        pts.append((px, py))

    # Connect lines
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    # Plot vertices
    for pt in pts:
        cv2.circle(canvas, pt, 2, color, -1, cv2.LINE_AA)

    # Focal marker for the latest session
    cv2.circle(canvas, pts[-1], 4, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, pts[-1], 6, (255, 255, 255), 1, cv2.LINE_AA)


def get_history_button_rects(
    W: int, H: int,
    current_filter: str = "ALL",
    page: int = 0,
    total_pages: int = 1,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """Return layout rectangles for all clickable buttons on History screen."""
    buttons: List[Tuple[str, Tuple[int, int, int, int], str, str]] = []

    # 1. Filter tabs (Top right)
    tab_w, tab_h = 74, 26
    filters = [("FILTER_ALL", "ALL [1]", "1"),
               ("FILTER_EASY", "EASY [2]", "2"),
               ("FILTER_MEDIUM", "MED [3]", "3"),
               ("FILTER_HARD", "HARD [4]", "4")]
    start_tab_x = W - 20 - (len(filters) * (tab_w + 6))
    tab_y = 30
    for idx, (action, label, key) in enumerate(filters):
        bx = start_tab_x + idx * (tab_w + 6)
        buttons.append((action, (bx, tab_y, tab_w, tab_h), label, key))

    # 2. Pagination buttons (Inside table card)
    if total_pages > 1:
        pag_y = H - 104
        buttons.append(("PREV_PAGE", (28, pag_y, 74, 22), "< PREV", "P"))
        buttons.append(("NEXT_PAGE", (160, pag_y, 74, 22), "NEXT >", "N"))

    # 3. Bottom action navigation buttons
    btn_y = H - 54
    btn_h = 36
    action_btns = [
        ("MAIN_MENU", "MAIN MENU [M]", "M"),
        ("LEVEL_SELECT", "LEVEL SELECT [L]", "L"),
        ("EXIT", "EXIT [ESC]", "ESC"),
    ]
    spacing = 14
    margin = 24
    total_w = W - 2 * margin
    btn_w = (total_w - (len(action_btns) - 1) * spacing) // len(action_btns)

    for idx, (action, label, key) in enumerate(action_btns):
        bx = margin + idx * (btn_w + spacing)
        buttons.append((action, (bx, btn_y, btn_w, btn_h), label, key))

    return buttons


def draw_history_screen(
    canvas: np.ndarray,
    W: int,
    H: int,
    sessions: Sequence[Dict[str, Any]],
    current_filter: str = "ALL",
    page: int = 0,
    rows_per_page: int = 7,
    mouse_pos: Optional[Tuple[float, float]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str]]:
    """
    Render clinical progress and historical performance screen:
    - Top header with longitudinal framing & difficulty filter tabs
    - Left Panel: Historical sessions table (Session, Difficulty, Time, Distance, Efficiency, Accuracy, Smoothness, Collisions)
    - Right Panel: 5 Trend Graphs (Time, Distance, Efficiency, Accuracy, Smoothness)
    - If empty: "No previous sessions available."
    - Non-diagnostic measurement visualization only.
    """
    # ── 1. Semi-translucent dark slate backdrop ──────────────────────────────
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (14, 17, 23), -1)
    cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)

    # Filter sessions
    norm_filter = current_filter.upper()
    if norm_filter in ("ALL", ""):
        filtered = list(sessions)
    else:
        filtered = [s for s in sessions if norm_filter in s.get("difficulty", "").upper()]

    total_sessions = len(filtered)
    total_pages = max(1, (total_sessions + rows_per_page - 1) // rows_per_page) if total_sessions > 0 else 1
    page = max(0, min(page, total_pages - 1))

    # ── 2. Header Bar ────────────────────────────────────────────────────────
    draw_text(canvas, "REHABILITATION PROGRESS & HISTORY", (24, 38),
              font_scale=0.76, color=(245, 250, 255), thickness=2)
    draw_text(canvas, "Recorded Game Performance Measurements", (24, 60),
              font_scale=0.44, color=(120, 210, 160), thickness=1)

    # ── 3. Filter Tabs (Top Right) ───────────────────────────────────────────
    buttons = get_history_button_rects(W, H, current_filter=norm_filter, page=page, total_pages=total_pages)

    for action, (bx, by, bw, bh), label, _ in buttons:
        if action.startswith("FILTER_"):
            target_filter = action.replace("FILTER_", "")
            is_active = (target_filter == norm_filter) or (target_filter == "ALL" and norm_filter in ("ALL", ""))
            is_hover = False
            if mouse_pos:
                mx, my = mouse_pos
                is_hover = bx <= mx <= bx + bw and by <= my <= by + bh

            if is_active:
                bg_col = (55, 95, 80)
                border_col = (100, 220, 150)
                txt_col = (230, 255, 240)
            elif is_hover:
                bg_col = (38, 48, 62)
                border_col = (120, 160, 200)
                txt_col = (240, 248, 255)
            else:
                bg_col = (22, 28, 36)
                border_col = (50, 65, 82)
                txt_col = (150, 175, 200)

            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)
            draw_text(canvas, label, (bx + 8, by + 18), font_scale=0.38, color=txt_col, thickness=1)

    # ── 4. Main Body: Empty State or Active Panels ───────────────────────────
    if total_sessions == 0:
        # Centered Empty State Card
        card_w, card_h = 540, 200
        cx = (W - card_w) // 2
        cy = (H - card_h) // 2 - 10

        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (55, 75, 95), 1, cv2.LINE_AA)

        # Subtle info beacon circle
        cv2.circle(canvas, (cx + card_w // 2, cy + 50), 22, (35, 48, 65), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + card_w // 2, cy + 50), 22, (80, 115, 155), 1, cv2.LINE_AA)
        draw_text(canvas, "i", (cx + card_w // 2 - 4, cy + 57), font_scale=0.60, color=(160, 205, 250), thickness=2)

        draw_text(canvas, "No previous sessions available.", (cx + 100, cy + 105),
                  font_scale=0.66, color=(240, 245, 250), thickness=2)
        
        filter_hint = f" (for {norm_filter})" if norm_filter != "ALL" else ""
        draw_text(canvas, f"Complete a maze level to record performance history{filter_hint}.",
                  (cx + 65, cy + 138), font_scale=0.44, color=(140, 165, 190))

    else:
        # Layout Geometry
        card_top = 74
        card_bot = H - 68
        card_h = card_bot - card_top
        card_gap = 16

        left_w = int((W - 2 * 20 - card_gap) * 0.53)
        right_w = (W - 2 * 20 - card_gap) - left_w

        c1_x = 20
        c2_x = c1_x + left_w + card_gap

        # ── Left Card: Historical Sessions Table ─────────────────────────────
        cv2.rectangle(canvas, (c1_x, card_top), (c1_x + left_w, card_top + card_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (c1_x, card_top), (c1_x + left_w, card_top + card_h), (55, 75, 95), 1, cv2.LINE_AA)

        # Card Title
        draw_text(canvas, f"HISTORICAL SESSIONS ({total_sessions} total)", (c1_x + 14, card_top + 24),
                  font_scale=0.46, color=(140, 195, 240), thickness=2)

        # Table Column Headers
        th_y = card_top + 46
        cv2.line(canvas, (c1_x + 12, th_y + 8), (c1_x + left_w - 12, th_y + 8), (45, 60, 75), 1)

        cols = [
            ("#",        c1_x + 14),
            ("Diff",     c1_x + 46),
            ("Time",     c1_x + 100),
            ("Dist",     c1_x + 155),
            ("Eff",      c1_x + 215),
            ("Acc",      c1_x + 265),
            ("Smooth",   c1_x + 315),
            ("Hits",     c1_x + 372),
        ]
        for name, col_x in cols:
            draw_text(canvas, name, (col_x, th_y), font_scale=0.38, color=(130, 155, 180), thickness=1)

        # Display rows in reverse chronological order (newest first on page 0)
        table_sessions = list(reversed(filtered))
        start_idx = page * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_sessions)
        page_rows = table_sessions[start_idx:end_idx]

        row_y = th_y + 28
        row_h = 32

        for i, s in enumerate(page_rows):
            # Alternating subtle row striping
            bg_row = (22, 28, 38) if i % 2 == 0 else (18, 22, 30)
            cv2.rectangle(canvas, (c1_x + 10, row_y - 18), (c1_x + left_w - 10, row_y + row_h - 18), bg_row, -1)

            sess_num = f"#{total_sessions - (start_idx + i)}"
            diff_tag = str(s.get("difficulty", "EASY"))[:4].upper()
            if "EAS" in diff_tag:
                diff_c = (100, 215, 140)
            elif "HAR" in diff_tag:
                diff_c = (90, 100, 245)
            else:
                diff_c = (60, 190, 250)

            t_val = f"{float(s.get('completion_time', 0.0)):.1f}s"
            d_val = f"{int(round(float(s.get('actual_distance', 0.0))))}px"
            eff_val = f"{float(s.get('path_efficiency', 0.0)):.1f}%"
            acc_val = f"{float(s.get('accuracy', 100.0)):.0f}%"
            sm_val  = f"{float(s.get('smoothness', 100.0)):.0f}"
            hit_val = str(s.get("collision_count", 0))
            hit_c   = (80, 95, 235) if int(s.get("collision_count", 0)) > 0 else (170, 195, 215)

            draw_text(canvas, sess_num, (c1_x + 14, row_y), font_scale=0.36, color=(140, 160, 185))
            draw_text(canvas, diff_tag, (c1_x + 46, row_y), font_scale=0.36, color=diff_c)
            draw_text(canvas, t_val,    (c1_x + 100, row_y), font_scale=0.36, color=(225, 235, 245))
            draw_text(canvas, d_val,    (c1_x + 155, row_y), font_scale=0.36, color=(210, 225, 240))
            draw_text(canvas, eff_val,  (c1_x + 215, row_y), font_scale=0.36, color=(120, 215, 160))
            draw_text(canvas, acc_val,  (c1_x + 265, row_y), font_scale=0.36, color=(200, 160, 225))
            draw_text(canvas, sm_val,   (c1_x + 315, row_y), font_scale=0.36, color=(235, 205, 120))
            draw_text(canvas, hit_val,  (c1_x + 372, row_y), font_scale=0.36, color=hit_c)

            row_y += row_h

        # Pagination row at bottom of left card
        if total_pages > 1:
            pag_y = card_bot - 18
            draw_text(canvas, f"Page {page + 1} of {total_pages}", (c1_x + left_w // 2 - 32, pag_y),
                      font_scale=0.38, color=(140, 165, 190))

            for action, (bx, by, bw, bh), label, _ in buttons:
                if action in ("PREV_PAGE", "NEXT_PAGE"):
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (25, 34, 45), -1)
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (55, 75, 100), 1, cv2.LINE_AA)
                    draw_text(canvas, label, (bx + 8, by + 16), font_scale=0.36, color=(180, 205, 235))

        # ── Right Card: 5 Trend Graphs ───────────────────────────────────────
        cv2.rectangle(canvas, (c2_x, card_top), (c2_x + right_w, card_top + card_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (c2_x, card_top), (c2_x + right_w, card_top + card_h), (55, 75, 95), 1, cv2.LINE_AA)

        draw_text(canvas, "PERFORMANCE TRENDS OVER SESSIONS", (c2_x + 14, card_top + 24),
                  font_scale=0.46, color=(140, 195, 240), thickness=2)

        # Extract series (chronological: oldest to newest for trend graph)
        from metrics.history_reader import extract_trend_series
        trends = extract_trend_series(filtered)

        graph_x = c2_x + 12
        graph_w = right_w - 24
        # 5 trend graphs stacked vertically
        graph_top = card_top + 38
        available_h = card_h - 48
        graph_h = (available_h - 4 * 6) // 5

        trend_configs = [
            ("Time over sessions",       trends["completion_time"], " s",   (230, 200, 80)),
            ("Distance over sessions",   trends["actual_distance"], " px",  (240, 180, 70)),
            ("Efficiency over sessions", trends["path_efficiency"], "%",    (100, 220, 140)),
            ("Accuracy over sessions",   trends["accuracy"],        "%",    (210, 150, 220)),
            ("Smoothness over sessions", trends["smoothness"],      "/100", (60, 205, 245)),
        ]

        for idx, (title, vals, unit, col) in enumerate(trend_configs):
            gy = graph_top + idx * (graph_h + 6)
            _draw_trend_sparkline(
                canvas=canvas,
                x=graph_x,
                y=gy,
                w=graph_w,
                h=graph_h,
                title=title,
                values=vals,
                unit=unit,
                color=col,
            )

    # ── 5. Bottom Footnote & Navigation Action Buttons ───────────────────────
    footnote = "* Visualizes recorded measurements over time. Does not evaluate medical recovery or automated diagnosis."
    draw_text(canvas, footnote, (24, H - 58), font_scale=0.34, color=(105, 125, 145))

    for action, (bx, by, bw, bh), label, key in buttons:
        if action in ("MAIN_MENU", "LEVEL_SELECT", "EXIT"):
            is_hover = False
            if mouse_pos:
                mx, my = mouse_pos
                is_hover = bx <= mx <= bx + bw and by <= my <= by + bh

            if action == "MAIN_MENU":
                base_col = (30, 48, 68)
                border_col = (85, 140, 200) if is_hover else (60, 95, 135)
                text_col = (245, 250, 255)
            elif action == "LEVEL_SELECT":
                base_col = (25, 52, 45)
                border_col = (70, 185, 130) if is_hover else (45, 120, 85)
                text_col = (225, 250, 235)
            else:  # EXIT
                base_col = (45, 25, 25)
                border_col = (195, 80, 80) if is_hover else (125, 50, 50)
                text_col = (255, 225, 225)

            if is_hover:
                base_col = tuple(min(255, c + 25) for c in base_col)

            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), base_col, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)
            draw_text(canvas, label, (bx + bw // 2 - 56, by + 23),
                      font_scale=0.44, color=text_col, thickness=1)

    return buttons
