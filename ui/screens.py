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

    # Difficulty buttons
    labels = {
        1: ("1  EASY",   config.EASY),
        2: ("2  MEDIUM", config.MEDIUM),
        3: ("3  HARD",   config.HARD),
    }
    btn_colors = {1: (50, 200, 80), 2: (30, 180, 255), 3: (60, 60, 220)}

    for diff, (label, _) in labels.items():
        bx, by, bw, bh = W // 2 - 150, 220 + (diff - 1) * 100, 300, 60
        col = btn_colors[diff]
        if diff == selected_difficulty:
            cv2.rectangle(canvas, (bx - 4, by - 4), (bx + bw + 4, by + bh + 4),
                          col, 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1)
            text_col = (10, 10, 10)
        else:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 60), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, 1, cv2.LINE_AA)
            text_col = col
        ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        draw_text(canvas, label,
                  (bx + (bw - ls[0]) // 2, by + (bh + ls[1]) // 2),
                  font_scale=0.8, color=text_col, thickness=2, shadow=False)

    # Key-binding hints
    for i, line in enumerate([
        "ENTER / SPACE  ->  Select Level",
        "1 / 2 / 3      ->  Change difficulty",
        "L              ->  Cycle tracked landmark",
        "D              ->  Toggle debug telemetry",
        "M              ->  Toggle mouse fallback",
        "ESC            ->  Quit",
    ]):
        draw_text(canvas, line, (W // 2 - 200, 560 + i * 26),
                  font_scale=0.5, color=(140, 180, 210))


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
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)[0]
    draw_text(canvas, title, ((W - ts[0]) // 2, 85),
              font_scale=1.3, color=(120, 220, 255), thickness=2)

    diff_label = f"DIFFICULTY: {difficulty_name.upper()}  (Press 1, 2, 3 to switch)"
    ds = cv2.getTextSize(diff_label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
    draw_text(canvas, diff_label, ((W - ds[0]) // 2, 130),
              font_scale=0.62, color=(160, 200, 230))

    # Level Card Box
    bx, by, bw, bh = W // 2 - 270, 170, 540, 300
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (22, 32, 48), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 160, 220), 2)

    # Card content
    header = f"Level {level_index + 1} of {total_levels}"
    draw_text(canvas, header, (bx + 35, by + 45), font_scale=0.65, color=(255, 200, 80), thickness=2)

    lvl_title = f"{level_name}"
    draw_text(canvas, lvl_title, (bx + 35, by + 90), font_scale=0.85, color=(230, 240, 255), thickness=2)

    # Details
    draw_text(canvas, f"Optimal Minimum Path:   {min_path_distance:.1f} px",
              (bx + 35, by + 145), font_scale=0.58, color=(180, 210, 235))
    draw_text(canvas, f"Geometric Obstacles:    {wall_count} barriers",
              (bx + 35, by + 185), font_scale=0.58, color=(180, 210, 235))
    draw_text(canvas, f"Movement Space:         Continuous 2D free navigation",
              (bx + 35, by + 225), font_scale=0.55, color=(140, 185, 215))
    draw_text(canvas, f"Status:                 READY to load",
              (bx + 35, by + 265), font_scale=0.55, color=(100, 230, 160))

    # Pulsing selection prompt
    pulse = 0.8 + 0.2 * math.sin(anim_t * 4.0)
    prompt_col = tuple(int(c * pulse) for c in (80, 255, 160))
    confirm_text = "Press ENTER or SPACE to Load Level"
    cs = cv2.getTextSize(confirm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    draw_text(canvas, confirm_text, ((W - cs[0]) // 2, 515),
              font_scale=0.75, color=prompt_col, thickness=2)

    # Navigation instructions
    nav_hints = [
        "LEFT / RIGHT or N / P  ->  Previous / Next Level",
        "1 / 2 / 3              ->  Change Difficulty (Easy / Med / Hard)",
        "M / ESC                ->  Return to Main Menu",
    ]
    for i, line in enumerate(nav_hints):
        draw_text(canvas, line, (W // 2 - 230, 565 + i * 28),
                  font_scale=0.52, color=(140, 175, 205))


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


def draw_paused_overlay(canvas: np.ndarray, W: int, H: int) -> None:
    """Clinical paused state overlay."""
    bw, bh = 340, 90
    bx, by = (W - bw) // 2, (H - bh) // 2
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (70, 95, 125), 1, cv2.LINE_AA)

    draw_text(canvas, "SESSION PAUSED", (bx + 55, by + 38),
              font_scale=0.85, color=(240, 245, 250), thickness=2)
    draw_text(canvas, "Press P to continue", (bx + 85, by + 68),
              font_scale=0.50, color=(140, 175, 210))


def draw_win_overlay(
    canvas:      np.ndarray,
    W: int, H:   int,
    anim_t:      float,
    final_metrics: dict,
    wall_hits:   int,
    elapsed:     float,
) -> None:
    """Professional medical/research prototype clinical session results summary."""
    m = final_metrics
    raw_eff = m.get("path_efficiency", 0.0)
    eff_pct = raw_eff * 100.0 if 0.0 < raw_eff <= 1.0 else raw_eff

    min_dist = m.get("minimum_distance", m.get("min_path_distance_px", 0.0))
    act_dist = m.get("actual_distance", m.get("actual_distance_px", 0.0))
    traj_acc = m.get("trajectory_accuracy", 0.0)
    mean_dev = m.get("mean_path_deviation_px", 0.0)
    dev_events = m.get("deviation_events", 0)
    t_outside = m.get("time_outside_route_s", 0.0)

    rows = [
        # 1. Accuracy Dimension
        ("Trajectory accuracy", f"{traj_acc:.1f}%"),
        ("Mean path deviation", f"{mean_dev:.1f} px"),
        ("Deviation events",    str(dev_events)),
        # 2. Path Efficiency Dimension
        ("Path efficiency",     f"{eff_pct:.1f}%"),
        # 3. Distance Dimension
        ("Minimum path",        f"{min_dist:.1f} px"),
        ("Actual distance",     f"{act_dist:.1f} px"),
        # 4. Collision Count Dimension
        ("Wall collisions",     str(wall_hits)),
        # 5. Time Dimension
        ("Completion time",     f"{m.get('completion_time_s', elapsed):.2f} s"),
        ("Time outside route",  f"{t_outside:.2f} s"),
        # Smoothness & Kinematics
        ("Game smoothness",     f"{m.get('smoothness_score', 100.0):.1f}/100"),
        ("Norm. jerk score",    f"{m.get('normalised_jerk', 0):.2f}"),
        ("Tremor index",        f"{m.get('tremor_index', 0) * 100:.1f}%"),
    ]

    bw, bh = 490, 420
    bx, by = (W - bw) // 2, (H - bh) // 2 - 10

    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (65, 110, 85), 1, cv2.LINE_AA)

    # Header
    title = "REHABILITATION SESSION COMPLETE"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    draw_text(canvas, title, (bx + (bw - ts[0]) // 2, by + 34),
              font_scale=0.72, color=(240, 250, 245), thickness=2)

    sub = "Target Reached - Clinical Movement Evaluation"
    ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
    draw_text(canvas, sub, (bx + (bw - ss[0]) // 2, by + 58),
              font_scale=0.44, color=(130, 200, 160))

    # Divider line
    cv2.line(canvas, (bx + 20, by + 72), (bx + bw - 20, by + 72), (45, 65, 55), 1, cv2.LINE_AA)

    # Metrics Table
    row_h = 24
    table_top = by + 94
    for i, (label, val) in enumerate(rows):
        ry = table_top + i * row_h
        draw_text(canvas, label, (bx + 26, ry),
                  font_scale=0.46, color=(145, 170, 195))
        vs = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        draw_text(canvas, val, (bx + bw - 26 - vs[0], ry),
                  font_scale=0.48, color=(235, 245, 250), thickness=1)

    # Footer navigation
    hint = "Press R to restart  |  N for next level  |  M for menu  |  ESC to quit"
    hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
    draw_text(canvas, hint, ((W - hs[0]) // 2, H - 25),
              font_scale=0.46, color=(130, 165, 150))


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
