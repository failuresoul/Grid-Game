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
    glow_layers: int = 3,
) -> None:
    """Draw a filled circle with concentric translucent glow rings."""
    for i in range(glow_layers, 0, -1):
        glow_r  = radius + i * 5
        alpha   = 0.15 / i
        overlay = img.copy()
        cv2.circle(overlay, centre, glow_r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, centre, radius, color, -1, cv2.LINE_AA)
    # Bright specular highlight
    hi_r = max(radius // 3, 2)
    hi_c = (centre[0] - radius // 4, centre[1] - radius // 4)
    bright = tuple(min(255, int(c * 1.5)) for c in color)
    cv2.circle(img, hi_c, hi_r, bright, -1, cv2.LINE_AA)


def draw_star(
    img:     np.ndarray,
    centre:  Tuple[int, int],
    outer_r: int,
    color:   Tuple[int, int, int],
) -> None:
    """Draw a filled 5-pointed star polygon."""
    pts   = []
    inner = outer_r // 2
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r     = outer_r if i % 2 == 0 else inner
        pts.append([
            int(centre[0] + r * math.cos(angle)),
            int(centre[1] + r * math.sin(angle)),
        ])
    poly = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img, poly, color, cv2.LINE_AA)
    cv2.polylines(img, poly, True, (255, 255, 255), 1, cv2.LINE_AA)


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
        "ENTER / SPACE  →  Start game",
        "P              →  Pause",
        "R              →  Restart",
        "ESC            →  Quit",
        "1 / 2 / 3      →  Change difficulty",
    ]):
        draw_text(canvas, line, (W // 2 - 200, 560 + i * 30),
                  font_scale=0.5, color=(140, 180, 210))


# ─────────────────────────────────────────────────────────────────────────────
#  In-game overlays
# ─────────────────────────────────────────────────────────────────────────────

def draw_waiting_overlay(canvas: np.ndarray, W: int, H: int, anim_t: float) -> None:
    """Shown before the player starts moving — prompts to show hand."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    pulse = 0.7 + 0.3 * math.sin(anim_t * 3)
    color = tuple(int(c * pulse) for c in (100, 220, 255))
    draw_text(canvas, "SHOW YOUR HAND", (W // 2 - 160, H // 2 - 30),
              font_scale=1.0, color=color, thickness=2)
    draw_text(canvas, "Move away from START to begin", (W // 2 - 200, H // 2 + 20),
              font_scale=0.6, color=(160, 200, 230))


def draw_paused_overlay(canvas: np.ndarray, W: int, H: int) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    draw_text(canvas, "PAUSED", (W // 2 - 80, H // 2),
              font_scale=1.4, color=(200, 220, 255), thickness=3)
    draw_text(canvas, "Press P to continue", (W // 2 - 130, H // 2 + 50),
              font_scale=0.65, color=(140, 180, 220))


def draw_win_overlay(
    canvas:      np.ndarray,
    W: int, H:   int,
    anim_t:      float,
    final_metrics: dict,
    wall_hits:   int,
    elapsed:     float,
) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (5, 25, 15), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    pulse = 0.8 + 0.2 * math.sin(anim_t * 5)
    color = tuple(int(c * pulse) for c in config.WIN_COLOR)
    draw_text(canvas, "GOAL REACHED!", (W // 2 - 160, H // 2 - 155),
              font_scale=1.3, color=color, thickness=3)

    m = final_metrics
    raw_eff = m.get("path_efficiency", 0.0)
    if 0.0 < raw_eff <= 1.0:
        eff_pct = raw_eff * 100.0
    else:
        eff_pct = raw_eff

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
    bx, by = W // 2 - 240, H // 2 - 135
    box_w = 480
    row_h = 26
    box_h = len(rows) * row_h + 16
    cv2.rectangle(canvas, (bx - 10, by - 10),
                  (bx + box_w, by + box_h), (20, 40, 30), -1)
    cv2.rectangle(canvas, (bx - 10, by - 10),
                  (bx + box_w, by + box_h), config.WIN_COLOR, 1)
    for i, (label, val) in enumerate(rows):
        ry = by + i * row_h + 18
        draw_text(canvas, label, (bx, ry),
                  font_scale=0.48, color=config.HUD_LABEL_COLOR)
        draw_text(canvas, val, (bx + 290, ry),
                  font_scale=0.50, color=config.HUD_TEXT_COLOR, thickness=1)

    draw_text(canvas, "Press R to restart  |  ESC to quit",
              (W // 2 - 200, H - 35), font_scale=0.55, color=(140, 200, 160))


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
