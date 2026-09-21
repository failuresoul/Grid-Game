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
        "H              ->  View Progress & History (Sessions & Trends)",
        "L              ->  Cycle tracked landmark",
        "D              ->  Toggle debug telemetry",
        "M              ->  Toggle mouse fallback",
        "ESC            ->  Quit",
    ]):
        draw_text(canvas, line, (W // 2 - 200, 555 + i * 24),
                  font_scale=0.48, color=(140, 180, 210))


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
    draw_text(canvas, status_label, (W - 250, 36),
              font_scale=0.48, color=(90, 220, 140), thickness=1)

    # Difficulty badge
    draw_text(canvas, f"Difficulty: {diff_tag}", (W - 250, 60),
              font_scale=0.48, color=diff_col, thickness=1)

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
