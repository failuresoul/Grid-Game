"""
metrics/adaptive.py — Clinical Adaptive Difficulty Recommendation System

Evaluates multi-dimensional motor quality (accuracy, path efficiency, smoothness,
completion status, and collisions) rather than raw time to recommend appropriate
difficulty progression or reinforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class AdaptiveRecommendation:
    """Encapsulates an adaptive difficulty recommendation."""
    action: str              # "INCREASE" | "MAINTAIN" | "DECREASE"
    current_difficulty: str  # "EASY" | "MEDIUM" | "HARD"
    target_difficulty: str   # "EASY" | "MEDIUM" | "HARD"
    badge_title: str         # Short visual tag
    headline: str            # Primary guidance text
    rationale: str           # Detailed reasoning citing metrics
    color: Tuple[int, int, int] = (100, 220, 140)  # BGR
    criteria_met: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "current_difficulty": self.current_difficulty,
            "target_difficulty": self.target_difficulty,
            "badge_title": self.badge_title,
            "headline": self.headline,
            "rationale": self.rationale,
            "criteria_met": self.criteria_met,
        }


def _is_high_performance(
    metrics: Dict[str, Any],
    wall_hits: int,
    is_completed: bool,
    acc_th: float,
    eff_th: float,
    sm_th: float,
    max_coll: int,
) -> Tuple[bool, Dict[str, bool]]:
    """Determine if a single session meets all criteria for high motor control."""
    acc = float(metrics.get("trajectory_accuracy", metrics.get("accuracy", 100.0)))
    eff = float(metrics.get("path_efficiency", 0.0))
    # normalize efficiency if stored as ratio
    if 0.0 < eff <= 1.0:
        eff *= 100.0
    smooth = float(metrics.get("smoothness_score", metrics.get("smoothness", 100.0)))

    c_comp = bool(is_completed)
    c_acc  = acc >= acc_th
    c_eff  = eff >= eff_th
    c_sm   = smooth >= sm_th
    c_coll = wall_hits <= max_coll

    criteria = {
        "completion": c_comp,
        "accuracy":   c_acc,
        "efficiency": c_eff,
        "smoothness": c_sm,
        "collisions": c_coll,
    }
    all_met = all(criteria.values())
    return all_met, criteria


def evaluate_adaptive_difficulty(
    current_metrics: Dict[str, Any],
    current_difficulty: str = "EASY",
    wall_hits: int = 0,
    completion_status: str = "COMPLETED",
    session_history: Optional[Sequence[Dict[str, Any]]] = None,
    accuracy_threshold: Optional[float] = None,
    efficiency_threshold: Optional[float] = None,
    smoothness_threshold: Optional[float] = None,
    max_collisions: Optional[int] = None,
    consistency_window: Optional[int] = None,
) -> Optional[AdaptiveRecommendation]:
    """
    Evaluate session kinematics and generate clinical difficulty recommendation.

    Returns:
        AdaptiveRecommendation, or None if ADAPTIVE_DIFFICULTY is disabled in config.
    """
    if not getattr(config, "ADAPTIVE_DIFFICULTY", True):
        return None

    # Load thresholds (configurable with overrides)
    acc_th = accuracy_threshold if accuracy_threshold is not None else getattr(config, "ACCURACY_THRESHOLD", 90.0)
    eff_th = efficiency_threshold if efficiency_threshold is not None else getattr(config, "EFFICIENCY_THRESHOLD", 85.0)
    sm_th  = smoothness_threshold if smoothness_threshold is not None else getattr(config, "SMOOTHNESS_THRESHOLD", 75.0)
    max_c  = max_collisions if max_collisions is not None else getattr(config, "MAX_COLLISIONS_THRESHOLD", 1)
    window = consistency_window if consistency_window is not None else getattr(config, "ADAPTIVE_CONSISTENCY_WINDOW", 2)

    low_acc  = getattr(config, "LOW_ACCURACY_THRESHOLD", 65.0)
    low_eff  = getattr(config, "LOW_EFFICIENCY_THRESHOLD", 50.0)
    low_sm   = getattr(config, "LOW_SMOOTHNESS_THRESHOLD", 40.0)
    high_c   = getattr(config, "HIGH_COLLISIONS_THRESHOLD", 4)

    diff_upper = current_difficulty.strip().upper()
    is_comp = completion_status.strip().upper() in ("COMPLETED", "WON", "SUCCESS")

    acc = float(current_metrics.get("trajectory_accuracy", current_metrics.get("accuracy", 100.0)))
    raw_eff = float(current_metrics.get("path_efficiency", 0.0))
    eff = raw_eff * 100.0 if 0.0 < raw_eff <= 1.0 else raw_eff
    smooth = float(current_metrics.get("smoothness_score", current_metrics.get("smoothness", 100.0)))

    # Evaluate current session
    is_strong, criteria = _is_high_performance(
        current_metrics, wall_hits, is_comp, acc_th, eff_th, sm_th, max_c
    )

    # ── 1. Check for Consistent High Performance (Promotion) ─────────────────
    if is_strong:
        consecutive_strong = 1
        if session_history and window > 1:
            # Check previous sessions of the same difficulty
            same_diff_history = [
                s for s in session_history
                if diff_upper in str(s.get("difficulty", "")).upper()
            ]
            # Take the most recent sessions up to window - 1
            for prev_s in reversed(same_diff_history[:window - 1]):
                p_status = prev_s.get("completion_status", "COMPLETED")
                p_hits = int(prev_s.get("collision_count", 0))
                p_strong, _ = _is_high_performance(
                    prev_s, p_hits, p_status in ("COMPLETED", "WON"),
                    acc_th, eff_th, sm_th, max_c
                )
                if p_strong:
                    consecutive_strong += 1
                else:
                    break

        if consecutive_strong >= window:
            # Consistent high performance established!
            if "EAS" in diff_upper:
                return AdaptiveRecommendation(
                    action="INCREASE",
                    current_difficulty="EASY",
                    target_difficulty="MEDIUM",
                    badge_title="SUGGEST ADVANCING",
                    headline="Consistent high motor control. Suggest increasing to MEDIUM.",
                    rationale=f"High accuracy ({acc:.0f}%), efficiency ({eff:.0f}%), smoothness ({smooth:.0f}/100) with {wall_hits} wall hits.",
                    color=(90, 225, 140),  # Mint
                    criteria_met=criteria,
                )
            elif "MED" in diff_upper:
                return AdaptiveRecommendation(
                    action="INCREASE",
                    current_difficulty="MEDIUM",
                    target_difficulty="HARD",
                    badge_title="SUGGEST ADVANCING",
                    headline="Excellent corridor adherence. Suggest increasing to HARD.",
                    rationale=f"Consistent accuracy ({acc:.0f}%) and smooth control ({smooth:.0f}/100) across sessions.",
                    color=(90, 225, 140),  # Mint
                    criteria_met=criteria,
                )
            else:  # HARD
                return AdaptiveRecommendation(
                    action="MAINTAIN",
                    current_difficulty="HARD",
                    target_difficulty="HARD",
                    badge_title="MAXIMUM DIFFICULTY",
                    headline="Demonstrated high control on HARD. Continue regular practice.",
                    rationale=f"Outstanding motor control with {wall_hits} collisions and {acc:.0f}% accuracy.",
                    color=(60, 205, 245),  # Amber
                    criteria_met=criteria,
                )
        else:
            # Strong performance, but needs consistency
            needed = window - consecutive_strong
            return AdaptiveRecommendation(
                action="MAINTAIN",
                current_difficulty=diff_upper,
                target_difficulty=diff_upper,
                badge_title="STRONG RUN",
                headline=f"Excellent run! Complete {needed} more consistent session to advance.",
                rationale=f"Accuracy ({acc:.0f}%) and smoothness ({smooth:.0f}) met. Reinforce consistency.",
                color=(100, 215, 170),
                criteria_met=criteria,
            )

    # ── 2. Check for Severe Struggle / Fatigue (Demotion / Easier Level) ──────
    is_struggling = (
        (not is_comp) or
        (acc < low_acc) or
        (eff < low_eff) or
        (smooth < low_sm) or
        (wall_hits >= high_c)
    )

    if is_struggling:
        reasons = []
        if not is_comp:
            reasons.append("incomplete")
        if wall_hits >= high_c:
            reasons.append(f"{wall_hits} collisions")
        if acc < low_acc:
            reasons.append(f"{acc:.0f}% acc")
        if eff < low_eff:
            reasons.append(f"{eff:.0f}% eff")
        if smooth < low_sm:
            reasons.append(f"{smooth:.0f} smoothness")
        reason_str = ", ".join(reasons)

        if "HAR" in diff_upper:
            return AdaptiveRecommendation(
                action="DECREASE",
                current_difficulty="HARD",
                target_difficulty="MEDIUM",
                badge_title="SUGGEST EASIER LEVEL",
                headline="High difficulty detected. Suggest using MEDIUM level.",
                rationale=f"Observed {reason_str}. Wider corridors reduce compensatory motor strain.",
                color=(80, 95, 245),  # Coral
                criteria_met=criteria,
            )
        elif "MED" in diff_upper:
            return AdaptiveRecommendation(
                action="DECREASE",
                current_difficulty="MEDIUM",
                target_difficulty="EASY",
                badge_title="SUGGEST EASIER LEVEL",
                headline="Corridor challenge detected. Suggest using EASY level.",
                rationale=f"Observed {reason_str}. An easier level reinforces calm motor control.",
                color=(80, 95, 245),  # Coral
                criteria_met=criteria,
            )
        else:  # EASY
            return AdaptiveRecommendation(
                action="DECREASE",
                current_difficulty="EASY",
                target_difficulty="EASY",
                badge_title="SUGGEST REPEATING",
                headline="Suggest repeating current difficulty with focus on control.",
                rationale=f"Observed {reason_str}. Focus on calm, unhurried trajectory over speed.",
                color=(80, 95, 245),  # Coral
                criteria_met=criteria,
            )

    # ── 3. Moderate Steady Performance (Maintain & Practice) ─────────────────
    return AdaptiveRecommendation(
        action="MAINTAIN",
        current_difficulty=diff_upper,
        target_difficulty=diff_upper,
        badge_title="STEADY PRACTICE",
        headline="Steady execution. Suggest repeating current difficulty to practice.",
        rationale=f"Performance ({acc:.0f}% acc, {smooth:.0f} smooth, {wall_hits} hits) is developing well.",
        color=(215, 190, 90),  # Calm Gold/Blue
        criteria_met=criteria,
    )
