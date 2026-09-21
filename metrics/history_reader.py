"""
metrics/history_reader.py — Rehabilitation Session History & Progress Loader

Reads recorded sessions from data/sessions/ (JSON & CSV),
standardizes historical performance measurements, provides difficulty filtering,
and extracts trend metrics for longitudinal visualization.
"""

from __future__ import annotations

import os
import glob
import json
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import config

log = logging.getLogger(__name__)


def _parse_timestamp(ts_str: str) -> datetime:
    """Safely parse diverse timestamp formats into datetime object."""
    if not ts_str:
        return datetime.min
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d_%H%M%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return datetime.min


def _normalize_session_dict(raw: Dict[str, Any], default_id: str = "") -> Dict[str, Any]:
    """Standardize a single session dictionary to ensure uniform metric types."""
    ts_str = str(raw.get("timestamp", ""))
    sid = str(raw.get("session_id", default_id or ts_str))
    diff = str(raw.get("difficulty", "EASY")).strip().upper()
    status = str(raw.get("completion_status", "COMPLETED")).strip().upper()

    # Float & Int fields with safe fallback
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _to_int(val: Any, default: int = 0) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Handle alternate field names from older CSV logs
    c_time = _to_float(raw.get("completion_time", raw.get("completion_time_s", 0.0)))
    act_d  = _to_float(raw.get("actual_distance", raw.get("actual_distance_px", raw.get("path_length_px", 0.0))))
    min_d  = _to_float(raw.get("minimum_distance", raw.get("min_path_distance_px", raw.get("ideal_distance_px", 0.0))))
    
    raw_eff = _to_float(raw.get("path_efficiency", 0.0))
    # Normalize efficiency if stored as 0.0-1.0 ratio
    eff = raw_eff * 100.0 if 0.0 < raw_eff <= 1.0 else raw_eff

    acc = _to_float(raw.get("accuracy", raw.get("trajectory_accuracy", raw.get("corridor_adherence_pct", 100.0))))
    smooth = _to_float(raw.get("smoothness", raw.get("smoothness_score", raw.get("game_smoothness_score", 100.0))))
    colls = _to_int(raw.get("collision_count", raw.get("wall_hits", raw.get("collisions", 0))))
    devs = _to_int(raw.get("deviation_count", raw.get("deviation_events", 0)))

    seed_val = raw.get("maze_seed")
    if seed_val is not None:
        try:
            seed_val = int(seed_val)
        except (ValueError, TypeError):
            seed_val = str(seed_val)

    return {
        "timestamp":         ts_str,
        "session_id":        sid,
        "difficulty":        diff,
        "maze_seed":         seed_val,
        "completion_status": status,
        "completion_time":   round(c_time, 2),
        "actual_distance":   round(act_d, 1),
        "minimum_distance":  round(min_d, 1),
        "path_efficiency":   round(eff, 1),
        "accuracy":          round(acc, 1),
        "smoothness":        round(smooth, 1),
        "collision_count":   colls,
        "deviation_count":   devs,
        "level_name":        str(raw.get("level_name", raw.get("level", "Unknown"))),
        "parsed_dt":         _parse_timestamp(ts_str),
    }


def load_all_sessions(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scan session directory, load all JSON and CSV session records,
    deduplicate, and return sorted chronologically (oldest to newest).
    Guaranteed never to crash if directory is missing or files are corrupted.
    """
    actual_dir = data_dir if data_dir is not None else getattr(config, "METRICS_SAVE_DIR", "data/sessions")
    if not os.path.isdir(actual_dir):
        return []

    sessions: List[Dict[str, Any]] = []
    seen_ids = set()

    # 1. Load JSON session files (primary source with full trajectory)
    try:
        json_pattern = os.path.join(actual_dir, "*.json")
        for jpath in glob.glob(json_pattern):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                # Require at least one session-identifying attribute
                if not any(k in data for k in ("timestamp", "completion_status", "difficulty", "completion_time", "actual_distance")):
                    continue
                norm = _normalize_session_dict(data, default_id=os.path.basename(jpath))
                sid = norm["session_id"]
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    sessions.append(norm)
            except Exception as e:
                log.warning(f"Could not read session JSON '{jpath}': {e}")
    except Exception as e:
        log.error(f"Error scanning session JSON files in '{actual_dir}': {e}")

    # 2. Supplement from daily CSV files (legacy or additional sessions)
    try:
        csv_pattern = os.path.join(actual_dir, "*_sessions.csv")
        for cpath in glob.glob(csv_pattern):
            try:
                with open(cpath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        sid = row.get("session_id", f"{os.path.basename(cpath)}_{idx}")
                        if sid not in seen_ids:
                            seen_ids.add(sid)
                            norm = _normalize_session_dict(row, default_id=sid)
                            sessions.append(norm)
            except Exception as e:
                log.warning(f"Could not read session CSV '{cpath}': {e}")
    except Exception as e:
        log.error(f"Error scanning session CSV files in '{actual_dir}': {e}")

    # Sort chronologically by timestamp
    sessions.sort(key=lambda s: s.get("parsed_dt", datetime.min))
    return sessions


def filter_sessions(sessions: Sequence[Dict[str, Any]], difficulty: str = "ALL") -> List[Dict[str, Any]]:
    """
    Filter session list by difficulty: 'ALL', 'EASY', 'MEDIUM', or 'HARD'.
    """
    diff_filter = difficulty.strip().upper()
    if diff_filter in ("ALL", "", "ANY"):
        return list(sessions)
    return [s for s in sessions if diff_filter in s.get("difficulty", "").upper()]


def extract_trend_series(sessions: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extract continuous series for trend plotting:
      - completion_time
      - actual_distance
      - path_efficiency
      - accuracy
      - smoothness
      - collision_count
    """
    series: Dict[str, List[float]] = {
        "completion_time": [],
        "actual_distance": [],
        "path_efficiency": [],
        "accuracy": [],
        "smoothness": [],
        "collision_count": [],
    }
    for s in sessions:
        series["completion_time"].append(float(s.get("completion_time", 0.0)))
        series["actual_distance"].append(float(s.get("actual_distance", 0.0)))
        series["path_efficiency"].append(float(s.get("path_efficiency", 0.0)))
        series["accuracy"].append(float(s.get("accuracy", 100.0)))
        series["smoothness"].append(float(s.get("smoothness", 100.0)))
        series["collision_count"].append(float(s.get("collision_count", 0)))
    return series
