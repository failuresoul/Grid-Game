"""
metrics/session_recorder.py — Rehabilitation Session Recording System

Responsible for recording completed rehabilitation game sessions:
  - Top-level clinical & performance metrics (JSON & CSV)
  - Full continuous trajectory data (timestamps, x/y coords)
  - Automatic directory creation
  - Crash-proof error handling (guaranteed never to crash the game)
"""

from __future__ import annotations

import os
import json
import csv
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import config

log = logging.getLogger(__name__)


def generate_session_id(timestamp: Optional[datetime] = None) -> str:
    """Generate a unique, human-readable session identifier."""
    t = timestamp or datetime.now()
    short_uid = uuid.uuid4().hex[:8]
    return f"session_{t.strftime('%Y%m%d_%H%M%S')}_{short_uid}"


def ensure_session_dir(save_dir: Optional[str] = None) -> bool:
    """
    Safely ensure the session output directory exists.
    Guaranteed never to raise an unhandled exception.
    """
    try:
        actual_dir = save_dir if save_dir is not None else getattr(config, "METRICS_SAVE_DIR", "data/sessions")
        os.makedirs(actual_dir, exist_ok=True)
        return True
    except Exception as e:
        actual_dir = save_dir if save_dir is not None else getattr(config, "METRICS_SAVE_DIR", "data/sessions")
        log.error(f"Failed to create session directory '{actual_dir}': {e}")
        return False


def build_session_filename(
    now: datetime,
    save_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Construct filename matching:
      session_YYYY_MM_DD_HHMMSS.json
    If a file with that name already exists in the same second,
    appends an incremental sequence or unique suffix so no session is overwritten.
    """
    actual_dir = save_dir if save_dir is not None else getattr(config, "METRICS_SAVE_DIR", "data/sessions")
    base_name = f"session_{now.strftime('%Y_%m_%d_%H%M%S')}"
    target_path = os.path.join(actual_dir, f"{base_name}.json")
    if not os.path.exists(target_path):
        return target_path

    # Disambiguate if multiple sessions finish in the exact same second
    counter = 1
    while True:
        disambiguated = os.path.join(actual_dir, f"{base_name}_{counter}.json")
        if not os.path.exists(disambiguated):
            return disambiguated
        counter += 1


def save_session(
    *,
    timestamp: Optional[str] = None,
    session_id: Optional[str] = None,
    difficulty: str = "Unknown",
    maze_seed: Optional[Any] = None,
    completion_status: str = "COMPLETED",
    completion_time: float = 0.0,
    actual_distance: float = 0.0,
    minimum_distance: float = 0.0,
    path_efficiency: float = 0.0,
    accuracy: float = 100.0,
    smoothness: float = 100.0,
    collision_count: int = 0,
    deviation_count: int = 0,
    trajectory: Optional[Sequence[Any]] = None,
    optimal_path: Optional[Sequence[Any]] = None,
    level_name: str = "Level",
    game_performance_metrics: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
    save_csv_also: bool = True,
) -> Optional[str]:
    """
    Save complete rehabilitation session recording.

    Required fields:
      - timestamp:         ISO/human-readable timestamp string
      - session_id:        Unique session identifier
      - difficulty:        Difficulty tag (e.g. EASY / MEDIUM / HARD)
      - maze_seed:         Integer seed or null / "FIXED"
      - completion_status: Status (e.g. COMPLETED / WON / TIMEOUT / ABORTED)
      - completion_time:   Duration of game/session in seconds
      - actual_distance:   Total path length traversed in px
      - minimum_distance:  Optimal collision-free path length in px
      - path_efficiency:   Ratio (minimum / actual * 100) percentage
      - accuracy:          Trajectory corridor adherence percentage (0-100)
      - smoothness:        Game-derived movement smoothness score (0-100)
      - collision_count:   Total wall contact count
      - deviation_count:   Count of divergence/deviation episodes
      - trajectory:        List of point samples [{'x': x, 'y': y, 't': t}, ...]

    Guaranteed never to crash if directory doesn't exist or write fails.

    Returns:
        Absolute filepath of the saved JSON file, or None on failure.
    """
    try:
        now_dt = datetime.now()
        ts_str = timestamp or now_dt.strftime("%Y-%m-%d %H:%M:%S")
        sid = session_id or generate_session_id(now_dt)
        actual_dir = save_dir if save_dir is not None else getattr(config, "METRICS_SAVE_DIR", "data/sessions")

        if not ensure_session_dir(actual_dir):
            return None

        # Clean trajectory serialization
        clean_traj: List[Dict[str, float]] = []
        if trajectory:
            for pt in trajectory:
                if isinstance(pt, dict):
                    x = float(pt.get("x", 0.0))
                    y = float(pt.get("y", 0.0))
                    t = float(pt.get("t", 0.0))
                elif isinstance(pt, (tuple, list)):
                    x = float(pt[0])
                    y = float(pt[1])
                    t = float(pt[2]) if len(pt) > 2 else 0.0
                else:
                    continue
                clean_traj.append({"x": round(x, 2), "y": round(y, 2), "t": round(t, 3)})

        # Clean optimal waypoints serialization
        clean_opt: List[Dict[str, float]] = []
        if optimal_path:
            for wp in optimal_path:
                if isinstance(wp, dict):
                    clean_opt.append({"x": round(float(wp.get("x", 0.0)), 2), "y": round(float(wp.get("y", 0.0)), 2)})
                elif isinstance(wp, (tuple, list)) and len(wp) >= 2:
                    clean_opt.append({"x": round(float(wp[0]), 2), "y": round(float(wp[1]), 2)})

        # Seed normalization
        normalized_seed: Optional[Union[int, str]] = None
        if maze_seed is not None:
            if isinstance(maze_seed, (int, float)):
                normalized_seed = int(maze_seed)
            elif str(maze_seed).upper() in ("NONE", "NULL", ""):
                normalized_seed = None
            else:
                try:
                    normalized_seed = int(maze_seed)
                except (ValueError, TypeError):
                    normalized_seed = str(maze_seed)

        # Structure full session record
        session_data: Dict[str, Any] = {
            "timestamp":         ts_str,
            "session_id":        sid,
            "difficulty":        str(difficulty).upper(),
            "maze_seed":         normalized_seed,
            "completion_status": str(completion_status).upper(),
            "completion_time":   round(float(completion_time), 2),
            "actual_distance":   round(float(actual_distance), 1),
            "minimum_distance":  round(float(minimum_distance), 1),
            "path_efficiency":   round(float(path_efficiency), 1),
            "accuracy":          round(float(accuracy), 1),
            "smoothness":        round(float(smoothness), 1),
            "collision_count":   int(collision_count),
            "deviation_count":   int(deviation_count),
            "trajectory":        clean_traj,
            "optimal_path":      clean_opt,
            "level_name":        str(level_name),
            "game_performance_metrics": game_performance_metrics or {},
        }

        json_path = build_session_filename(now_dt, save_dir=actual_dir, session_id=sid)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        log.info(f"Rehabilitation session JSON saved -> {json_path}")

        # Maintain CSV compatibility if enabled
        if save_csv_also:
            _append_session_to_csv(
                session_data=session_data,
                save_dir=actual_dir,
                now_dt=now_dt,
            )

        return os.path.abspath(json_path)

    except Exception as e:
        log.error(f"Safe session recording failed: {e}", exc_info=True)
        return None


def _append_session_to_csv(
    session_data: Dict[str, Any],
    save_dir: str,
    now_dt: datetime,
) -> Optional[str]:
    """Append summary session row to daily CSV file."""
    try:
        csv_filename = now_dt.strftime("%Y-%m-%d") + "_sessions.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        file_exists = os.path.isfile(csv_path)

        # Extract primary metrics
        extra_metrics = session_data.get("game_performance_metrics", {}) or {}
        row: Dict[str, Any] = {
            "timestamp":         session_data["timestamp"],
            "session_id":        session_data["session_id"],
            "difficulty":        session_data["difficulty"],
            "level":             session_data.get("level_name", "Unknown"),
            "maze_seed":         str(session_data["maze_seed"]) if session_data["maze_seed"] is not None else "FIXED",
            "completion_status": session_data["completion_status"],
            "completion_time_s": session_data["completion_time"],
            "actual_distance":   session_data["actual_distance"],
            "minimum_distance":  session_data["minimum_distance"],
            "path_efficiency":   session_data["path_efficiency"],
            "accuracy":          session_data["accuracy"],
            "smoothness":        session_data["smoothness"],
            "collision_count":   session_data["collision_count"],
            "deviation_count":   session_data["deviation_count"],
            "trajectory_points": len(session_data.get("trajectory", [])),
        }

        # Include additional metrics for regression analysis
        for k, v in extra_metrics.items():
            if k not in row and not isinstance(v, (dict, list)):
                row[k] = v

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return csv_path
    except Exception as e:
        log.error(f"Failed to append to session CSV: {e}")
        return None


def load_session(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a previously recorded session JSON file.
    Useful for longitudinal rehabilitation progress analysis.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to load session from '{filepath}': {e}")
        return None
