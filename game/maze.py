"""
game/maze.py — MazeGenerator & Level Management

Holds the level library and supplies levels on demand.
Supports:
  - Pre-designed continuous 2D levels across Easy, Medium, and Hard
  - Conceptual maze (user's START -> entrance -> obstacle -> END design)
  - Procedural continuous 2D maze generation (wide corridors, randomized geometric obstacles)
  - JSON serialization & loading to easily modify and share mazes

Dependency chain:
    game.maze → game.levels, config
"""

from __future__ import annotations
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from game.levels import (
    Level,
    MazeBuilder,
    Obstacle,
    PolygonObstacle,
    RectObstacle,
    conceptual_level,
    easy_levels,
    medium_levels,
    hard_levels,
)

log = logging.getLogger(__name__)


class MazeGenerator:
    """
    Builds and serves the library of continuous 2D geometric maze levels.

    Usage:
        gen   = MazeGenerator()
        level = gen.get_level(difficulty=1, index=0)
        level = gen.next_level(difficulty=2)
        proc  = gen.generate_procedural_level(difficulty=2, seed=42)
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        W, H = canvas_w, canvas_h
        self._library: Dict[int, List[Level]] = {
            1: easy_levels(W, H),
            2: medium_levels(W, H),
            3: hard_levels(W, H),
        }
        # Sequential index per difficulty (for next_level())
        self._cursors: Dict[int, int] = {1: 0, 2: 0, 3: 0}

        total = sum(len(v) for v in self._library.values())
        log.info(f"MazeGenerator ready — {total} levels across 3 difficulties")

    def get_level(self, difficulty: int, index: int = 0) -> Level:
        """
        Return a specific level by difficulty and index (wraps around).
        Respects config.MAZE_TYPE:
          - "FIXED": returns curated deterministic levels from library.
          - "RANDOM": dynamically generates a solvable procedural level with config.MAZE_SEED.

        Args:
            difficulty: 1 = Easy, 2 = Medium, 3 = Hard
            index:      0-based level index within the difficulty pool
        """
        if getattr(config, "MAZE_TYPE", "FIXED").upper() == "RANDOM":
            seed = getattr(config, "MAZE_SEED", None)
            return self.generate_procedural_level(difficulty=difficulty, seed=seed)

        pool = self._library.get(difficulty, self._library[1])
        return pool[index % len(pool)]

    def next_level(self, difficulty: int) -> Level:
        """Return the next level sequentially and advance the cursor."""
        if getattr(config, "MAZE_TYPE", "FIXED").upper() == "RANDOM":
            seed = getattr(config, "MAZE_SEED", None)
            return self.generate_procedural_level(difficulty=difficulty, seed=seed)

        pool = self._library.get(difficulty, self._library[1])
        idx  = self._cursors.get(difficulty, 0)
        level = pool[idx % len(pool)]
        self._cursors[difficulty] = (idx + 1) % len(pool)
        return level

    def level_count(self, difficulty: int) -> int:
        """Number of levels available for a given difficulty."""
        return len(self._library.get(difficulty, []))

    def get_conceptual_level(self) -> Level:
        """Return a fresh instance of the user's conceptual maze."""
        return conceptual_level(self.canvas_w, self.canvas_h)

    def add_custom_level(self, difficulty: int, level: Level) -> None:
        """Register a new level in the difficulty pool."""
        if difficulty not in self._library:
            self._library[difficulty] = []
        self._library[difficulty].append(level)
        log.info(f"Registered custom level '{level.name}' under difficulty {difficulty}")

    def save_level_to_file(self, level: Level, filepath: str | Path) -> None:
        """Export a level to a readable JSON file for easy editing."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(level.to_dict(), f, indent=2)
        log.info(f"Saved level '{level.name}' to {path}")

    def load_level_from_file(self, filepath: str | Path) -> Level:
        """Load a level definition from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        level = Level.from_dict(data)
        log.info(f"Loaded level '{level.name}' from {path}")
        return level

    def generate_procedural_level(
        self,
        difficulty: int = 1,
        seed: Optional[int] = None,
    ) -> Level:
        """
        Generate a validated, physically solvable procedural continuous 2D maze.
        Evaluates player radius collision clearance, computes analytical waypoints,
        and guarantees reachability.
        """
        from game.procedural import generate_procedural_maze
        return generate_procedural_maze(
            difficulty=difficulty,
            seed=seed,
            width=self.canvas_w,
            height=self.canvas_h,
        )

