"""
game/maze.py — MazeGenerator

Holds the level library and supplies levels on demand.
Thin façade over game.levels — the actual level data lives there.

Dependency chain:
    game.maze → game.levels, config
"""

from __future__ import annotations
import logging
from typing import Dict, List

import config
from game.levels import Level, easy_levels, medium_levels, hard_levels

log = logging.getLogger(__name__)


class MazeGenerator:
    """
    Builds and serves the library of pre-designed maze levels.

    Usage:
        gen   = MazeGenerator()
        level = gen.get_level(difficulty=1, index=0)
        level = gen.next_level(difficulty=2)   # advance sequentially
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
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

        Args:
            difficulty: 1 = Easy, 2 = Medium, 3 = Hard
            index:      0-based level index within the difficulty pool
        """
        pool = self._library.get(difficulty, self._library[1])
        return pool[index % len(pool)]

    def next_level(self, difficulty: int) -> Level:
        """Return the next level sequentially and advance the cursor."""
        pool = self._library.get(difficulty, self._library[1])
        idx  = self._cursors.get(difficulty, 0)
        level = pool[idx % len(pool)]
        self._cursors[difficulty] = (idx + 1) % len(pool)
        return level

    def level_count(self, difficulty: int) -> int:
        """Number of levels available for a given difficulty."""
        return len(self._library.get(difficulty, []))
