"""
maze_generator.py — Continuous-space Level Layout Generator

Levels are NOT grid-based.  The player moves freely in 2-D pixel space.
Obstacles are axis-aligned rectangles.  The generator guarantees:
  - A clear navigable path from Start to End always exists.
  - Minimum corridor width is respected per difficulty.
  - Start and End zones are always reachable.

Level structure:
  - walls:   List[pygame.Rect-style dicts]  →  {"x", "y", "w", "h"}
             (stored as plain tuples (x, y, w, h) to avoid pygame dependency)
  - start:   (cx, cy)  — centre of the start zone
  - end:     (cx, cy)  — centre of the end zone
  - start_r: radius of start zone
  - end_r:   radius of end zone
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import config


# Axis-aligned rectangle: (x, y, w, h) — top-left origin, all in pixels
Rect = Tuple[int, int, int, int]


@dataclass
class Level:
    walls:   List[Rect]
    start:   Tuple[int, int]
    end:     Tuple[int, int]
    start_r: int = 24
    end_r:   int = 24
    name:    str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Handcrafted levels — deterministic, clinically sensible layouts
# ─────────────────────────────────────────────────────────────────────────────
#
# Coordinate system: (0,0) = top-left of canvas
# Canvas size is read from config (default 1000 × 700)
#
# Wall tuples: (left, top, width, height)

def _easy_levels(W: int, H: int) -> List[Level]:
    """
    Easy: Very open space with 2–3 gentle obstacles.
    Large corridors (≥ 120 px wide).  Designed for severe motor impairment.
    """
    margin = 60   # keep walls away from edges
    levels = []

    # ── Easy Level 1 — Three-room layout ─────────────────────────────────────
    # Two horizontal walls with a wide gap in the middle, plus one vertical divider
    levels.append(Level(
        walls=[
            # Top horizontal wall with gap on the right
            (margin, H//3, W//2 - 80, 22),
            # Bottom horizontal wall with gap on the left
            (W//2 + 80, H*2//3, W//2 - 80 - margin, 22),
            # Short vertical divider in the centre
            (W//2 - 11, H//3 + 22, 22, H//3 - 22),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-1: Three Rooms",
    ))

    # ── Easy Level 2 — Single S-curve ────────────────────────────────────────
    levels.append(Level(
        walls=[
            # Upper horizontal bar — leaves right side open
            (margin, H//3, W * 2 // 3, 22),
            # Lower horizontal bar — leaves left side open
            (W//3, H * 2 // 3, W * 2 // 3 - margin, 22),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-2: S-Curve",
    ))

    # ── Easy Level 3 — Open field with one island ─────────────────────────────
    levels.append(Level(
        walls=[
            # Centre island block
            (W//2 - 80, H//2 - 60, 160, 120),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-3: Island",
    ))

    return levels


def _medium_levels(W: int, H: int) -> List[Level]:
    """
    Medium: Multiple turns and bends.  Corridors ≥ 70 px.
    """
    margin = 60
    levels = []

    # ── Medium Level 1 — Zigzag corridor ─────────────────────────────────────
    t = 22   # wall thickness
    levels.append(Level(
        walls=[
            # Row 1 — blocks left side, opens right
            (margin, H//5,        W - margin - 130, t),
            # Row 2 — blocks right side, opens left
            (margin + 130, H*2//5, W - margin - 130, t),
            # Row 3 — blocks left side, opens right
            (margin, H*3//5,      W - margin - 130, t),
            # Row 4 — blocks right side, opens left
            (margin + 130, H*4//5, W - margin - 130, t),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Medium-1: Zigzag",
    ))

    # ── Medium Level 2 — Spiralling boxes ────────────────────────────────────
    # Outer frame with gaps on alternating sides
    t = 22
    levels.append(Level(
        walls=[
            # Outer top — gap on right
            (margin, margin + 80,       W - margin - 120, t),
            # Outer right — gap on bottom
            (W - margin - t, margin,    t, H - margin - 120),
            # Middle top — gap on left
            (margin + 120, margin + 200, W - margin - 240, t),
            # Middle left — gap on top
            (margin + 120, margin + 200, t, H - margin - 320),
            # Inner top — gap on right
            (margin + 120, H - margin - 180, W - margin - 280, t),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W//2, H//2),
        name="Medium-2: Spiral",
    ))

    # ── Medium Level 3 — Grid of pillars ─────────────────────────────────────
    pillar_size = 70
    gap = 140   # gap between pillars
    pillars = []
    for row in range(1, 4):
        for col in range(1, 5):
            px = col * (pillar_size + gap) - 40
            py = row * (pillar_size + gap) - 20
            if px + pillar_size < W - margin and py + pillar_size < H - margin:
                pillars.append((px, py, pillar_size, pillar_size))
    levels.append(Level(
        walls=pillars,
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Medium-3: Pillars",
    ))

    return levels


def _hard_levels(W: int, H: int) -> List[Level]:
    """
    Hard: Many obstacles, narrow passages (≥ 40 px), dead ends.
    """
    margin = 50
    t = 20   # wall thickness
    levels = []

    # ── Hard Level 1 — Dense zigzag with dead ends ───────────────────────────
    walls = []
    # Five horizontal rows of walls; alternate which side is open
    row_ys = [H//6, H*2//6, H*3//6, H*4//6, H*5//6]
    for i, y in enumerate(row_ys):
        if i % 2 == 0:
            # Gap on the right (last 80 px open)
            walls.append((margin, y, W - margin - 80, t))
            # Dead-end stub on left
            walls.append((margin, y + t, t, 55))
        else:
            # Gap on the left (first 80 px open)
            walls.append((margin + 80, y, W - margin - 80, t))
            # Dead-end stub on right
            walls.append((W - margin - t, y + t, t, 55))
    levels.append(Level(
        walls=walls,
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Hard-1: Dense Zigzag",
    ))

    # ── Hard Level 2 — Narrow corridor maze ──────────────────────────────────
    # Alternating horizontal barriers that force a winding path
    walls2 = []
    segs = 8
    seg_h = (H - 2 * margin) // segs
    for i in range(segs):
        y = margin + i * seg_h
        if i % 2 == 0:
            # Wall on left, gap on far right
            walls2.append((margin, y, W - margin - 55, t))
        else:
            # Wall on right, gap on far left
            walls2.append((margin + 55, y, W - margin - 55, t))
    levels.append(Level(
        walls=walls2,
        start=(margin + 20, H - margin - 20),
        end=(W - margin - 20, margin + 20),
        name="Hard-2: Narrow Corridors",
    ))

    # ── Hard Level 3 — Cross-hatch with small gaps ───────────────────────────
    walls3 = []
    gap_size = 55
    # Vertical columns
    for col in range(1, 7):
        x = col * (W // 7)
        # Random gap position per column (seeded for reproducibility)
        rng = random.Random(col * 17)
        gap_y = rng.randint(margin + 60, H - margin - gap_size - 60)
        # Top segment
        walls3.append((x - t//2, margin, t, gap_y - margin))
        # Bottom segment
        walls3.append((x - t//2, gap_y + gap_size, t, H - margin - gap_y - gap_size))
    # Horizontal cross-bars with small gaps
    for row in range(1, 5):
        y = row * (H // 5)
        rng = random.Random(row * 31)
        gap_x = rng.randint(margin + 60, W - margin - gap_size - 60)
        walls3.append((margin, y - t//2, gap_x - margin, t))
        walls3.append((gap_x + gap_size, y - t//2, W - margin - gap_x - gap_size, t))
    levels.append(Level(
        walls=walls3,
        start=(margin + 20, H - margin - 20),
        end=(W - margin - 20, margin + 20),
        name="Hard-3: Cross-Hatch",
    ))

    return levels


# ─────────────────────────────────────────────────────────────────────────────
#  MazeGenerator — selects and returns levels by difficulty
# ─────────────────────────────────────────────────────────────────────────────

class MazeGenerator:
    """
    Holds the library of pre-designed levels and supplies them on demand.

    Usage:
        gen = MazeGenerator()
        level = gen.get_level(difficulty=1, index=0)
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
        W, H = canvas_w, canvas_h
        self._levels = {
            1: _easy_levels(W, H),
            2: _medium_levels(W, H),
            3: _hard_levels(W, H),
        }
        self._current_indices = {1: 0, 2: 0, 3: 0}

    def get_level(self, difficulty: int, index: int = 0) -> Level:
        """
        Return a specific level from the library.

        Args:
            difficulty: 1=Easy, 2=Medium, 3=Hard
            index:      level index within the difficulty (wraps around)
        """
        pool = self._levels.get(difficulty, self._levels[1])
        return pool[index % len(pool)]

    def next_level(self, difficulty: int) -> Level:
        """Advance to the next level in the difficulty pool and return it."""
        pool = self._levels.get(difficulty, self._levels[1])
        idx = self._current_indices[difficulty]
        level = pool[idx % len(pool)]
        self._current_indices[difficulty] = (idx + 1) % len(pool)
        return level

    def level_count(self, difficulty: int) -> int:
        return len(self._levels.get(difficulty, []))
