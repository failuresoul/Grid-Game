"""
game/levels.py — Level Data Definitions

Defines the Level dataclass and all handcrafted level layouts.
Levels are NOT grid-based: the player moves freely in 2-D pixel space.
Obstacles are axis-aligned rectangles (x, y, w, h).

All geometry constants (zone radii, wall thickness) are read from config
so they can be tuned without touching level logic.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import config

# Axis-aligned rectangle: (left, top, width, height) -- all in pixels
Rect = Tuple[int, int, int, int]


@dataclass
class Level:
    """
    Describes one maze level.

    Attributes:
        walls:   List of obstacle rectangles (x, y, w, h).
        start:   Centre of the start zone (cx, cy).
        end:     Centre of the end/goal zone (cx, cy).
        start_r: Radius of the start zone circle (px).
        end_r:   Radius of the end zone circle (px).
        name:    Human-readable level name shown in the HUD.
    """
    walls:   List[Rect]
    start:   Tuple[int, int]
    end:     Tuple[int, int]
    start_r: int = config.START_ZONE_RADIUS  # read from config, default 24
    end_r:   int = config.END_ZONE_RADIUS    # read from config, default 24
    name:    str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Level factory functions
#  Each takes canvas dimensions (W, H) so levels scale to any canvas size.
# ─────────────────────────────────────────────────────────────────────────────

def easy_levels(W: int, H: int) -> List[Level]:
    """
    Easy: Very open space, 1–3 gentle obstacles.
    Corridors ≥ 120 px.  Designed for severe motor impairment.
    """
    margin = 60
    levels: List[Level] = []

    # ── Easy-1: Three Rooms ───────────────────────────────────────────────────
    levels.append(Level(
        walls=[
            (margin,       H // 3,          W // 2 - 80,            22),
            (W // 2 + 80,  H * 2 // 3,      W // 2 - 80 - margin,   22),
            (W // 2 - 11,  H // 3 + 22,     22,                      H // 3 - 22),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-1: Three Rooms",
    ))

    # ── Easy-2: S-Curve ───────────────────────────────────────────────────────
    levels.append(Level(
        walls=[
            (margin,        H // 3,        W * 2 // 3,              22),
            (W // 3,        H * 2 // 3,    W * 2 // 3 - margin,     22),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-2: S-Curve",
    ))

    # ── Easy-3: Island ────────────────────────────────────────────────────────
    levels.append(Level(
        walls=[
            (W // 2 - 80, H // 2 - 60, 160, 120),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Easy-3: Island",
    ))

    return levels


def medium_levels(W: int, H: int) -> List[Level]:
    """
    Medium: Multiple turns and bends.  Corridors ≥ 70 px.
    """
    margin = 60
    t = config.WALL_THICKNESS  # wall thickness from config
    levels: List[Level] = []

    # ── Medium-1: Zigzag ──────────────────────────────────────────────────────
    levels.append(Level(
        walls=[
            (margin,        H // 5,        W - margin - 130, t),
            (margin + 130,  H * 2 // 5,    W - margin - 130, t),
            (margin,        H * 3 // 5,    W - margin - 130, t),
            (margin + 130,  H * 4 // 5,    W - margin - 130, t),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Medium-1: Zigzag",
    ))

    # ── Medium-2: Spiral ──────────────────────────────────────────────────────
    levels.append(Level(
        walls=[
            (margin,            margin + 80,        W - margin - 120,    t),
            (W - margin - t,    margin,             t,                   H - margin - 120),
            (margin + 120,      margin + 200,       W - margin - 240,    t),
            (margin + 120,      margin + 200,       t,                   H - margin - 320),
            (margin + 120,      H - margin - 180,   W - margin - 280,    t),
        ],
        start=(margin + 30, H - margin - 20),
        end=(W // 2, H // 2),
        name="Medium-2: Spiral",
    ))

    # ── Medium-3: Pillars ─────────────────────────────────────────────────────
    pillar_size = 70
    gap = 140
    pillars: List[Rect] = []
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


def hard_levels(W: int, H: int) -> List[Level]:
    """
    Hard: Many obstacles, narrow passages (≥ 40 px), dead ends.
    """
    margin = 50
    t = config.WALL_THICKNESS  # wall thickness from config
    levels: List[Level] = []

    # ── Hard-1: Dense Zigzag with dead-end stubs ──────────────────────────────
    walls: List[Rect] = []
    for i, y in enumerate([H // 6, H * 2 // 6, H * 3 // 6, H * 4 // 6, H * 5 // 6]):
        if i % 2 == 0:
            walls.append((margin,      y,      W - margin - 80, t))
            walls.append((margin,      y + t,  t,               55))
        else:
            walls.append((margin + 80, y,      W - margin - 80, t))
            walls.append((W - margin - t, y + t, t,             55))
    levels.append(Level(
        walls=walls,
        start=(margin + 30, H - margin - 20),
        end=(W - margin - 30, margin + 20),
        name="Hard-1: Dense Zigzag",
    ))

    # ── Hard-2: Narrow Corridors ──────────────────────────────────────────────
    walls2: List[Rect] = []
    seg_h = (H - 2 * margin) // 8
    for i in range(8):
        y = margin + i * seg_h
        if i % 2 == 0:
            walls2.append((margin,      y, W - margin - 55, t))
        else:
            walls2.append((margin + 55, y, W - margin - 55, t))
    levels.append(Level(
        walls=walls2,
        start=(margin + 20, H - margin - 20),
        end=(W - margin - 20, margin + 20),
        name="Hard-2: Narrow Corridors",
    ))

    # ── Hard-3: Cross-Hatch with random small gaps ────────────────────────────
    walls3: List[Rect] = []
    gap_size = 55
    for col in range(1, 7):
        x = col * (W // 7)
        rng = random.Random(col * 17)
        gap_y = rng.randint(margin + 60, H - margin - gap_size - 60)
        walls3.append((x - t // 2, margin,              t, gap_y - margin))
        walls3.append((x - t // 2, gap_y + gap_size,    t, H - margin - gap_y - gap_size))
    for row in range(1, 5):
        y = row * (H // 5)
        rng = random.Random(row * 31)
        gap_x = rng.randint(margin + 60, W - margin - gap_size - 60)
        walls3.append((margin,             y - t // 2, gap_x - margin,                   t))
        walls3.append((gap_x + gap_size,   y - t // 2, W - margin - gap_x - gap_size,    t))
    levels.append(Level(
        walls=walls3,
        start=(margin + 20, H - margin - 20),
        end=(W - margin - 20, margin + 20),
        name="Hard-3: Cross-Hatch",
    ))

    return levels
