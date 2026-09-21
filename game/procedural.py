"""
game/procedural.py — Player-Radius-Aware Procedural Continuous 2D Maze Generator

Generates continuous 2D mazes with guaranteed solvability for upper-limb
stroke rehabilitation:
  - Random seed support with 100% reproducibility.
  - Start and End generation with guaranteed obstacle-free margins.
  - Geometric obstacle placement (rectangles and polygons).
  - Player-radius-aware path validation (A* on continuous clearance grid).
  - Minimum path calculation with line-of-sight waypoint smoothing.
  - Automatic regeneration if a candidate maze is unsolvable or too simple.
"""

from __future__ import annotations
import heapq
import logging
import math
import random
from typing import List, Tuple, Optional, Any, Sequence

import numpy as np

import config
from game.levels import Level, MazeBuilder, RectObstacle, PolygonObstacle
from game.collision import resolve_against_obstacles

log = logging.getLogger(__name__)


def line_of_sight(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: Sequence[Any],
    player_radius: float,
    step_size: float = 8.0,
) -> bool:
    """
    Return True if a circular cursor with player_radius can travel in a straight line
    from p1 to p2 without colliding with any obstacle.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return True

    steps = max(1, int(dist // step_size))
    for s in range(1, steps):
        cx = p1[0] + dx * (s / steps)
        cy = p1[1] + dy * (s / steps)
        _, _, hit = resolve_against_obstacles(cx, cy, player_radius, obstacles)
        if hit:
            return False
    return True


def smooth_path_waypoints(
    raw_path: List[Tuple[float, float]],
    obstacles: Sequence[Any],
    player_radius: float,
) -> List[Tuple[float, float]]:
    """
    Simplify a dense grid path into minimal analytical waypoints using
    greedy line-of-sight shortcutting.
    """
    if len(raw_path) <= 2:
        return list(raw_path)

    smoothed = [raw_path[0]]
    current_idx = 0
    n = len(raw_path)

    while current_idx < n - 1:
        furthest = current_idx + 1
        # Look ahead as far as possible
        for next_idx in range(n - 1, current_idx, -1):
            if line_of_sight(raw_path[current_idx], raw_path[next_idx], obstacles, player_radius):
                furthest = next_idx
                break
        smoothed.append(raw_path[furthest])
        current_idx = furthest

    return smoothed


class ProceduralMazeGenerator:
    """
    Procedural 2D continuous maze generator with guaranteed physical solvability.
    """

    def __init__(
        self,
        canvas_w: int = config.CANVAS_WIDTH,
        canvas_h: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

    def generate(
        self,
        difficulty: int = config.DEFAULT_DIFFICULTY,
        seed: Optional[int] = None,
        max_attempts: int = config.MAZE_MAX_REGEN_ATTEMPTS,
    ) -> Level:
        """
        Generate a validated, physically solvable procedural maze.
        If a generated candidate is unsolvable or invalid, automatically
        regenerates with a new seed until valid.

        Args:
            difficulty: 1 = Easy, 2 = Medium, 3 = Hard
            seed:       Optional integer seed for reproducibility (None = randomized).
            max_attempts: Maximum regeneration attempts.

        Returns:
            Fully populated, guaranteed solvable Level instance.
        """
        initial_seed = seed if seed is not None else random.randint(1000, 999999)
        current_seed = initial_seed

        diff_cfg = config.DIFFICULTIES.get(difficulty, config.EASY)
        player_radius = float(diff_cfg.player_radius)

        for attempt in range(1, max_attempts + 1):
            level = self._try_generate_candidate(difficulty, current_seed, player_radius)
            if level is not None:
                if attempt > 1:
                    log.info(f"Procedural maze generated after {attempt} attempts (Seed: {current_seed})")
                return level

            # Advance seed for next automatic regeneration attempt
            current_seed = (current_seed + 1013904223) & 0x7FFFFFFF

        log.warning(f"Could not generate procedural maze after {max_attempts} attempts; using deterministic fallback.")
        from game.levels import create_easy_level, create_medium_level, create_hard_level
        if difficulty == 1:
            lvl = create_easy_level(self.canvas_w, self.canvas_h)
        elif difficulty == 2:
            lvl = create_medium_level(self.canvas_w, self.canvas_h)
        else:
            lvl = create_hard_level(self.canvas_w, self.canvas_h)
        lvl.seed = initial_seed
        return lvl

    def _try_generate_candidate(
        self,
        difficulty: int,
        seed: int,
        player_radius: float,
    ) -> Optional[Level]:
        """
        Attempt to generate one candidate maze and validate its solvability.
        Returns the validated Level if successful, or None if invalid.
        """
        rng = random.Random(seed)
        W, H = self.canvas_w, self.canvas_h
        margin = 60
        t = config.WALL_THICKNESS

        diff_name = {1: "EASY", 2: "MEDIUM", 3: "HARD"}.get(difficulty, "EASY")
        builder = MazeBuilder(f"Procedural {diff_name.capitalize()} (Seed: {seed})", width=W, height=H)
        builder.set_difficulty_tag(diff_name)
        builder.set_seed(seed)
        builder.set_description(f"Procedurally generated continuous {diff_name.lower()} maze (Seed: {seed}).")

        # ── 1. Target Radii & Start/End Zones ──────────────────────────────────
        if difficulty == 1:
            target_r = 36
            start_pos = (margin + 50, H - margin - 50)
            end_pos   = (W - margin - 50, margin + 50)
            obs_count = rng.randint(3, 5)
            min_path_target = 800.0
        elif difficulty == 2:
            target_r = 28
            quadrant = rng.randint(0, 1)
            if quadrant == 0:
                start_pos = (margin + 40, H - margin - 40)
                end_pos   = (W - margin - 40, margin + 40)
            else:
                start_pos = (margin + 40, margin + 40)
                end_pos   = (W - margin - 40, H - margin - 40)
            obs_count = rng.randint(6, 9)
            min_path_target = 1300.0
        else:
            target_r = 22
            quadrant = rng.randint(0, 1)
            if quadrant == 0:
                start_pos = (margin + 30, H - margin - 30)
                end_pos   = (W - margin - 30, margin + 30)
            else:
                start_pos = (margin + 30, margin + 30)
                end_pos   = (W - margin - 30, H - margin - 30)
            obs_count = rng.randint(10, 14)
            min_path_target = 2200.0

        builder.set_start(start_pos[0], start_pos[1], r=target_r)
        builder.set_end(end_pos[0], end_pos[1], r=target_r)
        sx, sy = builder.start
        ex, ey = builder.end

        # ── 2. Obstacle Generation ─────────────────────────────────────────────
        zone_clearance = target_r + player_radius + 35.0

        placed = 0
        attempts = 0
        while placed < obs_count and attempts < 150:
            attempts += 1
            is_poly = (difficulty > 1) and (rng.random() < 0.28)

            if is_poly:
                r_poly = rng.randint(28, 48)
                cx = rng.randint(margin + 70, W - margin - 70)
                cy = rng.randint(margin + 70, H - margin - 70)

                if math.hypot(cx - sx, cy - sy) < zone_clearance + r_poly:
                    continue
                if math.hypot(cx - ex, cy - ey) < zone_clearance + r_poly:
                    continue

                builder.add_polygon([
                    (cx, cy - r_poly),
                    (cx + r_poly, cy),
                    (cx, cy + r_poly),
                    (cx - r_poly, cy),
                ])
                placed += 1
            else:
                orientation = rng.choice(["h", "v"])
                if orientation == "v":
                    ow = float(t)
                    oh = float(rng.randint(int(H * 0.28), int(H * 0.58)))
                    ox = float(rng.randint(margin + 60, W - margin - 60))
                    v_align = rng.choice(["top", "bottom", "mid"])
                    if v_align == "top":
                        oy = 0.0
                    elif v_align == "bottom":
                        oy = float(H) - oh
                    else:
                        oy = float(rng.randint(margin, H - margin - int(oh)))
                else:
                    ow = float(rng.randint(int(W * 0.20), int(W * 0.45)))
                    oh = float(t)
                    ox = float(rng.randint(margin + 40, W - margin - int(ow) - 40))
                    oy = float(rng.randint(margin + 40, H - margin - 40))

                nx_s = max(ox, min(sx, ox + ow))
                ny_s = max(oy, min(sy, oy + oh))
                if math.hypot(sx - nx_s, sy - ny_s) < zone_clearance:
                    continue

                nx_e = max(ox, min(ex, ox + ow))
                ny_e = max(oy, min(ey, oy + oh))
                if math.hypot(ex - nx_e, ey - ny_e) < zone_clearance:
                    continue

                builder.add_rect(ox, oy, ow, oh)
                placed += 1

        # ── 3. Player-Radius-Aware Path Validation (Continuous Clearance Grid) ──
        grid_res = 12
        cols = int(W // grid_res)
        rows = int(H // grid_res)
        obstacles = builder.obstacles

        passable = np.zeros((rows, cols), dtype=bool)
        test_r = player_radius + 4.0

        for r in range(rows):
            cy = r * grid_res + grid_res / 2.0
            if cy - test_r < 0 or cy + test_r > H:
                continue
            for c in range(cols):
                cx = c * grid_res + grid_res / 2.0
                if cx - test_r < 0 or cx + test_r > W:
                    continue
                _, _, hit = resolve_against_obstacles(cx, cy, test_r, obstacles)
                if not hit:
                    passable[r, c] = True

        start_c = int(sx // grid_res)
        start_r = int(sy // grid_res)
        end_c = int(ex // grid_res)
        end_r = int(ey // grid_res)

        if not (0 <= start_r < rows and 0 <= start_c < cols and passable[start_r, start_c]):
            return None
        if not (0 <= end_r < rows and 0 <= end_c < cols and passable[end_r, end_c]):
            return None

        # ── 4. A* Search on Traversable Clearance Grid ─────────────────────────
        def h_cost(r: int, c: int) -> float:
            return math.hypot((r - end_r) * grid_res, (c - end_c) * grid_res)

        frontier = []
        heapq.heappush(frontier, (h_cost(start_r, start_c), 0.0, start_r, start_c))
        came_from = {}
        cost_so_far = {(start_r, start_c): 0.0}

        found = False
        while frontier:
            _, g, cr, cc = heapq.heappop(frontier)
            if cr == end_r and cc == end_c:
                found = True
                break

            for dr, dc, step_cost in [
                (-1, 0, grid_res), (1, 0, grid_res),
                (0, -1, grid_res), (0, 1, grid_res),
                (-1, -1, grid_res * 1.414), (-1, 1, grid_res * 1.414),
                (1, -1, grid_res * 1.414), (1, 1, grid_res * 1.414),
            ]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc]:
                    if dr != 0 and dc != 0:
                        if not (passable[cr, nc] and passable[nr, cc]):
                            continue
                    new_cost = g + step_cost
                    if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                        cost_so_far[(nr, nc)] = new_cost
                        priority = new_cost + h_cost(nr, nc)
                        heapq.heappush(frontier, (priority, new_cost, nr, nc))
                        came_from[(nr, nc)] = (cr, cc)

        if not found:
            return None

        # Reconstruct grid path
        curr = (end_r, end_c)
        raw_path = []
        while curr in came_from:
            raw_path.append((curr[1] * grid_res + grid_res / 2.0, curr[0] * grid_res + grid_res / 2.0))
            curr = came_from[curr]
        raw_path.append((float(sx), float(sy)))
        raw_path.reverse()
        raw_path.append((float(ex), float(ey)))

        # ── 5. Line-of-Sight Waypoint Smoothing & Minimum Path Calculation ──────
        smoothed_waypoints = smooth_path_waypoints(raw_path, obstacles, player_radius + 2.0)
        builder.set_optimal_path(smoothed_waypoints)

        if builder.min_path_distance < min_path_target:
            return None

        return builder.build()


_generator = ProceduralMazeGenerator()


def generate_procedural_maze(
    difficulty: int = config.DEFAULT_DIFFICULTY,
    seed: Optional[int] = None,
    width: int = config.CANVAS_WIDTH,
    height: int = config.CANVAS_HEIGHT,
) -> Level:
    """Convenience functional interface to generate a validated procedural maze."""
    gen = ProceduralMazeGenerator(canvas_w=width, canvas_h=height)
    return gen.generate(difficulty=difficulty, seed=seed)
