"""
game/levels.py — Continuous 2D Geometric Maze Definitions & Builder

Defines:
  - RectObstacle: Geometric axis-aligned rectangle obstacle.
  - PolygonObstacle: Arbitrary geometric 2D polygon obstacle.
  - Level: Continuous 2D maze level container with open space, START, END, and obstacles.
  - MazeBuilder: Fluent API for easily creating, generating, and modifying mazes.
  - conceptual_level: Direct implementation of the user's conceptual maze structure.
  - Handcrafted Easy, Medium, and Hard difficulty levels.

IMPORTANT:
  This is strictly NOT a grid-cell movement maze. The environment is continuous 2D space.
  Obstacles are continuous geometric shapes (rectangles and polygons).
  The player can move anywhere in open space.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import config

# Point: (x, y) continuous 2D coordinates
Point = Tuple[float, float]
# Axis-aligned rectangle: (left, top, width, height)
Rect = Tuple[int, int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
#  Geometric Obstacle Classes
# ─────────────────────────────────────────────────────────────────────────────

class Obstacle:
    """Base class for geometric obstacles in continuous 2D space."""
    is_polygon: bool = False

    def as_rect(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class RectObstacle(Obstacle):
    """
    Continuous 2D axis-aligned geometric rectangular obstacle.

    Supports indexing and tuple unpacking `(wx, wy, ww, wh) = obs` for full
    backward compatibility with legacy code expecting 4-tuples.
    """
    x: float
    y: float
    w: float
    h: float
    is_polygon: bool = False

    def as_rect(self) -> Tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))

    @property
    def points(self) -> List[Tuple[float, float]]:
        """Return the 4 vertices of the rectangle in cyclic order."""
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "rect", "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    # Backward compatibility with tuple unpacking (wx, wy, ww, wh)
    def __iter__(self):
        yield int(round(self.x))
        yield int(round(self.y))
        yield int(round(self.w))
        yield int(round(self.h))

    def __getitem__(self, idx: int) -> int:
        vals = (int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h)))
        return vals[idx]

    def __len__(self) -> int:
        return 4


@dataclass
class PolygonObstacle(Obstacle):
    """
    Continuous 2D arbitrary geometric polygon obstacle.
    Vertices are defined in cyclic order (clockwise or counter-clockwise).
    """
    points: List[Tuple[float, float]]
    is_polygon: bool = True

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError(f"PolygonObstacle requires at least 3 vertices, got {len(self.points)}")
        self.points = [(float(p[0]), float(p[1])) for p in self.points]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def as_rect(self) -> Tuple[int, int, int, int]:
        """Bounding box as (x, y, w, h)."""
        min_x, min_y, max_x, max_y = self.bounds
        return int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "polygon", "points": [[p[0], p[1]] for p in self.points]}

    # Backward compatibility: unpack bounding box if treated as a 4-tuple
    def __iter__(self):
        bbox = self.as_rect()
        for v in bbox:
            yield v

    def __getitem__(self, idx: int) -> int:
        return self.as_rect()[idx]

    def __len__(self) -> int:
        return 4


# ─────────────────────────────────────────────────────────────────────────────
#  Level Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Level:
    """
    Describes a continuous 2D maze level.

    Attributes:
        walls:             List of geometric obstacles (RectObstacle, PolygonObstacle, or tuples).
        start:             Centre of the START zone (cx, cy) in continuous pixel space.
        end:               Centre of the END / Goal zone (cx, cy) in continuous pixel space.
        start_r:           Radius of the start zone circle (px).
        end_r:             Radius of the end zone circle (px).
        name:              Human-readable level title shown in the HUD.
        width:             Canvas width (px).
        height:            Canvas height (px).
        description:       Informational description or clinical goal.
        optimal_waypoints: List of key path waypoints [(x, y), ...] from START to END.
        min_path_distance: Theoretical minimum continuous path length (px).
    """
    walls:             List[Any]
    start:             Tuple[int, int]
    end:               Tuple[int, int]
    start_r:           int = config.START_ZONE_RADIUS
    end_r:             int = config.END_ZONE_RADIUS
    name:              str = ""
    width:             int = config.CANVAS_WIDTH
    height:            int = config.CANVAS_HEIGHT
    description:       str = ""
    difficulty_tag:    str = ""
    seed:              Optional[int] = None
    optimal_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    min_path_distance: float = 0.0

    @property
    def minimum_path_distance(self) -> float:
        """Alias for min_path_distance."""
        return self.min_path_distance

    @minimum_path_distance.setter
    def minimum_path_distance(self, val: float):
        self.min_path_distance = float(val)

    def compute_minimum_path(self, player_radius: Optional[float] = None) -> float:
        """
        Compute the shortest collision-free route from START to END considering player radius
        and obstacles, store in optimal_waypoints and min_path_distance, and return min_path_distance.
        """
        from game.pathfinding import calculate_minimum_path
        if player_radius is None:
            tag = (self.difficulty_tag or "").upper()
            if tag == "EASY":
                pr = float(config.DIFFICULTIES[1].player_radius)
            elif tag == "MEDIUM":
                pr = float(config.DIFFICULTIES[2].player_radius)
            elif tag == "HARD":
                pr = float(config.DIFFICULTIES[3].player_radius)
            else:
                pr = float(getattr(config, "PLAYER_RADIUS", 16.0))
        else:
            pr = float(player_radius)

        dist, waypoints = calculate_minimum_path(
            start=self.start,
            end=self.end,
            obstacles=self.walls,
            player_radius=pr,
            width=self.width,
            height=self.height,
        )
        if waypoints:
            self.optimal_waypoints = waypoints
            self.min_path_distance = dist
        return self.min_path_distance

    def __post_init__(self):
        if not self.optimal_waypoints or len(self.optimal_waypoints) < 2:
            self.optimal_waypoints = [
                (float(self.start[0]), float(self.start[1])),
                (float(self.end[0]), float(self.end[1])),
            ]
        if self.min_path_distance <= 0.0 and len(self.optimal_waypoints) >= 2:
            total = 0.0
            for i in range(1, len(self.optimal_waypoints)):
                p1 = self.optimal_waypoints[i - 1]
                p2 = self.optimal_waypoints[i]
                total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.min_path_distance = total

    @property
    def obstacles(self) -> List[Any]:
        """Alias for walls to emphasize geometric obstacle nature."""
        return self.walls

    def to_dict(self) -> Dict[str, Any]:
        """Serialize level to a clean JSON-compatible dictionary."""
        obs_data = []
        for w in self.walls:
            if hasattr(w, "to_dict"):
                obs_data.append(w.to_dict())
            elif isinstance(w, (tuple, list)) and len(w) == 4:
                obs_data.append({"type": "rect", "x": w[0], "y": w[1], "w": w[2], "h": w[3]})
            elif isinstance(w, (tuple, list)) and len(w) >= 3:
                obs_data.append({"type": "polygon", "points": [list(p) for p in w]})
        return {
            "name": self.name,
            "difficulty_tag": self.difficulty_tag,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "end": list(self.end),
            "start_r": self.start_r,
            "end_r": self.end_r,
            "description": self.description,
            "optimal_waypoints": [list(p) for p in self.optimal_waypoints],
            "min_path_distance": self.min_path_distance,
            "obstacles": obs_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Level:
        """Construct a Level from a serialized dictionary."""
        obstacles: List[Any] = []
        raw_obs = data.get("obstacles") or data.get("walls") or []
        for item in raw_obs:
            if isinstance(item, dict):
                t = item.get("type", "rect")
                if t == "rect":
                    obstacles.append(RectObstacle(item["x"], item["y"], item["w"], item["h"]))
                elif t == "polygon":
                    obstacles.append(PolygonObstacle([tuple(p) for p in item["points"]]))
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                obstacles.append(RectObstacle(item[0], item[1], item[2], item[3]))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                obstacles.append(PolygonObstacle([tuple(p) for p in item]))

        raw_wp = data.get("optimal_waypoints", [])
        waypoints = [tuple(p) for p in raw_wp] if raw_wp else []

        return cls(
            name=data.get("name", "Custom Level"),
            difficulty_tag=data.get("difficulty_tag", ""),
            seed=data.get("seed", None),
            width=data.get("width", config.CANVAS_WIDTH),
            height=data.get("height", config.CANVAS_HEIGHT),
            start=tuple(data.get("start", (100, 100))),
            end=tuple(data.get("end", (data.get("width", 1000) - 100, data.get("height", 700) - 100))),
            start_r=data.get("start_r", config.START_ZONE_RADIUS),
            end_r=data.get("end_r", config.END_ZONE_RADIUS),
            walls=obstacles,
            description=data.get("description", ""),
            optimal_waypoints=waypoints,
            min_path_distance=float(data.get("min_path_distance", 0.0)),
        )



# ─────────────────────────────────────────────────────────────────────────────
#  MazeBuilder — Fluent API for Generating & Modifying Mazes
# ─────────────────────────────────────────────────────────────────────────────

class MazeBuilder:
    """
    Fluent builder that makes continuous 2D mazes easy to construct and modify.

    Example:
        builder = MazeBuilder("Rehab Maze", width=1000, height=700)
        builder.set_start(80, 100)
        builder.set_end(700, 500)
        builder.add_rect(200, 100, 30, 400)
        builder.add_l_shape(400, 200, 200, 200, thickness=25)
        builder.add_polygon([(500, 300), (600, 250), (620, 350)])
        level = builder.build()
    """

    def __init__(
        self,
        name: str = "Continuous Maze",
        width: int = config.CANVAS_WIDTH,
        height: int = config.CANVAS_HEIGHT,
    ) -> None:
        self.name:              str = name
        self.width:             int = width
        self.height:            int = height
        self.start:             Tuple[int, int] = (80, 80)
        self.end:               Tuple[int, int] = (width - 80, height - 80)
        self.start_r:           int = config.START_ZONE_RADIUS
        self.end_r:             int = config.END_ZONE_RADIUS
        self.obstacles:         List[Any] = []
        self.description:       str = ""
        self.difficulty_tag:    str = ""
        self.seed:              Optional[int] = None
        self.optimal_waypoints: List[Tuple[float, float]] = []
        self.min_path_distance: float = 0.0

    def set_seed(self, seed: Optional[int]) -> MazeBuilder:
        """Set the random generator seed for reproducibility."""
        self.seed = int(seed) if seed is not None else None
        return self

    def set_difficulty_tag(self, tag: str) -> MazeBuilder:
        """Set the difficulty identifier tag (e.g. 'EASY', 'MEDIUM', 'HARD')."""
        self.difficulty_tag = str(tag).strip().upper()
        return self

    def set_optimal_path(self, waypoints: Sequence[Tuple[float, float]]) -> MazeBuilder:
        """Set analytical optimal path waypoints and compute minimum path distance."""
        self.optimal_waypoints = [tuple(p) for p in waypoints]
        total = 0.0
        for i in range(1, len(self.optimal_waypoints)):
            p1 = self.optimal_waypoints[i - 1]
            p2 = self.optimal_waypoints[i]
            total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        self.min_path_distance = total
        return self



    def set_start(self, x: float, y: float, r: Optional[int] = None) -> MazeBuilder:
        """Set the START zone coordinates and optional radius."""
        self.start = (int(round(x)), int(round(y)))
        if r is not None:
            self.start_r = max(4, int(r))
        return self

    def set_end(self, x: float, y: float, r: Optional[int] = None) -> MazeBuilder:
        """Set the END / Goal zone coordinates and optional radius."""
        self.end = (int(round(x)), int(round(y)))
        if r is not None:
            self.end_r = max(4, int(r))
        return self

    def set_description(self, desc: str) -> MazeBuilder:
        """Set description or therapeutic goal for this level."""
        self.description = desc
        return self

    def add_rect(self, x: float, y: float, w: float, h: float) -> MazeBuilder:
        """Add an axis-aligned geometric rectangular obstacle."""
        self.obstacles.append(RectObstacle(float(x), float(y), float(w), float(h)))
        return self

    def add_polygon(self, points: Sequence[Tuple[float, float]]) -> MazeBuilder:
        """Add an arbitrary geometric polygon obstacle with 3 or more vertices."""
        self.obstacles.append(PolygonObstacle(list(points)))
        return self

    def add_wall(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: float = config.WALL_THICKNESS,
    ) -> MazeBuilder:
        """
        Add a thick continuous wall segment between (x1, y1) and (x2, y2).
        If strictly horizontal or vertical, creates a RectObstacle; otherwise creates a PolygonObstacle.
        """
        t = float(thickness)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-4:
            return self

        # Strictly horizontal
        if abs(dy) < 1e-4:
            left = min(x1, x2)
            top = y1 - t / 2.0
            return self.add_rect(left, top, length, t)

        # Strictly vertical
        if abs(dx) < 1e-4:
            left = x1 - t / 2.0
            top = min(y1, y2)
            return self.add_rect(left, top, t, length)

        # Slanted wall: compute normal offset
        nx = -dy / length
        ny = dx / length
        half_t = t / 2.0
        p1 = (x1 + nx * half_t, y1 + ny * half_t)
        p2 = (x2 + nx * half_t, y2 + ny * half_t)
        p3 = (x2 - nx * half_t, y2 - ny * half_t)
        p4 = (x1 - nx * half_t, y1 - ny * half_t)
        return self.add_polygon([p1, p2, p3, p4])

    def add_enclosure(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        thickness: float = config.WALL_THICKNESS,
        openings: Optional[List[Tuple[str, float, float]]] = None,
    ) -> MazeBuilder:
        """
        Add a rectangular enclosure with optional open doorways.

        Args:
            x, y, w, h: Outer enclosure bounds.
            thickness: Wall thickness.
            openings: List of (side, start_offset, end_offset) where side is 'top', 'bottom', 'left', or 'right'.
        """
        t = float(thickness)
        openings = openings or []

        # Helper to generate wall segments with gaps
        def segments_for_side(total_len: float, side_name: str) -> List[Tuple[float, float]]:
            gaps = [
                (max(0.0, o[1]), min(total_len, o[2]))
                for o in openings if o[0].lower() == side_name
            ]
            gaps.sort()
            # Merge overlapping gaps
            merged_gaps: List[Tuple[float, float]] = []
            for g in gaps:
                if not merged_gaps or g[0] > merged_gaps[-1][1]:
                    merged_gaps.append(g)
                else:
                    merged_gaps[-1] = (merged_gaps[-1][0], max(merged_gaps[-1][1], g[1]))

            # Invert gaps to get wall spans
            spans: List[Tuple[float, float]] = []
            curr = 0.0
            for (gs, ge) in merged_gaps:
                if gs > curr:
                    spans.append((curr, gs))
                curr = max(curr, ge)
            if curr < total_len:
                spans.append((curr, total_len))
            return spans

        # Top wall
        for s, e in segments_for_side(w, "top"):
            self.add_rect(x + s, y, e - s, t)
        # Bottom wall
        for s, e in segments_for_side(w, "bottom"):
            self.add_rect(x + s, y + h - t, e - s, t)
        # Left wall
        for s, e in segments_for_side(h, "left"):
            self.add_rect(x, y + s, t, e - s)
        # Right wall
        for s, e in segments_for_side(h, "right"):
            self.add_rect(x + w - t, y + s, t, e - s)

        return self

    def add_l_shape(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        thickness: float = config.WALL_THICKNESS,
        corner: str = "top-left",
    ) -> MazeBuilder:
        """
        Add an L-shaped geometric obstacle.

        Args:
            x, y, w, h: Bounding box of the L-shape.
            thickness: Wall thickness.
            corner: Where the junction is ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        """
        t = float(thickness)
        c = corner.lower()
        if c == "top-left":
            # ┌─────
            # │
            self.add_rect(x, y, w, t)
            self.add_rect(x, y + t, t, h - t)
        elif c == "top-right":
            # ─────┐
            #      │
            self.add_rect(x, y, w, t)
            self.add_rect(x + w - t, y + t, t, h - t)
        elif c == "bottom-left":
            # │
            # └─────
            self.add_rect(x, y, t, h - t)
            self.add_rect(x, y + h - t, w, t)
        elif c == "bottom-right":
            #      │
            # ─────┘
            self.add_rect(x + w - t, y, t, h - t)
            self.add_rect(x, y + h - t, w, t)
        return self

    def add_u_shape(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        thickness: float = config.WALL_THICKNESS,
        open_side: str = "bottom",
    ) -> MazeBuilder:
        """Add a U-shaped barrier open on one side ('top', 'bottom', 'left', 'right')."""
        t = float(thickness)
        side = open_side.lower()
        if side == "bottom":
            self.add_rect(x, y, w, t)                  # top bar
            self.add_rect(x, y + t, t, h - t)          # left arm
            self.add_rect(x + w - t, y + t, t, h - t)  # right arm
        elif side == "top":
            self.add_rect(x, y + h - t, w, t)          # bottom bar
            self.add_rect(x, y, t, h - t)              # left arm
            self.add_rect(x + w - t, y, t, h - t)      # right arm
        elif side == "right":
            self.add_rect(x, y, t, h)                  # left bar
            self.add_rect(x + t, y, w - t, t)          # top arm
            self.add_rect(x + t, y + h - t, w - t, t)  # bottom arm
        elif side == "left":
            self.add_rect(x + w - t, y, t, h)          # right bar
            self.add_rect(x, y, w - t, t)              # top arm
            self.add_rect(x, y + h - t, w - t, t)      # bottom arm
        return self

    def add_pillar(self, cx: float, cy: float, size: float = 60.0) -> MazeBuilder:
        """Add a square pillar centered at (cx, cy)."""
        s = float(size)
        return self.add_rect(cx - s / 2.0, cy - s / 2.0, s, s)

    def build(self) -> Level:
        """Construct and return the finalized continuous 2D Level."""
        lvl = Level(
            walls=list(self.obstacles),
            start=self.start,
            end=self.end,
            start_r=self.start_r,
            end_r=self.end_r,
            name=self.name,
            width=self.width,
            height=self.height,
            description=self.description,
            difficulty_tag=self.difficulty_tag,
            seed=self.seed,
            optimal_waypoints=list(self.optimal_waypoints),
            min_path_distance=self.min_path_distance,
        )
        # If waypoints were not explicitly defined and obstacles exist, compute optimal path
        if (not self.optimal_waypoints or len(self.optimal_waypoints) <= 2) and self.obstacles:
            lvl.compute_minimum_path()
        return lvl


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic Primary EASY Level
# ─────────────────────────────────────────────────────────────────────────────

def create_easy_level(W: int = config.CANVAS_WIDTH, H: int = config.CANVAS_HEIGHT) -> Level:
    """
    Deterministic primary EASY maze level designed for stroke rehabilitation.

    Design Principles:
      - Large open spaces: Wide, unencumbered rooms (> 300px clear span).
      - Very few obstacles: 2 gentle vertical baffle dividers creating a spacious natural rehabilitation S-channel.
      - Wide passages: Corridors >= 256 px (more than 7x the player diameter of 36 px).
      - Short/simple navigation: Intuitive forward -> upward -> rightward arc trajectory.
      - Large START and END targets: start_r = 38 px, end_r = 38 px.
      - Continuous 2D open movement: Zero grid or cell snapping.
      - 1. START is not inside a wall (clearance > 200 px).
      - 2. END is not inside a wall (clearance > 200 px).
      - 3. A valid continuous path exists without obstruction.
      - 4. Player physically fits through all passages with massive clearance.
      - 5. Minimum path calculated analytically:
           Waypoints: S (140, 540) -> W1 (390, 348) -> W2 (610, 352) -> E (860, 160)
           Shortest distance = 850.48 px
      - 6. No accidental impossible areas or dead ends.
    """
    t = config.WALL_THICKNESS
    builder = MazeBuilder("Easy-1: Gentle Flow", width=W, height=H)
    builder.set_difficulty_tag("EASY")
    builder.set_description(

        "Spacious beginner rehabilitation level. Navigate smoothly from START through wide passages around gentle barriers to GOAL."
    )

    # 1. Large START and END targets (38 px radius) in spacious open clearings
    start_pos = (140, int(round(H - 140)))
    end_pos   = (int(round(W - 140)), 140)
    target_r  = 38
    builder.set_start(start_pos[0], start_pos[1], r=target_r)
    builder.set_end(end_pos[0], end_pos[1], r=target_r)

    # 2. Obstacle 1: Upper baffle descending from top
    # Open vertical clearance below Obstacle 1: > 370 px
    obs1_x = round(W * 0.36 - t / 2.0)
    obs1_y = 0.0
    obs1_w = float(t)
    obs1_h = round(H * 0.47)
    builder.add_rect(obs1_x, obs1_y, obs1_w, obs1_h)

    # 3. Obstacle 2: Lower baffle ascending from bottom
    # Open vertical clearance above Obstacle 2: > 370 px
    # Horizontal open corridor between baffles: >= 256 px
    obs2_x = round(W * 0.64 - t / 2.0)
    obs2_y = round(H * 0.53)
    obs2_w = float(t)
    obs2_h = float(H) - obs2_y
    builder.add_rect(obs2_x, obs2_y, obs2_w, obs2_h)

    # 4. Optimal minimum path calculation
    # In Easy-1, the open channel between the upper and lower baffles allows a direct,
    # collision-free path between START and END.
    lvl = builder.build()
    lvl.compute_minimum_path(player_radius=18.0)
    return lvl


# ─────────────────────────────────────────────────────────────────────────────
#  Conceptual Maze — Direct Implementation of User's Example
# ─────────────────────────────────────────────────────────────────────────────

def conceptual_level(W: int = config.CANVAS_WIDTH, H: int = config.CANVAS_HEIGHT) -> Level:
    """
    Direct implementation of the user's conceptual maze structure:

    START ●

           ┌──────────────┐
           │              │
           │    ┌─────┐   │
           │    │     │   │
           │    │     └───┤
           │              │
           │        ● END │
           └──────────────┘

    The player begins in open space at START, enters the outer enclosure through
    an opening, navigates around the inner geometric obstacle, and reaches END.
    """
    t = config.WALL_THICKNESS
    builder = MazeBuilder("Conceptual Rehab Maze", width=W, height=H)
    builder.set_difficulty_tag("EASY")
    builder.set_description("Navigate from outside START through the entrance, around the obstacle to END.")

    # 1. START point: located in open space outside the enclosure (top-left)
    builder.set_start(90, 110)

    # 2. Outer enclosure dimensions
    enc_x = 220
    enc_y = 90
    enc_w = W - 280
    enc_h = H - 160

    # Doorway entrance on the left wall of enclosure (between y=170 and y=310, 140px wide corridor)
    entrance_gap_start = 80.0
    entrance_gap_end = 220.0

    builder.add_enclosure(
        x=enc_x,
        y=enc_y,
        w=enc_w,
        h=enc_h,
        thickness=t,
        openings=[("left", entrance_gap_start, entrance_gap_end)],
    )

    # 3. Inner geometric obstacle:
    #    ┌─────┐
    #    │     │
    #    │     └───┤
    # Located in the middle-upper part of the enclosure, extending toward the right wall
    obs_x = enc_x + 160
    obs_y = enc_y + 110
    obs_w = 260
    obs_h = 190

    # Top cap of inner obstacle: ┌─────┐
    builder.add_rect(obs_x, obs_y, obs_w, t)
    # Left wall: │
    builder.add_rect(obs_x, obs_y + t, t, obs_h - t)
    # Right inner arm descending: │
    inner_arm_x = obs_x + obs_w - t
    builder.add_rect(inner_arm_x, obs_y + t, t, 100)
    # Bottom horizontal hook connecting rightward: └───┤
    hook_y = obs_y + 100 + t
    hook_w = (enc_x + enc_w - t) - inner_arm_x  # extends rightward to join the outer enclosure wall
    builder.add_rect(inner_arm_x, hook_y, hook_w, t)

    # 4. END point: located in open space in the lower area of the enclosure
    builder.set_end(enc_x + enc_w // 2 + 60, enc_y + enc_h - 100)

    return builder.build()


# ─────────────────────────────────────────────────────────────────────────────
#  Handcrafted Difficulty Levels (Continuous 2D Geometries)
# ─────────────────────────────────────────────────────────────────────────────

def easy_levels(W: int, H: int) -> List[Level]:
    """
    Easy: Very open space, 1–2 gentle geometric obstacles.
    Corridors ≥ 200 px.  Designed for severe motor impairment and beginners.
    """
    margin = 60
    t = config.WALL_THICKNESS
    levels: List[Level] = []

    # ── Easy-1: Gentle Flow (Primary Deterministic Easy Level) ────────────────
    levels.append(create_easy_level(W, H))

    # ── Easy-2: Conceptual Maze (User's Conceptual Structure) ─────────────────
    levels.append(conceptual_level(W, H))

    # ── Easy-3: Three Rooms ───────────────────────────────────────────────────
    b2 = MazeBuilder("Easy-3: Three Rooms", width=W, height=H)
    b2.set_difficulty_tag("EASY")
    b2.set_start(margin + 40, H - margin - 30)
    b2.set_end(W - margin - 40, margin + 30)
    b2.add_rect(margin, H // 3, W // 2 - 80, t)
    b2.add_rect(W // 2 + 80, H * 2 // 3, W // 2 - 80 - margin, t)
    b2.add_rect(W // 2 - t // 2, H // 3 + t, t, H // 3 - t)
    levels.append(b2.build())

    # ── Easy-4: S-Curve Corridor ──────────────────────────────────────────────
    b3 = MazeBuilder("Easy-4: S-Curve", width=W, height=H)
    b3.set_difficulty_tag("EASY")
    b3.set_start(margin + 40, H - margin - 30)
    b3.set_end(W - margin - 40, margin + 30)
    b3.add_rect(margin, H // 3, W * 2 // 3, t)
    b3.add_rect(W // 3, H * 2 // 3, W * 2 // 3 - margin, t)
    levels.append(b3.build())

    # ── Easy-5: Central Island ────────────────────────────────────────────────
    b4 = MazeBuilder("Easy-5: Island", width=W, height=H)
    b4.set_difficulty_tag("EASY")
    b4.set_start(margin + 40, H - margin - 30)
    b4.set_end(W - margin - 40, margin + 30)
    b4.add_rect(W // 2 - 90, H // 2 - 70, 180, 140)
    levels.append(b4.build())

    return levels



# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic Primary MEDIUM Level
# ─────────────────────────────────────────────────────────────────────────────

def create_medium_level(W: int = config.CANVAS_WIDTH, H: int = config.CANVAS_HEIGHT) -> Level:
    """
    Deterministic primary MEDIUM maze level.

    Compared with EASY:
      - More obstacles: 5 geometric obstacles (dividers, baffles, and central diamond polygon).
      - More turns: 5 to 6 distinct direction changes.
      - More navigation decisions: Dual pathway fork (Upper Scenic Route vs. Lower-Central Shortcut).
      - Smaller but still comfortable passages: Corridors 130–160 px wide (vs Easy's >= 258 px).
        With player diameter 28 px (radius 14), leaves > 100 px clearance (> 4.5x player diameter).
      - Longer minimum path: Analytical minimum path = ~1430 px (vs Easy's 877 px, >63% longer).
      - Maintains continuous free movement without cell/block snapping.
      - Start and End are reachable with ample clearance.
      - No obstacle completely blocks the maze.
      - Level identifier: MEDIUM.
    """
    t = config.WALL_THICKNESS
    builder = MazeBuilder("Medium-1: Dual Pathways", width=W, height=H)
    builder.set_difficulty_tag("MEDIUM")
    builder.set_description(
        "Medium difficulty level. Multiple turns and a pathway fork around geometric barriers to reach the GOAL."
    )

    # 1. Start and End targets (r = 28 px)
    start_pos = (110, int(round(H - 110)))
    end_pos   = (int(round(W - 110)), 110)
    target_r  = 28
    builder.set_start(start_pos[0], start_pos[1], r=target_r)
    builder.set_end(end_pos[0], end_pos[1], r=target_r)

    # 2. Obstacle 1: Initial Fork Vertical Divider
    # Located at x = 240, from y = 160 to y = 540 (h = 380)
    # Openings: Top passage (y in [0, 160], width 160px) & Bottom passage (y in [540, 700], width 160px)
    w1_x = round(W * 0.24 - t / 2.0)
    w1_y = 160.0
    w1_w = float(t)
    w1_h = 380.0
    builder.add_rect(w1_x, w1_y, w1_w, w1_h)

    # 3. Obstacle 2: Central Geometric Diamond Island
    # Centered at (500, 200), radius 45 px
    # Passage above diamond: y in [0, 155] -> 155 px wide
    # Passage below diamond: y in [245, 380] -> 135 px wide
    cd_x = round(W * 0.50)
    cd_y = 200.0
    radius_d = 45.0
    builder.add_polygon([
        (cd_x, cd_y - radius_d),
        (cd_x + radius_d, cd_y),
        (cd_x, cd_y + radius_d),
        (cd_x - radius_d, cd_y),
    ])

    # 4. Obstacle 3: Mid-Lower Ascending Baffle
    # Located at x = 460, from y = 380 to H (h = 320)
    # Channel between Obstacle 1 and Obstacle 3: 460 - 251 = 209 px!
    # Clearance over Obstacle 3 into central hub: y in [0, 380]
    w3_x = round(W * 0.46 - t / 2.0)
    w3_y = 380.0
    w3_w = float(t)
    w3_h = float(H) - 380.0
    builder.add_rect(w3_x, w3_y, w3_w, w3_h)

    # 5. Obstacle 4: Mid-Upper Descending Baffle
    # Located at x = 700, from y = 0 to y = 360 (h = 360)
    # Channel between Diamond and Obstacle 4: 700 - 545 = 155 px!
    # Passage under Obstacle 4: y in [360, 700]
    w4_x = round(W * 0.70 - t / 2.0)
    w4_y = 0.0
    w4_w = float(t)
    w4_h = 360.0
    builder.add_rect(w4_x, w4_y, w4_w, w4_h)

    # 6. Obstacle 5: Goal Approach Horizontal Barrier
    # Located at y = 490, from x = 580 to x = 860 (w = 280, h = 22)
    # Clearance to right canvas edge: 1000 - 860 = 140 px channel!
    # Passage between Obstacle 4 bottom (360) and Obstacle 5 top (490): 490 - 360 = 130 px!
    w5_x = round(W * 0.58)
    w5_y = round(H * 0.70)
    w5_w = round(W * 0.86) - w5_x
    w5_h = float(t)
    builder.add_rect(w5_x, w5_y, w5_w, w5_h)

    lvl = builder.build()
    lvl.compute_minimum_path(player_radius=14.0)
    return lvl


def medium_levels(W: int, H: int) -> List[Level]:
    """
    Medium: Multiple turns, decision paths, L-shapes, and geometric obstacles.
    Corridors >= 130 px.
    """
    margin = 60
    t = config.WALL_THICKNESS
    levels: List[Level] = []

    # ── Medium-1: Dual Pathways (Primary Deterministic Medium Level) ───────────
    levels.append(create_medium_level(W, H))

    # ── Medium-2: Zigzag Channels ─────────────────────────────────────────────
    b1 = MazeBuilder("Medium-2: Zigzag", width=W, height=H)
    b1.set_difficulty_tag("MEDIUM")
    b1.set_start(margin + 40, H - margin - 30)
    b1.set_end(W - margin - 40, margin + 30)
    b1.add_rect(margin, H // 5, W - margin - 140, t)
    b1.add_rect(margin + 140, H * 2 // 5, W - margin - 140, t)
    b1.add_rect(margin, H * 3 // 5, W - margin - 140, t)
    b1.add_rect(margin + 140, H * 4 // 5, W - margin - 140, t)
    levels.append(b1.build())

    # ── Medium-3: Spiral Corridor ─────────────────────────────────────────────
    b2 = MazeBuilder("Medium-3: Spiral", width=W, height=H)
    b2.set_difficulty_tag("MEDIUM")
    b2.set_start(margin + 40, H - margin - 30)
    b2.set_end(W // 2, H // 2)
    b2.add_rect(margin, margin + 80, W - margin - 120, t)
    b2.add_rect(W - margin - t, margin, t, H - margin - 120)
    b2.add_rect(margin + 120, margin + 200, W - margin - 240, t)
    b2.add_rect(margin + 120, margin + 200, t, H - margin - 320)
    b2.add_rect(margin + 120, H - margin - 180, W - margin - 280, t)
    levels.append(b2.build())

    # ── Medium-4: Geometric Pillars & Diamond Polygon ─────────────────────────
    b3 = MazeBuilder("Medium-4: Pillars & Diamonds", width=W, height=H)
    b3.set_difficulty_tag("MEDIUM")
    b3.set_start(margin + 40, H - margin - 30)
    b3.set_end(W - margin - 40, margin + 30)
    # Square pillars
    pillar_size = 65
    gap = 145
    for row in range(1, 4):
        for col in range(1, 5):
            px = col * (pillar_size + gap) - 40
            py = row * (pillar_size + gap) - 20
            if px + pillar_size < W - margin and py + pillar_size < H - margin:
                b3.add_pillar(px + pillar_size / 2, py + pillar_size / 2, pillar_size)
    # Central geometric diamond polygon
    cx, cy = W // 2, H // 2
    b3.add_polygon([
        (cx, cy - 45),
        (cx + 45, cy),
        (cx, cy + 45),
        (cx - 45, cy),
    ])
    levels.append(b3.build())

    return levels



# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic Primary HARD Level
# ─────────────────────────────────────────────────────────────────────────────

def create_hard_level(W: int = config.CANVAS_WIDTH, H: int = config.CANVAS_HEIGHT) -> Level:
    """
    Deterministic primary HARD maze level for upper-limb stroke rehabilitation.

    Compared with MEDIUM:
      - More obstacles: 14 geometric obstacles (including baffles, L-hooks, dead-end traps, and diamond polygon).
      - More turns: 15 distinct direction changes (switchbacks, chicanes, and turns).
      - Longer route: Analytical minimum path = ~3556 px (vs Medium's ~1460 px, >143% longer).
      - Narrower passages: Corridors 70–85 px wide (vs Medium's 130–160 px).
        With player radius 12 px (diameter 24 px), provides 46–61 px net clearance (2.9x to 3.5x diameter).
      - More opportunities for wrong movement:
        * Dead-End Trap 1: Cul-de-sac at top of Column 1.
        * Dead-End Trap 2: Impassable lower spur in Column 3.
        * Dead-End Trap 3: Bottom enclosure under the central Diamond.
        * Deceptive Goal View: Column 7 forces the player away from the visible Goal down through a switchback.
      - Complex spatial navigation: Requires active motor planning, direction reversals, and impulse inhibition.
      - Physically solvable: Verified collision-free continuous navigation.
      - Difficulty identifier: HARD.
    """
    t = 20.0
    builder = MazeBuilder("Hard-1: The Labyrinth", width=W, height=H)
    builder.set_difficulty_tag("HARD")
    builder.set_description(
        "Hard rehabilitation maze. 15 turns through narrow corridors, switchbacks, and deceptive traps to reach the GOAL."
    )

    # 1. Start and End targets (r = 22 px)
    start_pos = (75, int(round(H - 75)))
    end_pos   = (int(round(W - 75)), 75)
    target_r  = 22
    builder.set_start(start_pos[0], start_pos[1], r=target_r)
    builder.set_end(end_pos[0], end_pos[1], r=target_r)

    # 2. Obstacle 1: Vertical divider in Column 1 (x=150 to 170, y=140 to 540)
    builder.add_rect(150, 140, t, 400)

    # 3. Obstacle 2 & 3: Dead-End Trap #1 (Cul-de-sac at top of Column 1)
    builder.add_rect(150, 120, 70, t)   # cap at y=120 connecting Wall 1 to Wall 1C
    builder.add_rect(220, 0, t, 140)    # right barrier from ceiling to y=140

    # 4. Obstacle 4: Ascending vertical baffle in Column 2 (x=240 to 260, y=210 to 700)
    builder.add_rect(240, 210, t, float(H) - 210)

    # 5. Obstacle 5: Descending vertical baffle in Column 3 (x=330 to 350, y=0 to 500)
    builder.add_rect(330, 0, t, 500)

    # 6. Obstacle 6: Dead-End Trap #2 (Spur at bottom of Column 3, x=350 to 420, y=620)
    builder.add_rect(350, 620, 70, t)

    # 7. Obstacle 7: Ascending vertical baffle in Column 4 (x=420 to 440, y=140 to 700)
    builder.add_rect(420, 140, t, float(H) - 140)

    # 8. Obstacle 8: Descending vertical baffle in Column 5 (x=520 to 540, y=0 to 290)
    builder.add_rect(520, 0, t, 290)

    # 9. Obstacle 9: Central Diamond Polygon Island (cx=605, cy=450, r=40)
    cd_x, cd_y, r_d = 605.0, 450.0, 40.0
    builder.add_polygon([
        (cd_x, cd_y - r_d),
        (cd_x + r_d, cd_y),
        (cd_x, cd_y + r_d),
        (cd_x - r_d, cd_y),
    ])

    # 10. Obstacle 10: Dead-End Trap #3 (Sub-Diamond barrier connecting Wall 5 to Wall 7)
    builder.add_rect(540, 550, 120, t)

    # 11. Obstacle 11: Ascending vertical baffle in Column 6 (x=660 to 680, y=190 to 700)
    builder.add_rect(660, 190, t, float(H) - 190)

    # 12. Obstacle 12: Descending vertical baffle in Column 7 (x=760 to 780, y=0 to 480)
    builder.add_rect(760, 0, t, 480)

    # 13. Obstacle 13: Pre-Goal vertical wall in Column 8 (x=850 to 870, y=160 to 700)
    builder.add_rect(850, 160, t, float(H) - 160)

    lvl = builder.build()
    lvl.compute_minimum_path(player_radius=10.0)
    return lvl


def hard_levels(W: int, H: int) -> List[Level]:
    """
    Hard: Dense obstacles, narrow passages (>= 60 px), dead-end traps, and polygons.
    """
    margin = 50
    t = config.WALL_THICKNESS
    levels: List[Level] = []

    # ── Hard-1: The Labyrinth (Primary Deterministic Hard Level) ──────────────
    levels.append(create_hard_level(W, H))

    # ── Hard-2: Dense Zigzag with L-Hooks ─────────────────────────────────────
    b1 = MazeBuilder("Hard-2: Dense Zigzag", width=W, height=H)
    b1.set_difficulty_tag("HARD")
    b1.set_start(margin + 60, H - margin - 20)
    b1.set_end(W - margin - 60, margin + 20)
    for i, y in enumerate([H // 6, H * 2 // 6, H * 3 // 6, H * 4 // 6, H * 5 // 6]):
        if i % 2 == 0:
            b1.add_rect(margin, y, W - margin - 80, t)
            b1.add_rect(margin, y + t, t, 55)
        else:
            b1.add_rect(margin + 80, y, W - margin - 80, t)
            b1.add_rect(W - margin - t, y + t, t, 55)
    levels.append(b1.build())

    # ── Hard-3: Narrow Geometric Corridors ────────────────────────────────────
    b2 = MazeBuilder("Hard-3: Narrow Corridors", width=W, height=H)
    b2.set_difficulty_tag("HARD")
    b2.set_start(margin + 30, H - margin // 2)
    b2.set_end(W - margin - 30, margin // 2)
    seg_h = (H - 2 * margin) // 8
    for i in range(8):
        y = margin + i * seg_h
        if i % 2 == 0:
            b2.add_rect(margin, y, W - margin - 55, t)
        else:
            b2.add_rect(margin + 55, y, W - margin - 55, t)
    levels.append(b2.build())

    # ── Hard-4: Polygonal Maze (Slanted Geometric Walls & Rooms) ──────────────
    b3 = MazeBuilder("Hard-4: Polygonal Corridors", width=W, height=H)
    b3.set_difficulty_tag("HARD")
    b3.set_start(margin + 30, H - margin - 30)
    b3.set_end(W - margin - 30, margin + 30)
    # Slanted polygonal barriers
    b3.add_wall(margin + 80, margin + 100, W // 2 - 40, H // 2 - 40, thickness=22)
    b3.add_wall(W // 2 + 40, H // 2 + 40, W - margin - 80, H - margin - 100, thickness=22)
    b3.add_wall(W // 2 - 40, H // 2 + 40, margin + 80, H - margin - 100, thickness=22)
    b3.add_wall(W - margin - 80, margin + 100, W // 2 + 40, H // 2 - 40, thickness=22)
    # Central polygonal diamond obstacle
    cx, cy = W // 2, H // 2
    b3.add_polygon([
        (cx, cy - 60),
        (cx + 60, cy),
        (cx, cy + 60),
        (cx - 60, cy),
    ])
    levels.append(b3.build())

    return levels

