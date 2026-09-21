"""
scratch/render_game_states.py

Render visual verification frames for each state in the state machine:
  1. MENU
  2. LEVEL_SELECT
  3. READY
  4. PLAYING
  5. RESULTS (with restart prompt and clinical metrics breakdown)
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from game.maze import MazeGenerator
from game.game_engine import GameEngine, GameState
from metrics.performance import MetricsCollector
from ui.renderer import Renderer


def main():
    artifact_dir = r"C:\Users\USER\.gemini\antigravity-ide\brain\8a8dc661-aede-4c34-a6d5-0b9f7f8df645"
    os.makedirs(artifact_dir, exist_ok=True)

    maze_gen = MazeGenerator()
    renderer = Renderer()
    diff_cfg = config.DIFFICULTIES[1]  # Medium
    level = maze_gen.get_level(1, 0)

    # 1. Render MENU state
    canvas_menu = renderer.draw_start_screen(selected_difficulty=1)
    menu_path = os.path.join(artifact_dir, "state_menu_render.png")
    cv2.imwrite(menu_path, canvas_menu)
    print(f"Saved: {menu_path}")

    # 2. Render LEVEL_SELECT state
    canvas_level_select = renderer.draw_level_select_screen(
        difficulty_name=diff_cfg.name,
        level_index=0,
        total_levels=maze_gen.level_count(1),
        level_name=level.name,
        min_path_distance=level.minimum_path_distance,
        wall_count=len(level.walls),
    )
    lvl_select_path = os.path.join(artifact_dir, "state_level_select_render.png")
    cv2.imwrite(lvl_select_path, canvas_level_select)
    print(f"Saved: {lvl_select_path}")

    # 3. Render READY state (waiting at START, timer: 0.0s)
    collector = MetricsCollector(
        start=level.start,
        end=level.end,
        min_path_distance=level.min_path_distance,
        optimal_waypoints=level.optimal_waypoints,
    )
    engine = GameEngine(level=level, difficulty_cfg=diff_cfg, metrics_collector=collector)
    canvas_ready = renderer.draw(engine, dt=0.016)
    ready_path = os.path.join(artifact_dir, "state_ready_render.png")
    cv2.imwrite(ready_path, canvas_ready)
    print(f"Saved: {ready_path}")

    # 4. Render PLAYING state (moving along path, timer active, trail visible)
    engine.start_playing()
    engine._session_start = time.perf_counter() - 4.25  # realistic 4.25s session
    # Simulate movement along waypoints
    wps = level.optimal_waypoints
    for pt in wps[:len(wps)//2]:
        engine.update(cursor=pt, dt=0.033)
    canvas_playing = renderer.draw(engine, dt=0.016)
    playing_path = os.path.join(artifact_dir, "state_playing_render.png")
    cv2.imwrite(playing_path, canvas_playing)
    print(f"Saved: {playing_path}")

    # 5. Render RESULTS state (completed session, frozen timer, restart prompt)
    engine._session_start = time.perf_counter() - 7.85
    for pt in wps:
        engine.update(cursor=pt, dt=0.033)
    engine.complete_session(won=True)
    canvas_results = renderer.draw(engine, dt=0.016)
    results_path = os.path.join(artifact_dir, "state_results_render.png")
    cv2.imwrite(results_path, canvas_results)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
