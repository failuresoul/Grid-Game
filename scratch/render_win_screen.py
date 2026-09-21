"""
scratch/render_win_screen.py — Render and export final win overlay image with Path Efficiency
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import config
from game.levels import create_easy_level
from game.game_engine import GameEngine, GameState
from ui.renderer import Renderer
from ui.screens import draw_win_overlay

def main():
    W, H = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
    level = create_easy_level()
    level.minimum_path_distance = 300.0

    renderer = Renderer()
    engine = GameEngine(level=level, difficulty_cfg=config.DIFFICULTIES[1])

    # Simulated final session metrics matching user requirements
    final_metrics = {
        "completion_time_s": 14.85,
        "trajectory_accuracy": 82.5,
        "mean_path_deviation_px": 24.6,
        "deviation_events": 2,
        "minimum_distance": 300.0,
        "actual_distance": 420.0,
        "path_efficiency": 71.4,
        "smoothness_score": 88.5,
        "time_outside_route_s": 1.84,
        "normalised_jerk": 8.42,
        "tremor_index": 0.038,
        "peak_speed_px_s": 185.0,
        "rom_width_px": 520.0,
        "rom_height_px": 310.0,
    }

    engine.state = GameState.WIN
    engine.final_metrics = final_metrics

    # Render frame
    canvas = renderer.draw(engine, dt=0.016)

    out_path = r"C:\Users\USER\.gemini\antigravity-ide\brain\8a8dc661-aede-4c34-a6d5-0b9f7f8df645\win_screen_smoothness_render.png"
    cv2.imwrite(out_path, canvas)
    print(f"Saved win screen render to {out_path}")

if __name__ == "__main__":
    main()
