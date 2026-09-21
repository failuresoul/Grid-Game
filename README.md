# Continuous 2D Hand-Controlled Maze Game for Stroke Rehabilitation

An interactive, continuous 2D motor rehabilitation system designed for stroke patients and upper-limb physical therapy. The patient controls a circular cursor in continuous physical coordinate space using optical hand tracking (webcam via OpenCV and MediaPipe), navigating through custom rehabilitation mazes with real-time collision detection and comprehensive clinical kinematics logging.

---

## Key Clinical & Architectural Principles

### 1. Continuous 2D Space (Zero Grid-Stepping)
Unlike discrete tile- or grid-based games, movement in this system is entirely continuous:
$$\text{START} \longrightarrow \text{Free Hand Movement} \longrightarrow \text{Continuous Obstacle Navigation} \longrightarrow \text{END}$$
- **Sub-pixel floating-point positions**: $(p_x, p_y)$ coordinates are maintained as continuous IEEE 754 floats.
- **Continuous Collision Detection (CCD)**: Movement sweeps between frames are subdivided into micro-steps ($< 4\text{ px}$), preventing tunneling or teleporting through thin walls even during rapid intentional movements.
- **Natural Wall-Sliding**: Glancing contacts preserve tangential momentum along wall surfaces rather than artificially snapping or stalling.

### 2. Clinical Metrics (Quality Over Raw Speed)
In stroke motor recovery, **faster movement does NOT indicate better rehabilitation**. Rushed movements frequently introduce compensatory spasticity, jerks, and collisions. The system computes and records:
- **Actual Distance**: Cumulative Euclidean path length $\sum \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}$.
- **Minimum Path Distance**: Optimal collision-free path calculated via obstacle visibility graph and A* search.
- **Path Efficiency**: Ratio percentage $\frac{\text{Minimum Distance}}{\text{Actual Distance}} \times 100\%$.
- **Trajectory Corridor Accuracy**: Adherence percentage and mean perpendicular deviation from the optimal corridor.
- **Movement Smoothness**: Normalized jerk score $[0 - 100]$ evaluating acceleration profile continuity.
- **Wall Collisions**: Number of boundary and obstacle impacts.
- **Adaptive Difficulty**: Clinical recommendation engine suggesting level promotion, maintenance, or gentle repetition based on 5 motor quality criteria.

---

## System Architecture

```
CURRENT ACTIVE PIPELINE (Camera-only hand tracking):
  ┌─────────────────────────┐
  │         CAMERA          │  (Standard USB / built-in webcam)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  OpenCV HAND TRACKING   │  (Index fingertip / wrist landmark detection)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │     PLAYER POSITION     │  (Smoothed continuous 2D coordinates: px, py)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │        MAZE GAME        │  (Continuous collision engine & state machine)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │   PERFORMANCE METRICS   │  (Accuracy, Efficiency, Smoothness, Collisions)
  └─────────────────────────┘

FUTURE INTEGRATION PIPELINE (Sensor Fusion):
  ┌─────────────────────────┐
  │       EMG SENSOR        │  (Surface electrodes: Biceps / Forearm / Extensors)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │     EMG PROCESSING      │  (Bandpass 20-450Hz, Notch 50/60Hz, Rectification)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ MUSCLE ACTIVATION DATA  │  (RMS, Peak, MVC Normalized Activation [0.0 - 1.0])
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │    PERFORMANCE DATA     │  (Correlate muscle effort vs kinematic smoothness)
  └─────────────────────────┘
```

---

## Installation & Requirements

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Standard USB or built-in webcam

### Installation
```bash
# 1. Clone or navigate to the repository directory
cd "Grid Game"

# 2. Install required dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)
- `opencv-python`: High-performance rendering, video capture, and UI overlays.
- `mediapipe`: Markerless 21-point hand skeleton tracking.
- `numpy`: Vectorized kinematics, signal filtering, and path geometry.

---

## Running the Game

### Exact Launch Command
```bash
python main.py
```

### Running Without EMG Hardware
- `EMG_ENABLED = False` is already configured by default in `config.py`.
- **Zero EMG hardware is required.** The game launches immediately using your webcam for continuous hand tracking.
- If no webcam is attached, the game automatically falls back to continuous mouse cursor control.
- When EMG is disabled, the UI explicitly displays:
  ```
  EMG: Not Connected / Disabled
  ```
- **Strict Anti-Fake-Measurement Guarantee**: The game never fabricates or presents fake synthetic sensor numbers (no fake RMS, peak, or activation measurements).

---

## Controls & Keybindings

| Key | Context | Action |
| :--- | :--- | :--- |
| **`ENTER` / `SPACE`** | Menu | Select difficulty / Proceed to Level Select |
| **`1` / `2` / `3`** | Menu / Level Select | Switch difficulty (`1: EASY`, `2: MEDIUM`, `3: HARD`) |
| **`H`** | Menu | View Progress & History screen (Session logs & trend charts) |
| **`P` / `N`** | History Screen | Previous / Next page of historical sessions |
| **`L`** | Any Screen | Cycle tracked hand landmark (Index Tip, Wrist, Thumb Tip) |
| **`D`** | Gameplay | Toggle real-time telemetry debug overlay (Raw vs. Smoothed coordinates) |
| **`M`** | Gameplay | Toggle mouse cursor fallback control |
| **`R`** | Gameplay / Results | Restart level to `READY` state (resets timer and trajectory) |
| **`P`** | Gameplay | Pause / Resume session |
| **`ESC`** | Any Screen | Exit / Return to Main Menu |

---

## Where Future EMG Hardware Will Connect

The codebase includes a fully isolated EMG hardware abstraction layer in `emg/emg_interface.py`. When physical electromyography sensors arrive:

1. **Enable in `config.py`**:
   ```python
   # Change from False to True:
   EMG_ENABLED = True

   # Configure physical connection:
   EMG_PORT = "COM3"           # or "/dev/ttyUSB0" on Linux
   EMG_DEVICE = "bitalino"     # Options: "bitalino", "myo", "custom_serial", "lsl"
   EMG_SAMPLE_RATE = 1000      # 1000 Hz
   EMG_CHANNELS = (0,)         # Active analog channel indices
   ```
2. **Device Drivers in `emg/emg_interface.py`**:
   - Uncomment the driver blocks inside `initialize_emg()`, `read_emg()`, and `close_emg()`.
   - Pre-implemented connection stubs are provided for:
     - **BITalino** (serial / bluetooth via `bitalino` package)
     - **Myo Armband** (USB dongle via `myo-python`)
     - **Generic Serial ADC** (microcontroller streaming comma-separated values via `pyserial`)
     - **Lab Streaming Layer (LSL)** (high-precision network streams via `pylsl`)
3. **Clinical Integration in `main.py` & `game/game_engine.py`**:
   - Trajectory samples will record muscle activation $[0.0 - 1.0]$ alongside spatial coordinates $(x, y, t)$, enabling correlation of co-contraction and spasticity with path deviation.

---

## Project Structure

```
Grid Game/
├── config.py                 # Centralized configuration, thresholds & color palettes
├── main.py                   # Application entry point & high-level state machine
├── requirements.txt          # Production dependencies (opencv, mediapipe, numpy)
├── README.md                 # System overview, installation, and user manual
│
├── game/                     # Core Game Physics & Level Logic
│   ├── collision.py          # Continuous Collision Detection (CCD) & obstacle resolution
│   ├── player.py             # Continuous 2D player cursor & cumulative distance tracking
│   ├── levels.py             # Geometric level blueprints (polygons, corridors, waypoints)
│   ├── maze.py               # Solvable maze generation (curated & procedural)
│   └── game_engine.py        # Session state machine (READY, PLAYING, RESULTS, TIMEOUT)
│
├── vision/                   # Optical Hand Tracking & Filter Pipeline
│   ├── hand_tracker.py       # MediaPipe 21-keypoint landmark extraction
│   └── smoothing.py          # OneEuroFilter2D (adaptive low-pass) & EMA filters
│
├── metrics/                  # Clinical Kinematics & Rehabilitation Logging
│   ├── distance.py           # Euclidean path distance & path efficiency formulas
│   ├── accuracy.py           # Corridor deviation & route adherence metrics
│   ├── smoothness.py         # Normalized jerk & movement smoothness score [0-100]
│   ├── performance.py        # MetricsCollector aggregating all kinematic indicators
│   ├── adaptive.py           # 5-dimension adaptive difficulty recommendation engine
│   ├── session_recorder.py   # Crash-proof JSON & daily CSV session serialization
│   └── history_reader.py     # Progress loader, difficulty filters & trend series
│
├── ui/                       # Professional Medical/Research Prototype Renderer
│   ├── renderer.py           # Layered frame compositor (HUD, telemetry, overlays, PiP)
│   └── screens.py            # Menu, Level Select, Results, and History screen renderers
│
├── emg/                      # Isolated EMG Interface (Future Integration)
│   └── emg_interface.py      # Hardware driver stubs, DSP math, and anti-fake rules
│
├── data/
│   └── sessions/             # Automated session archives (*.json and *.csv)
│
└── scratch/                  # Automated Test Suites & Verification Utilities
    ├── test_production_review.py   # Complete 30-requirement & edge-case test suite
    ├── test_emg_integration.py     # Hardware isolation & anti-fake tests
    ├── test_adaptive_difficulty.py # Clinical recommendation & consistency tests
    └── render_game_states.py       # Headless screen rendering & visual inspection tool
```

---

## Automated Test Verification

To execute the complete regression test suite:

```bash
python -m unittest discover -s scratch -p "test_*.py"
```

**Test Results**: 109 out of 109 automated unit tests pass with zero regressions.
