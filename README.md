# 🧠 Continuous 2D Hand-Controlled Maze Game for Stroke Rehabilitation

An interactive, continuous 2D motor rehabilitation platform engineered for stroke recovery, upper-limb physical therapy, and kinematic motor assessment. Patients control a floating precision cursor in continuous physical coordinate space using optical hand tracking (OpenCV + MediaPipe), navigating through clinically calibrated mazes with sub-pixel collision physics, direct minimal path benchmarks, and real-time clinical kinematics logging.

---

## 📑 Table of Contents
- [Quick Start](#-quick-start)
- [How to Play: 100% Hand-Gesture Control](#-how-to-play-100-hand-gesture-control-no-keyboard-needed)
  - [Step 1: Main Menu & Direct Level Selection](#step-1-main-menu--direct-level-selection)
  - [Step 2: Starting the Maze at the Green START Beacon](#step-2-starting-the-maze-at-the-green-start-beacon)
  - [Step 3: Navigating the Maze in Continuous 2D Space](#step-3-navigating-the-maze-in-continuous-2d-space)
  - [Step 4: Goal Completion & Clinical Feedback](#step-4-goal-completion--clinical-feedback)
  - [Step 5: Results Screen & Level Progression](#step-5-results-screen--level-progression)
- [Moveable & Non-Overlapping Webcam Camera (PiP)](#-moveable--non-overlapping-webcam-camera-pip)
- [Direct Minimal Path Calculation](#-direct-minimal-path-calculation)
- [Clinical Kinematics & Performance Metrics](#-clinical-kinematics--performance-metrics)
- [Controls & Shortcuts Quick Reference](#-controls--shortcuts-quick-reference)
- [Architecture & Codebase Structure](#-architecture--codebase-structure)
- [Future EMG Hardware Integration (Hardware Abstraction)](#-future-emg-hardware-integration)
- [Automated Verification & Testing](#-automated-verification--testing)

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13
- **Hardware**: Any standard USB webcam or built-in laptop camera *(zero special sensors required)*.

### 2. Installation
```bash
# Navigate to the project root directory
cd "Grid Game"

# Install dependencies (OpenCV, MediaPipe, NumPy, Pytest)
pip install -r requirements.txt
```

### 3. Launch the Game
```bash
python main.py
```
> **Tip**: If no camera is connected or if testing without hands, mouse cursor fallback is automatically enabled so you can test immediately with your mouse.

---

## 🖐️ How to Play: 100% Hand-Gesture Control (No Keyboard Needed)

The game is designed to be operated **completely hands-free from first to last screen** using a **1.5-second dwell hold** gesture. A glowing circular reticle tracks your hand across every screen.

```
┌─────────────────┐       Hold 1.5s on Level      ┌─────────────────┐
│    MAIN MENU    │ ────────────────────────────> │   READY STATE   │
│ (EASY/MED/HARD) │                               │  (Maze Preview) │
└─────────────────┘                               └────────┬────────┘
                                                           │
                                                           │ Hold 1.5s inside
                                                           │ GREEN START beacon
                                                           ▼
┌─────────────────┐       Touch RED Goal Beacon   ┌─────────────────┐
│ RESULTS SCREEN  │ <──────────────────────────── │  ACTIVE MAZE    │
│ (Metrics & Next)│                               │ (PLAYING State) │
└─────────────────┘                               └─────────────────┘
```

---

### Step 1: Main Menu & Direct Level Selection
1. Position your hand comfortably in front of your webcam.
2. An amber/cyan glowing reticle follows your index fingertip in real time.
3. Move your hand over any difficulty card:
   - **`1 EASY`**: Wide corridors, gentle curves, baseline motor evaluation.
   - **`2 MEDIUM`**: Moderate corridor widths, diagonal turns, motor coordination challenge.
   - **`3 HARD`**: Narrow paths, multi-junction navigation, fine motor tremor control.
4. **Hold still for 1.5 seconds**: A green progress fill bar animates inside the selected button.
5. Once full, the game transitions **directly into the Game Section (`READY` state)** with Level 1 ready to play.

---

### Step 2: Starting the Maze at the Green START Beacon
To preserve kinematic measurement integrity, active gameplay begins **only when your hand is physically positioned at the origin**:
1. In the `READY` state, observe the maze layout and the **Direct Minimal Path** line.
2. Move your hand cursor inside the glowing **green START circle beacon**.
3. A circular green progress gauge wraps smoothly around the START circle.
4. **Hold inside START for 1.5 seconds**:
   - The green gauge reaches 100%.
   - Active gameplay (`PLAYING` state) begins immediately.
   - **Zero initial jump lines**: Because the hand is already resting at the START beacon, there is no artificial straight-line teleportation artifact.

---

### Step 3: Navigating the Maze in Continuous 2D Space
- **Continuous 2D Coordinates**: No tile grids or blocky steps. The player avatar ($r=10-16\text{ px}$) glides smoothly following your hand's exact $(x, y)$ coordinates.
- **Continuous Collision Detection (CCD)**: Even if you move your hand rapidly, CCD prevents tunneling through walls.
- **Wall Sliding**: Brushing against a wall produces a subtle tactile red pulse and lets you slide tangentially along obstacle boundaries without getting stuck.
- **Live HUD Feedback**: The telemetry card at the top displays your elapsed time, actual distance traveled, wall collisions, and efficiency.

---

### Step 4: Goal Completion & Clinical Feedback
- Guide your cursor to the **red/gold END beacon** at the maze exit.
- When the cursor touches the goal, the session timer freezes immediately, all kinematic computations are finalized, and your session is automatically archived to disk (`data/sessions/`).

---

### Step 5: Results Screen & Level Progression
The results screen provides detailed rehabilitation metrics and full gesture navigation:
- **`NEXT LEVEL`**: Hold for 1.5s to load the next maze in the difficulty tier.
- **`REPLAY`**: Hold for 1.5s to retry the same maze and improve movement efficiency.
- **`LEVEL SELECT`**: Hold for 1.5s to pick a specific level blueprint.
- **`MAIN MENU`**: Hold for 1.5s to change difficulty or view long-term recovery history.

---

## 📹 Moveable & Non-Overlapping Webcam Camera (PiP)

The webcam Picture-in-Picture (PiP) thumbnail lets patients and clinicians verify hand posture and landmark tracking during therapy. To ensure it never blocks your view:

### 1. Intelligent Auto-Placement
When any maze loads, the engine evaluates all 4 corners using a clinical penalty matrix:
- **Zero overlap** with START beacon.
- **Zero overlap** with END beacon.
- **Minimal obstruction** of active maze walls and corridors.
- *Example*: In `Easy-1`, the camera is automatically placed at the **Bottom-Right corner**, leaving START (Top-Left) and END (Bottom-Left) completely unobstructed.

### 2. Four Easy Ways to Move the Camera
| Method | How to Use |
| :--- | :--- |
| **🖐️ Hand Gesture (Header)** | Hover your hand cursor over the camera top bar `[CAM (DRAG / MOVE)]` for 1.5s to cycle to the next open corner. |
| **🖐️ Gesture Button** | Hover your hand cursor over the `MOVE CAM [V]` button for 1.5s. |
| **🖱️ Mouse Drag & Drop** | Click and drag the camera anywhere across the canvas with your mouse. |
| **⌨️ Keyboard Shortcut** | Press the **`V`** key to cycle corners instantly. |

---

## 📐 Direct Minimal Path Calculation

A key requirement in motor rehabilitation is calculating the **true direct Euclidean minimal path** without artificial right-angle detours.

```
FLAWED (Old Grid Waypoints):     DIRECT EUCLIDEAN (Current System):
      START ──┐                         START
              │                               \
              └───┐                            \   Direct Corridor Line
                  │                             \  (Exact Shortest Route)
                  └──> END                       \
                                                  └──> END
```

- **Continuous Distance-Transform String-Pulling**: The system determines the shortest obstacle-free route using visibility graph ray-casting and distance field gradient descent.
- **Exact Geometric Distances**:
  - **Easy-1**: **`833.6 px`** (Direct diagonal corridor through open space)
  - **Medium-1**: **`1017.3 px`** (Direct diagonal bypass)
  - **Hard-1**: **`2567.5 px`** (Optimal Euclidean multi-corridor route)
- **Visual Path Guide in READY State**: During the `READY` state, the direct minimal path line and its exact length (`MIN PATH: 833.6 px`) are rendered directly on the maze so patients can plan their movement trajectory before starting.

---

## 📊 Clinical Kinematics & Performance Metrics

> ⚠️ **Clinical Principle: Quality Over Speed**  
> In neurological motor rehabilitation, rapid movements often trigger spastic co-contractions and compensatory tremor. The system prioritizes smoothness, corridor adherence, and efficiency over raw completion speed.

| Metric | Clinical Meaning | Target / Formula |
| :--- | :--- | :--- |
| **Actual Distance** | Total cumulative path traveled by the patient's hand. | $\sum \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}$ |
| **Minimum Distance** | Shortest collision-free path from START to END. | Continuous Euclidean distance transform |
| **Path Efficiency** | Ratio of optimal path vs. actual hand trajectory. | $\frac{\text{Minimum Distance}}{\text{Actual Distance}} \times 100\%$ |
| **Trajectory Accuracy** | Percentage of time spent within the optimal movement corridor. | Points within corridor tolerance ($\pm 25\text{ px}$) |
| **Movement Smoothness** | Logarithmic normalized jerk profile assessing tremor and hesitation. | Score $[0 - 100]$, where $>80$ is smooth coordination |
| **Wall Collisions** | Boundary impacts indicating motor overshoot or difficulty with braking. | Count of obstacle touches |
| **Adaptive Difficulty** | AI recommendation assessing whether to promote, maintain, or repeat. | 5-point clinical evaluation rules |

---

## ⌨️ Controls & Shortcuts Quick Reference

While the entire game can be played using **only hand gestures**, physical keyboard and mouse controls are available for clinician oversight:

| Key / Input | Applicable Screen | Function |
| :--- | :--- | :--- |
| **Hand Hover (1.5s)** | All Screens | Activate hovered button or start beacon |
| **`ENTER` / `SPACE`** | Menu / Ready | Start maze or confirm selection |
| **`1` / `2` / `3`** | Menu / Level Select | Switch difficulty (`1: EASY`, `2: MEDIUM`, `3: HARD`) |
| **`V`** | Any Screen | Cycle webcam PiP thumbnail between 4 corners |
| **`L`** | Any Screen | Cycle tracked hand landmark (`INDEX_TIP` $\to$ `WRIST` $\to$ `THUMB_TIP`) |
| **`D`** | Gameplay | Toggle real-time telemetry debug overlay (Raw vs. Filtered coords) |
| **`M`** | Any Screen | Toggle mouse cursor fallback control |
| **`P`** | Gameplay | Pause / Resume session |
| **`R`** | Gameplay / Results | Restart current level cleanly to `READY` state |
| **`H`** | Menu | Open Session Progress & History dashboard |
| **`ESC`** | Any Screen | Return to Main Menu or exit application |

---

## 🏗️ Architecture & Codebase Structure

```
Grid Game/
├── config.py                 # Central configuration (resolution, colors, thresholds, smoothing)
├── main.py                   # Application controller, gesture engine, PiP compositor & main loop
├── requirements.txt          # Python library dependencies
├── README.md                 # System overview and comprehensive user guide
│
├── game/                     # Core Game Physics & Level Logic
│   ├── collision.py          # Continuous Collision Detection (CCD) & obstacle geometry
│   ├── player.py             # Continuous 2D player physics, actual distance accumulator
│   ├── levels.py             # Geometric level blueprints (polygons, corridors, start/end)
│   ├── maze.py               # Solvable level generator (curated blueprints + procedural)
│   ├── pathfinding.py        # Continuous Euclidean string-pulling & minimal path calculation
│   └── game_engine.py        # State machine (READY, PLAYING, RESULTS, PAUSED, TIMEOUT)
│
├── vision/                   # Computer Vision & Signal Processing
│   ├── hand_tracker.py       # MediaPipe 21-point hand skeleton tracking & landmark selection
│   └── smoothing.py          # 1€ Filter (One Euro) adaptive jitter removal + EMA filters
│
├── metrics/                  # Clinical Kinematics & Session Logging
│   ├── distance.py           # Euclidean path distance & path efficiency equations
│   ├── accuracy.py           # Optimal corridor adherence & perpendicular deviation
│   ├── smoothness.py         # Normalized jerk & clinical smoothness index
│   ├── adaptive.py           # 5-criterion adaptive difficulty recommendation engine
│   ├── performance.py        # Comprehensive MetricsCollector for active sessions
│   ├── session_recorder.py   # Atomic JSON and daily CSV session logging
│   └── history_reader.py     # Multi-session recovery history parser & trend aggregation
│
├── ui/                       # Professional Medical UI & Visualization
│   ├── renderer.py           # Canvas compositor (HUD, reticle, path guides, overlays)
│   └── screens.py            # Menu, Level Select, Results, and History screen layouts
│
├── emg/                      # Hardware Abstraction Layer (Future Integration)
│   └── emg_interface.py      # Isolated EMG hardware drivers, signal filters, anti-fake rules
│
└── scratch/                  # Automated Verification & Test Suites (110 Tests)
    ├── test_gesture_workflow_e2e.py # End-to-end hand gesture state transition verification
    ├── test_production_review.py    # 30-requirement clinical compliance test suite
    ├── test_minimum_path.py         # Euclidean minimal path & string-pulling tests
    ├── test_movement_smoothness.py  # Jerk and smoothness mathematical verification
    ├── test_adaptive_difficulty.py  # Clinical progression and regression logic tests
    └── test_emg_integration.py      # Hardware isolation & anti-fake measurement verification
```

---

## ⚡ Future EMG Hardware Integration

The system includes a dedicated, fully isolated hardware abstraction layer in `emg/emg_interface.py`.

### Strict Anti-Fake Guarantee
- By default, `config.EMG_ENABLED = False`.
- The game **never displays simulated or fake muscle sensor values**. When EMG is disabled, the HUD clearly shows `EMG: Not Connected / Disabled`.

### Connecting Physical EMG Hardware
When physical sEMG hardware is available (BITalino, Myo Armband, or custom serial ADC):
1. In `config.py`, change:
   ```python
   EMG_ENABLED = True
   EMG_PORT = "COM3"           # or "/dev/ttyUSB0" on Linux
   EMG_DEVICE = "bitalino"     # "bitalino", "myo", "custom_serial", or "lsl"
   ```
2. Hardware drivers in `emg/emg_interface.py` provide pre-implemented hooks for 20–450 Hz bandpass filtering, 50/60 Hz notch filtering, RMS envelope calculation, and MVC normalization $[0.0 - 1.0]$.
3. EMG activation will automatically log alongside trajectory coordinates $(x, y, t)$ to evaluate muscle effort against movement precision.

---

## 🧪 Automated Verification & Testing

The platform includes a test suite covering kinematics, collision detection, gesture transitions, and minimal path accuracy.

### Running All Tests
```bash
pytest scratch/
```

### Verification Status
```
============================ 110 passed in 57.74s =============================
```
- **110 of 110 tests pass with zero regressions.**
- Tests verify sub-pixel collision accuracy, continuous distance accumulation, 1.5s gesture dwell timings, minimal path precision, and isolated sensor handling.

---

## 📄 License & Clinical Disclaimer
This software is intended as an interactive prototype for upper-limb motor rehabilitation research and evaluation. It should be used in accordance with physical therapy protocols and clinical guidance.
