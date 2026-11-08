
# Team SOS — AutoCharger AI

**AgentVerse 2026 | UCL AI Society | Robotics Track**

AutoCharger AI is a simulated autonomous electric-vehicle charging assistant. It aims to identify a vehicle's charging port, align a robotic charging connector, and perform a controlled connection while responding to changing conditions.

## Run the demo (submission version)

This version combines Sebastian's operator control panel, obstacle courses and Level 11 master challenge with SAI's
camera-driven autonomy, LiDAR navigation and car-park judge app (branch `integration/autocharger-demo`).
Tested on Windows 11 with Python 3.13, MuJoCo 4.0.0, NumPy 2.5 and OpenCV 5.0. Simulation only; CPU only.

```bash
git clone https://github.com/sebastianborisov676-dotcom/AgentVerse-2026-SOS.git
cd AgentVerse-2026-SOS
bash scripts/setup.sh    # creates .venv, installs requirements-all.txt, downloads Google Robot (MuJoCo Menagerie)
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
```

The robot model is downloaded by `setup.sh` into `models/` and is not stored in Git.

**1. Operator control panel: obstacle Levels 1–11, moving car, master challenge**

```bash
python scripts/run_control_panel.py          # then open http://127.0.0.1:8765
```

The panel renders offscreen with EGL by default (Linux). On Windows or macOS set `MUJOCO_GL=glfw` first
(`export MUJOCO_GL=glfw`, or in PowerShell `$env:MUJOCO_GL="glfw"`).
Master-challenge demo settings: **Level 11, obstacles Random, Randomize start on, Car challenge on, seed 42**:
the robot detects the obstacle, detours, brakes when the car moves, re-targets and seats the connector.
Seed 43 ends in a deliberate safe stop when the cable strain-relief load reaches its 45 N limit.

**2. Judge app in the car park: 11 challenges, camera + LiDAR autonomy**

```bash
python scripts/interactive_test.py
```

Click the "robot view" window so it has keyboard focus. Keys `1`–`0` and `-` pick a challenge and `SPACE` starts it. `K` drops a bin bag, `N` parks neighbour cars,
`M` makes the car roll, and `WASD`/`QE` move the car.
The robot only sees through its own camera and LiDAR; an independent checker grades each run.

**3. Reproducible checks**

```bash
python scripts/run_course_batch.py --levels 1 9 11 --seeds 42 43 --output outputs/course_batch.json
python -m pytest tests
```

## Project objectives

- Detect the charging port using simulated camera observations.
- Estimate distance, position and orientation.
- Align the connector using robotic actuators.
- Verify alignment before attempting insertion.
- Detect failed alignment or excessive contact force.
- Stop, retract and retry when necessary.
- Support operator start, stop and emergency-stop commands.


## Team SOS

Developed for the AgentVerse 2026 Hackathon,
organised by UCL AI Society.

### Team Members

| Member | Role | Responsibilities |
|---|---|---|
| Sebastian Stoyanov | Team Leader | Robot control, charging-port alignment, simulation integration |
| SAI | Team Member | AutoCharger AI development; responsibilities to be confirmed |

### Development Branches

- `main` — Shared, stable version of the project.
- `feature/charger-alignment` — Sebastian's development branch.
- `sos-AutoCharger-AI` — SAI's development branch.

All features are developed independently and integrated through GitHub pull requests.


## Technology stack

- Python 3.14
- MuJoCo 4.0 — robotics and physics simulation
- NumPy — numerical calculations and alignment errors
- Pytest — automated testing
- Modal — optional cloud compute
- Ultralytics YOLO — planned local charging-port detection
- OpenCV — planned image processing

## Proposed architecture

Simulation (MuJoCo)
    |
    v
Sensors (camera, distance, joint positions, force)
    |
    v
Perception (charging-port localisation)
    |
    v
Alignment controller (position and orientation errors)
    |
    v
Robot actuators (rail, lift, rotation, extension)
    |
    v
Verification and recovery
    |
    +---- Feedback to sensors

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sebastianborisov676-dotcom/AgentVerse-2026-SOS.git
cd AgentVerse-2026-SOS
```

### 2. Create a virtual environment

Requires Python 3.11 or newer. The development environment currently uses Python 3.14.4.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install core dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify MuJoCo

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m mujoco.viewer
```

### 5. Optional computer vision dependencies

```bash
python -m pip install -r requirements-ai.txt
```

### 6. Run tests

```bash
python -m pytest tests/
```

### 7. Run the AutoCharger simulation (Google Robot + cable + CCS2-style connector)

```bash
python scripts/run_autocharge.py            # headless run with a report
python scripts/run_autocharge.py --viewer   # interactive viewer
python scripts/run_autocharge.py --drive-in --viewer   # start 3 m away, drive, dock, plug in
python scripts/visual_report.py             # replay every test into outputs/test_report/index.html
```

On Windows Git Bash, activate the environment with `source .venv/Scripts/activate`.
Design, validation results and the list of approximations are in
[docs/autocharge_cable_connector.md](docs/autocharge_cable_connector.md).

## Project structure

```text
AgentVerse/
├── docs/                 # Technical notes and diagrams
├── models/               # External simulation assets
├── src/                  # Simulation and controller code
├── tests/                # Automated tests
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-ai.txt
```

## Team workflow

- `main` — integrated project
- `feature/charger-alignment` — charging-port alignment development
- `sos-AutoCharger-AI` — SAI's development branch

Develop features independently and use pull requests to integrate and review changes.

## Project status

- [x] Git repository initialised
- [x] Python virtual environment configured
- [x] MuJoCo installed and viewer tested
- [x] Modal configured
- [x] Vehicle and charging-port simulation (simplified CCS2-style inlet)
- [x] Charging robot integrated (Google Robot, connector + flexible cable)
- [x] Alignment controller (alignment-gated slow insertion)
- [ ] Perception system
- [x] Recovery and safety testing (force / tension limits, retry, abort, e-stop)
- [ ] Final demonstration

## Simulation limitations

The initial prototype will use simulated vehicles, connectors and sensors. Successful alignment in simulation does not establish real-world charging safety or electrical compatibility.

## Credits

Developed by Team SOS for AgentVerse 2026.

Simulation: Google DeepMind MuJoCo.


## Progressive obstacle course

Run `python scripts/run_autocharge.py --level 2 --seed 0 --viewer` for the course demonstration,
or `python scripts/run_course_batch.py --engine panel --levels 1 2 5 10 --seeds 42 43 --workers 2` for headless evaluation.
See [course documentation](docs/obstacle_course.md) for manual layouts, the training API,
reproducibility and physics approximations.


## Local operator control panel

```bash
.venv/bin/python scripts/run_control_panel.py
```

Open **http://127.0.0.1:8765** for the level/seed selector, obstacle editor, live MuJoCo
RGB/depth views, telemetry, recording timeline and A/B comparisons. Default seed: **42**.
Levels 9–10 include the optional moving-car challenge. The panel uses a fixed physical
cable and limited-range ray observations; infeasible starts are reported.
See [control-panel documentation](docs/control_panel.md) for operating instructions,
checkpoint/playback distinctions, validation and explicit physics/perception approximations.


Current revision: [browser verification and parallel results](docs/operator_validation.md).
Sample layouts: [single avoidance box](layouts/box_avoidance.json) and
[stone + slope](layouts/stone_slope.json). The batch runner now defaults to the same
scenario builder and sensor-based controller as the panel; use `--engine legacy` only
for the earlier privileged-map baseline.

## Perception-driven autonomy, car park and judge tests (merged from `sos-AutoCharger-AI`)

These run alongside the control panel and use the same scene builder (`src/autocharge/scene.py`).

```bash
python scripts/interactive_test.py          # 11 judge challenges in the car park (keys 1-0 and -, SPACE starts)
python scripts/autonomy_demo.py             # camera-driven autonomy; arrow keys push the car live
python scripts/watch_tests.py               # replay the autonomy test scenarios with a PASS/FAIL checker
python scripts/benchmark.py                 # held-out trials: autonomous vs pre-programmed controller
python -m pytest tests                      # all automated tests from both branches
```

- **Perception** (`perception.py`): robot camera on the connector, ArUco fiducial beside the inlet, `solvePnP`
  pose with reprojection checks. Rule-based OpenCV pipeline; no learned model.
- **Autonomy** (`autonomy.py`): reachability check with the robot's own IK, drives the base closer when needed,
  360 deg search at three camera tilts, holds when perception is stale, re-plans when the car moves,
  safe stop on unexpected touch or when the car moves while the plug is engaged.
- **Navigation** (`navigation.py`): simulated planar LiDAR -> occupancy grid -> A* -> waypoint driving.
  Cable-aware: cells whose straight cable line back to the charger outlet would cross an observed obstacle
  cost more, so detours go round the side that keeps the charging cable from wrapping round and snagging.
- **Independent evaluator** (`evaluation.py`): judges docking from the physical socket pose; the agent never imports it.

Design notes, measured results and limitations: `docs/autocharge_cable_connector.md`, one-page summary:
`docs/technical_summary.pdf`.
