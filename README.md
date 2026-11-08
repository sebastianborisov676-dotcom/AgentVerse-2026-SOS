
# Team SOS — AutoCharger AI

**AgentVerse 2026 | UCL AI Society | Robotics Track**

AutoCharger AI is a simulated autonomous electric-vehicle charging assistant. It aims to identify a vehicle's charging port, align a robotic charging connector, and perform a controlled connection while responding to changing conditions.

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
python scripts/autonomy_demo.py             # camera-driven autonomy; arrow keys push the car live
python scripts/autonomy_demo.py --record outputs/demo.mp4 --move-at 3 --seed 11   # headless video
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
