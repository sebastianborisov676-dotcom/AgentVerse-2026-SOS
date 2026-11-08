# AutoCharger AI — Charging Cable, Connector and Insertion

Simulation of the Google Robot (MuJoCo Menagerie, Apache-2.0) carrying a simplified CCS2-style
connector on a flexible charging cable, and inserting it into a simplified vehicle inlet with
alignment checks, contact-force limits and cable-tension limits.

Built on MuJoCo 4.0.0. The original robot files are **not modified** — a test checks they are
byte-identical to upstream.

## Run it

```bash
source .venv/Scripts/activate          # Windows Git Bash  (Linux/macOS: .venv/bin/activate)
python scripts/run_autocharge.py                         # headless run + report
python scripts/run_autocharge.py --viewer                # interactive MuJoCo viewer
python scripts/run_autocharge.py --gif outputs/run.gif --csv outputs/telemetry.csv
python scripts/run_autocharge.py --perception-error-mm 6 # simulate a bad socket estimate
python -m pytest tests                                   # 15 validation tests (~3 min)
```

## Code layout

| File | Purpose |
|---|---|
| `src/autocharge/scene.py` | Builds the scene: includes the unmodified `robot.xml`, adds charger, cable, connector, vehicle socket; solves the home pose; routes the cable; writes the `home` keyframe. |
| `src/autocharge/controller.py` | Insertion state machine and Cartesian controller on top of the robot's original position servos. |
| `src/autocharge/metrics.py` | Alignment error, connector↔socket contact force, cable tension / strain / clearance. |
| `scripts/run_autocharge.py` | Demo runner (headless, viewer, GIF, CSV). |
| `tests/` | Stability and insertion/safety validation. |

## How each part is modelled

**Robot.** `models/mujoco_menagerie/google_robot/robot.xml` is pulled in with `<include>`; nothing
in it is edited. Two global simulation options had to change for the whole scene (not in the
robot file): `integrator="discrete"` (MuJoCo 4 rod elasticity does not run under the robot's
original `implicitfast`) and `timestep=0.002` (robot default 0.004).

**Connector.** A `connector` body is added as a child of `link_gripper` at the robot's
`gripper` site, with no joint, so it is rigidly mounted. It has a housing, a handle, a strain-relief
boot, an AC upper pin (Ø48 mm) and two DC lower pins (Ø16 mm), each pin with a domed tip that
acts as a lead-in chamfer. 0.9 kg, `gravcomp=1` like every link of the original arm.

**Cable.** MuJoCo 4 removed the old `composite` cable/`elasticity.cable` plugin; the replacement is
a 1-D `flexcomp` rod (discrete elastic rod: bending + torsion, per-edge twist DOFs):

* 40 segments, 2.94 m long, Ø24 mm, 0.9 kg/m (2.6 kg total), bending modulus 60 MPa,
  Rayleigh damping 0.02 s, friction 0.8.
* Inextensible edges via edge equality constraints (measured stretch ≤ 0.8 %).
* **Anchor:** the first edge is clamped (`<pin id="0 1"/>`) in the charger's outlet.
* **Strain relief:** the last cable point is constrained to the boot site and the second-last to
  a point one segment further back along the boot axis — this fixes both position and exit
  direction of the cable at the connector, so the cable cannot kink at the handle.
* **Slack:** the rest shape is straight; the start pose routes it out of the outlet, down to the
  floor, along the floor and up into the boot (equal-chord resampling, so it starts unstrained).
  Cable length / straight-line span = **3.2**.
* **Collisions:** the cable collides with the floor, the robot's collision meshes, the charger
  and the vehicle. It does not collide with the connector it is attached to (bitmask), nor with
  itself (`selfcollide="none"`).

**Vehicle socket.** Static car body with an inlet: one AC and two DC cavities, each a ring of 16
box segments plus a floor, 45 mm deep, radial clearance 2.0 mm (AC) / 1.5 mm (DC).

**Charger.** Static pedestal with an outlet holster at 1.05 m.

## Insertion controller

`APPROACH → ALIGN → INSERT → SEATED`, with `RETRACT` (retry, max 2), `ABORTED` and `ESTOP`.

* Drives the arm only through the robot's original position actuators. Each step: damped
  least-squares on the plug-face pose error, integrated into the servo set-points (removes sag),
  plus a proportional lead and model-based Coulomb friction compensation using the robot's own
  `frictionloss`/`kp` (without it the servos stick-slip and hunt by 2–5 mm).
* Alignment gate before insertion: lateral < 1.0 mm, angle < 1.0°, at the 60 mm standoff, held for
  0.25 s with no contact.
* Insertion at **5 mm/s**. Watchdogs every step: contact force (soft 30 N → pause, then retract
  after 1.5 s; hard 70 N → retract), lateral deflection > 2.5 mm or angle > 2.5° → retract,
  cable tension > 60 N or strain-relief load > 45 N → retract.
* Contact force is low-pass filtered (τ = 30 ms) to model a wrist F/T sensor; raw solver peaks are
  recorded separately.
* The socket pose is an **input** (`socket_estimate`). It defaults to ground truth; camera
  perception will supply it later. All alignment decisions use the estimate, so perception error
  behaves as it would on hardware.

## Validation results (all measured, `pytest tests`: 15 passed)

| Check | Result |
|---|---|
| Robot files unmodified | ✅ `git status` of Menagerie checkout clean |
| Connector rigid on gripper | ✅ child of `link_gripper`, 0 joints; home IK error 0.004 mm / 0.0001° |
| 10 s hold | ✅ no divergence or solver warnings; edge strain < 1.5 %; floor penetration < 2 mm; tension < 15 N |
| Large arm swing dragging the cable | ✅ stable, strain < 3 %, penetration < 4 mm |
| Nominal insertion | ✅ seated first try at t ≈ 17 s, 34 mm engagement, lateral 0.3 mm, 0.05°, zero contact force, cable tension ≤ 7 N |
| 3 mm perception error | ✅ seated; pin chamfers guide it in (≤ 29 N filtered) |
| 4 mm perception error | ✅ unexpected deflection detected → retract ×3 → abort |
| 6 mm perception error | ✅ jam detected at soft force limit → retract ×3 → abort, never seated |
| Hard contact-force limit alone | ✅ fires and aborts |
| Cable-tension limit / strain-relief limit | ✅ each fires and retracts (tested with lowered thresholds) |
| Emergency stop | ✅ arm freezes, < 3 mm drift over 1 s |
| Speed | ≈ real time on a laptop CPU (18 s simulated in 18 s) |

## What works vs. what is an approximation

**Works (physically simulated, validated):** rigid connector mount; cable anchored at the charger
and attached through a strain-relief point; cable bending, torsion, inextensibility, gravity sag,
friction, contact with floor/robot/charger/vehicle; slack routing; peg-in-hole contact between
the pins and socket cavities including chamfer guidance and jamming; alignment-gated slow
insertion; force, tension, deflection and e-stop safety responses; retry/abort logic.

**Approximations — do not read these as real-world validation:**

1. **Geometry is simplified.** Not to the IEC 62196-3 drawing: no CP/PP signal pins, no latch, no
   individual AC contacts, no inlet flap. Cavities are 16-sided polygons made of boxes, so the
   rim has seams.
2. **No latching or electrical contact.** "Seated" means the plug face reached 2 mm from the
   socket face. There is no latch click, no contact-pin spring force, no electrical model.
3. **Insertion force is unrealistically low.** Real CCS2 insertion needs roughly 50–100 N against
   contact springs; here the cavities have clearance and no spring contacts, so a well-aligned
   insertion sees ~0 N.
4. **Contact force is a solver quantity.** Per-step forces jump between contact points on the
   segmented rim (raw peaks up to ~250 N for one 2 ms step). The controller acts on a filtered
   signal; a real F/T sensor would also see the arm's own dynamics.
5. **Cable material parameters are effective values,** not measured from a real CCS2 cable.
   Cable self-collision is off, so it can pass through itself (not seen in these runs).
   Cable↔connector collision is off near the attachment.
6. **The strain relief is two point constraints,** not a tapered elastomer boot; cable twist at
   the connector end is not clamped.
7. **Global options changed** for the whole scene (discrete integrator, 2 ms timestep). The robot
   file is untouched, but the arm's integration differs slightly from Menagerie defaults.
8. **Connector gravity compensation** (`gravcomp=1`) follows the robot model's own convention
   for its links; the arm's servos do not carry the connector's weight. The cable's weight is
   *not* compensated and is a real disturbance.
9. **Fixed base.** The Menagerie Google Robot has no wheel joints; the "mobile base" is static.
10. **Perfect state knowledge.** The controller reads joint states and (by default) the true
    socket pose. Camera perception, latency and calibration error are not modelled yet —
    `--perception-error-mm` is a stand-in.
11. **Start-up settling.** The cable is released from a routed pose; `reset_to_home` holds 0.5 s so
    a ~20 N numerical tension transient settles before the controller arms its limits.

## Drive-in mission (robot starts 3 m away)

`python scripts/run_autocharge.py --drive-in --viewer`

The robot starts **3.04 m** from its docking spot, 0.5 m to the side and turned 15°, already
plugged into the charger with a **4.55 m** cable (64 segments). `ChargingMission`
(`src/autocharge/mission.py`) runs it in two stages:

1. **Drive and dock** (`src/autocharge/base_controller.py`): pure-pursuit along a smooth Bezier
   path at up to 0.35 m/s with acceleration limits, then a slow final correction until the base is
   within **1 cm and 0.5°** for 0.4 s. Brakes then hold the base. While driving, the arm holds its
   carry pose with an integral joint loop so the cable's drag does not push the connector around,
   and the drive stops if cable tension or boot load passes its limit.
2. **Align and insert**: the same alignment-gated insertion controller as before, started only
   after the base is parked.

Measured (test `test_drive_in_from_3m_docks_aligns_and_seats`): drove 3.12 m, parked 7.6 mm /
0.17° from target at t ≈ 11 s, aligned to 0.93 mm / 0.04°, seated at t ≈ 30 s; cable tension
≤ 12 N, boot load ≤ 14.5 N, cable end never more than 3 mm from the strain relief, base never
drove over the cable, no solver warnings. `test_drive_stops_when_cable_pulls_too_hard` shows the
base stops and brakes when the cable tension limit is hit, and the arm never starts.

Approximations specific to driving:

* **The base is not wheel-driven.** The Menagerie model has no wheel joints, so the scene gives the
  base planar x / y / yaw joints with velocity servos at load time (robot file untouched). The
  controller only commands forward speed and turn rate (differential-drive kinematics), but there
  is no wheel–floor contact, slip or rolling resistance: base–floor collision is switched off.
* **Brakes** are a stiff position hold through the same servos.
* **Docking uses the true base pose** (perfect odometry/localisation). Real docking would use
  wheel odometry plus a camera or fiducial on the charging bay.
* The 140 kg base mass comes from the robot model's mesh density, not a datasheet.

## Perception-driven autonomy (camera in the loop)

`python scripts/autonomy_demo.py` opens the MuJoCo viewer plus a robot-camera window showing the
detection, the estimated socket axis and every decision. Arrow keys and `[` `]` push the car,
`B` blocks the camera's view of the inlet, and `E` triggers an emergency stop. The robot is not told
about any of these changes. `--record outputs/demo.mp4 --move-at 3` writes a headless video.

**Loop (10 Hz perception, 500 Hz control):** Observe → Estimate → Decide → Act → Verify.

| Stage | Implementation | File |
|---|---|---|
| Observe | Robot camera on the connector (`connector_cam`, 960×720, 70° FOV, tilted 20° up) rendered by MuJoCo | `perception.py` |
| Detect | OpenCV ArUco detection of a fiducial on the car beside the inlet; sub-pixel corners | `perception.py` |
| Estimate | `solvePnP` (IPPE_SQUARE), reprojection-error check, camera pose from the arm's own joint angles, fixed marker→socket calibration | `perception.py` |
| Filter | Exponential filter; a jump > 4 mm / 1.5° means the car moved | `perception.py` (`PoseFilter`) |
| Decide | Agent states SEARCHING → TRACKING → SUCCESS / SAFE_STOP / FAILED on top of the insertion controller's APPROACH → ALIGN → INSERT → SEATED / RETRACT / ABORTED | `autonomy.py`, `controller.py` |
| Act | Damped-least-squares Cartesian servo on the robot's original position actuators | `controller.py` |
| Disturb | Scenario manager moves or covers the car independently of the robot | `scenarios.py` |

**Supervisor rules (all from measurements):**
- No reliable detection: the arm never moves. FAILED after 6 s.
- Detection stale for more than 0.35 s while approaching or aligning: hold position. Lost for more than 4 s: SAFE_STOP before any contact.
- Car moves before the pins engage: re-plan (back to approach, or back out of an insertion) and re-align.
- Car moves after the pins engage (more than 8 mm deep, past the chamfer), the plug is pushed off axis while engaged, or contact force passes 70 N while engaged: SAFE_STOP. The robot holds still and doesn't chase or yank with the plug inserted. Shallower chamfer deflection still backs out and retries.
- The force, cable-tension, strain-relief and deflection limits from the earlier controller still apply.

**Ground truth is never used for control.** The agent and perception code never read the
simulator's socket pose. It is used only by `evaluate` / `socket_estimate_error` and in test
assertions. `test_wrong_calibration_misleads_robot` shifts the marker calibration by 8 mm, and the
robot then follows its wrong perception and fails to seat. That shows the camera, not hidden data,
drives the motion.

**Measured (this laptop, CPU only):**
- Pose estimate at the start pose: about 0.2–0.6 mm and 0.1–0.2° from ground truth at 0.3 m range.
- Perception takes about 70–90 ms per frame (render + detect + PnP).
- Nominal run: seated with 0.13 mm real lateral error.
- Random parking (±3 cm along the axis, ±4 cm sideways, ±3 cm height, ±4°): seated.
- Car rolls 3 cm sideways and turns 3° during the approach: the robot detects the jump within 0.3 s and re-plans. Its first two insertion attempts drift about 2.5 mm, from joint stick-slip in the new arm pose, so the deflection watchdog backs out. It seats on attempt 3.
- Inlet view blocked: the arm never moves and the agent reports FAILED.
- Car moved while engaged: SAFE_STOP.

### Proof of independence

**Three separate roles.**

| Role | Code | Sees |
|---|---|---|
| Robot agent | `perception.py`, `autonomy.py`, `controller.py` | Camera images, camera calibration, its own joint angles, simulated force/cable load sensing |
| Environment | `scenarios.py` | Moves the car, changes lighting, closes the flap, blocks the view. Never talks to the agent |
| Independent evaluator | `evaluation.py` | Reads the physical socket pose (ground truth) to judge docking. Not imported by the agent |

**Experiments (all automated tests, `tests/test_independence.py` and `tests/test_autonomy.py`):**

| Experiment | What it proves | Result |
|---|---|---|
| Ground-truth poisoning: the simulator's socket "answer key" site shifted by 2 cm | The agent never reads it | Still docks (independent checker) |
| Calibration error injected (8 mm) | Actions follow perception | Robot follows its wrong estimate and doesn't seat |
| Camera blocked from the start | Perception gates movement | Arm never moves, reports FAILED |
| Camera blocked mid-approach (1.5–3 s) | Pauses on stale perception | Logs "hold", then "resume" when visible, docks |
| Car moves before engagement | Adaptation | Detects the jump and re-plans, docks |
| Car moves while engaged | Safety | SAFE_STOP, no chasing or yanking |
| Lighting at 15% / 3% | Robust or refuses when unsure | Docks / never moves |
| Charging flap closed | No false success | Jam detected by force, bounded retries, FAILED. Checker: not docked |
| Success claim vs independent checker | Success is verified | Agent's "seated" agrees with the checker (≥ 30 mm depth, inside clearance) |

**Held-out benchmark** (`python scripts/benchmark.py`): 10 trials with seeds 1000–1009, never used during development. Every trial has random parking (±3 cm along the axis, ±4 cm sideways, ±3 cm height, ±4°), and the odd-numbered trials also have a random car move during the approach. Each trial runs with both controllers. The independent checker judges every result.

| Controller | Scenario | Trials | Docked | Unsafe contacts (> 70 N) | Median sim time (docked) |
|---|---|---|---|---|---|
| Autonomous (camera) | random parking | 5 | **5/5** | 0 | 16.9 s |
| Autonomous (camera) | random parking + car moves | 5 | **5/5** | 0 | 17.1 s |
| Pre-programmed (fixed nominal coordinates) | random parking | 5 | 0/5 | 0 | – |
| Pre-programmed (fixed nominal coordinates) | random parking + car moves | 5 | 0/5 | 0 | – |

The pre-programmed controller jammed on the socket rim every time, at 12–44 N peak force, then backed out and aborted thanks to the force limits. The autonomous agent docked in all 10, with zero contact force in 10/10 and retries needed in 1/10 (trial 7: two retries after a car move). Ten trials is a small sample, so this is not a reliability figure for the real world.

**Not demonstrated** (no sensor or mechanism for it, so not claimed): obstacle detection in the arm's path (Level 7), glare and lens effects, recognising the inlet without the marker.

**Honest limits of the perception:**
- **The inlet is found by a fiducial marker on the car, not by recognising the bare CCS2 inlet.** The marker-to-socket transform is a fixed calibration of the inlet geometry. A learned inlet detector (the teammate's model) can replace `PortPerception._estimate` behind the same `PortEstimate` interface.
- **The final few millimetres of insertion are blind.** The marker leaves the camera's view, so they rely on the last estimate plus force sensing.
- Rendering is clean (no lens distortion); `pixel_noise` can add noise. Lighting changes have not been evaluated yet.
- The drive-in mission still docks the base on true odometry. Camera perception starts once the robot is parked.
