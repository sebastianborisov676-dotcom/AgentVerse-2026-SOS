# Local operator control panel

The panel runs the real MuJoCo model locally, with SAI's robot assets, connector,
elastic cable and insertion controller. It uses a browser for controls and JPEG views;
the browser does not implement or substitute the physics.

## Launch

```bash
cd /home/sebastianborisov676/Events/AgentVerse
.venv/bin/python scripts/run_control_panel.py
```

Open **http://127.0.0.1:8765** in the WSL host's browser. Use `--port 8766` if the port is
already occupied. Core dependencies plus Pillow are required; optional panel dependencies
are listed in `requirements-panel.txt`. The existing workspace already has them installed.
`MUJOCO_GL=egl` is selected before importing MuJoCo; explicitly set `MUJOCO_GL=glfw` or
`osmesa` if required by another machine. EGL software rendering works in this workspace;
GPU device permission warnings can appear while Mesa falls back to software rendering.
A camera failure is surfaced in the viewport rather than replaced with a mock image.

The server binds only to loopback. MuJoCo stepping, commands and rendering are serialized
on one worker thread so an OpenGL context is never shared between HTTP handler threads.
The browser polls telemetry and rendered frames. Wall-clock performance depends on the
machine; the display is interactive but a guaranteed real-time factor is not claimed.

## Layout and operating workflow

- **Top:** level 1–11 (Level 11: Master Challenge), scenario seed (default 42), placement mode, randomized-start toggle,
  moving-car challenge toggle, Build scenario and New seed.
- **Middle left:** obstacle type, X/Y sliders, rotation, manual obstacle list, add/move/remove,
  copy current layout into manual editing, save and load layout JSON.
- **Middle right:** MuJoCo 3D viewport, external orbit camera, robot RGB, metric depth,
  connector camera, and live position/force/perception/contact/docking/insertion telemetry.
- **Bottom:** Start, Pause, Resume, identical Restart, car safety-test request, Save recording,
  Return to live, visual playback timeline, checkpoint restore, and A/B comparison.

Start with Level 1 / seed 42, build, then Start. Pause freezes simulation time and physics;
it does not simulate braking while time continues. Resume continues the same state.
Restart archives the old attempt and rebuilds identical settings and seed. New seed
increments the seed and builds a separate experiment.

Manual/hybrid editing is enabled from Level 5. In random mode, **Edit current layout**
copies the realized obstacles into manual mode without silently deleting them. Select an
obstacle, adjust its sliders/type, then Move/rotate; Add and Remove also apply actual
changes. Edits rebuild a paused scene as a new attempt, rather than teleporting an obstacle
into a robot or cable during contact. Invalid edits retain the previous valid simulation
and show the reason. Seeded random objects in hybrid mode are regenerated from the same
seed; the manual list contains only the explicitly supplied objects. To edit any generated
object individually, first copy the complete layout into manual mode.

Save layout writes a uniquely named JSON file in `outputs/panel/layouts/` and downloads
it in the browser. Loading uses the browser's file selector. A/B comparison shows both
configurations and final recorded outcomes: keep the seed fixed, change one obstacle and
compare the archived and current attempts. It does not falsely label a multi-variable
edit as a controlled one-variable experiment; inspect the two configurations.

## Seeded starts and fixed physical cable

For Levels 1–10, nominal longitudinal start distance is `2.50 + 0.04*(level-1)` metres: 2.50 m at Level 1
and 2.86 m at Level 10. The modest progression is deliberate because the physical cable
is limited. Seeded variations are uniform ±0.5 m longitudinally, ±0.3 m laterally about
Y=0.5, and ±5 degrees about a 15-degree heading. The exact requested/actual start and
nominal distance are displayed and recorded. The scenario seed independently seeds the
layout and vehicle maneuver, so changing one obstacle does not change the robot pose.
Level 11 uses a narrower cable-feasible start envelope described in [Master Challenge](master_challenge.md).

Every panel scenario has the same **4.545351328 m, 64-segment cable**. The segment rest
lengths, total cable mass and elasticity do not increase for longer requested starts.
For shorter starts, additional cable is routed as floor slack before simulation. If the
current routing needs more physical length, the requested scenario is rejected. This is
a conservative feasibility check for the implemented router, not a proof that no other
rope route could ever fit. No invisible extension, cable constraint disablement or
teleportation is used to make a requested distance work.

Ground-truth geometry is permitted for reset validation only: course connectivity, start
clearance, initial robot/cable intersections, contacts during settling, finite physics
and post-settling cable loads. It is not handed to the navigation policy. Some seeds or
manual edits are infeasible; the interface reports the reason instead of silently choosing
a different pose or removing objects. The old batch baseline's behavior is unchanged;
the fixed-length/random-start settings are opt-in panel scene parameters.

## What the autonomous controller can observe

`Sensors.scan` uses actual `mj_ray` intersections, not obstacle coordinates:

- 81 rays over a 240-degree horizontal field at each of three heights (0.04/0.20/0.45 m).
- Maximum obstacle observation range 2.2 m; nearer surfaces occlude farther ones.
- Generic hit cells at 0.12 m resolution, with 0.07 m surface-location uncertainty.
- Hit memory expires after three simulated seconds. There is no semantic box/cone lookup.
- Scans update every 0.2 simulated seconds during driving.

The policy receives only these observed surface cells, base proprioception, cable/contact
measurements and an ideal fiducial pose measurement. It knows the fixed charging-station
floorplan/docking calibration. The previous `CourseEnv`/batch runner remains a separate
privileged-map baseline; its results must not be advertised as sensor-only navigation.

The fiducial sensor is an explicitly idealized substitute for image-based pose estimation:
no YOLO, neural detector or RGB/depth reconstruction is claimed. In panel scenes a visible
black/white marker above the inlet has a calibrated rigid offset to the charging port.
Its measurement is gated by 4.5 m range, the robot RGB camera's actual pose/frustum and a
line-of-sight ray. Robot RGB and metric depth are actual MuJoCo renders, independently
available to the operator; the policy currently uses rays rather than processing those
images. Marker pose is noiseless and perfectly calibrated when visible. Outside arm
motion it is sampled at 20 Hz; during arm motion it is checked every 2 ms physics step.

New observations that obstruct the remaining path trigger a route revision using the
observed cells. The controller tries smooth approaches and then grid A*, with an explicit
final approach. Surface-hit inflation differs from whole-object radius inflation; it
does not treat an observed surface point as an entire large box. Unknown space is treated
optimistically, so this is a local-navigation prototype, not certified exploration.
Limited range, coarse angular sampling, occlusion, thin obstacles, stale cells and cable
snagging can still cause safe stops. Complete articulated/cable swept-volume planning,
tracking of moving obstacles and automated recovery from every blocked route are future work.

## Base and moving-car mechanics

Driving commands follow the existing unicycle-style controller on planar velocity servos.
The panel adds a low-speed **XY docking trim** using the same actuators (up to 15 mm/s per
axis, with acceleration bounds), analogous to the existing planar brake approximation.
This removes cable-drag lateral drift during the final parking stage. It is an explicit
holonomic simulation approximation, **not validated differential-drive/wheel traction
physics**. The original driver and its defaults are unchanged. The panel uses tighter
parking tolerances and a slower approach; insertion tolerances and forces remain unchanged.

Levels 9–11 optionally use an explicitly movable vehicle with X/Y slide joints and a
yaw joint, a 1200 kg chassis and force-limited damped position servos. Levels 9 and 11
command a four-second quintic forward translation of 0.10–0.14 m along the vehicle's
heading, with no commanded yaw change. Level 10 retains its earlier 0.1–0.2 m translation,
small lateral displacement and 2–5 degree yaw challenge around the car centre. This is
constrained planar vehicle dynamics, not a tyre/suspension/steering model. The actor
starts parked and moves once when the robot is about 0.45 m from docking. Level 11 also
requires a sensor-driven avoidance replan before its actor can trigger. Turning Car
challenge off leaves the vehicle held at its original pose.

The policy detects marker motion, brakes, inhibits insertion and waits until measured
linear/angular speeds are below 0.001 m/s and 0.001 rad/s for at least 0.6 s. Those are
explicit prototype stationarity tolerances, not a claim of mathematically zero velocity.
It then updates the docking transform and socket estimate from the marker and replans.
An actuator-busy interlock additionally prevents insertion during commanded vehicle motion.

The Move car / safety test button respects a connector approach/contact interlock. If
requested during arm operation, the car stays stationary and the **existing insertion
controller's retraction** is invoked. Independently observed unexpected port motion during
arm operation freezes the actor and invokes the same recovery. Recovery ends in a safe
hold requiring an operator restart; it does not blindly retry insertion. Lost marker
observations during arm motion cause an immediate hold. No static body is teleported while
contacting the connector. All cable safety thresholds stay at 60 N tension / 45 N strain
relief; load checks remain active during waiting and recovery.

## Recordings, visual playback and controller checkpoints

Each attempt records at approximately 10 Hz (plus final and checkpoint frames): full
MuJoCo integration states, configuration, realized obstacles, actual starting pose, observed
ray points and marker measurements/timestamps, controller events/routes, contact and force
telemetry. Commands include user actions and their simulated timestamps. Files contain
MuJoCo/NumPy versions and a Python source hash.

`outputs/panel/runs/attempt_<id>.npz` contains numeric states and JSON metadata;
the matching `.mjb` contains the compiled scene including meshes. Loading uses no pickle.
The timeline restores recorded states and re-renders the selected camera from the saved
model. This is **visual playback**, not a rerun of the controller. Camera/route annotations
may be sampled differently; pixel-identical playback across graphics drivers is not claimed.
Recordings saved by this server remain selectable after restarting it.

For retained in-process attempts, a checkpoint about every two simulated seconds stores
full `MjData`, controller/sensor/vehicle memory, and accumulated telemetry. Restore chooses
the nearest preceding checkpoint, shows its actual time, and branches to a new attempt,
preserving the original future. These Python/controller checkpoints are deliberately not
serialized; disk-only recordings have visual playback but no controller restore.
The six most recent cached records can retain live checkpoints, including the original
source after branching or changing scenes. Older evicted records remain on disk as visual
playback. The checkpoint selector lists resumable live times; terminal checkpoints are
excluded. Restore rebuilds the source scene context and creates a new branch without
changing the original attempt. The tests verify exact same-machine initial/prefix states and one tested checkpoint
continuation; this is not a blanket claim of cross-platform or arbitrarily long bitwise
repeatability. Real-world camera noise, latency and hardware determinism are not modelled.

## Validation commands

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_control_panel.py
node --check src/autocharge/panel/static/app.js
```

The HTTP check starts its own ephemeral loopback server and shuts it down. It verifies
initialization, start/pause/resume, manual editing, A/B comparison, layout persistence,
identical restart, all five rendered views, visual playback and checkpoint restore.
It writes JPEGs and `outputs/panel_http_check/result.json`. This is HTTP/renderer integration
verification, not browser automation or a claim of a completed visual browser usability test.

New targeted tests cover seeded start envelopes, fixed cable length and infeasible poses,
same-seed initial/prefix repeatability, transactional manual edits, ray range/occlusion,
observation-driven route changes, continuous joint-driven car movement, motion interlocks,
unexpected port-motion recovery, disk visual-state restoration and complete checkpoint
continuation. Full mission tests exercise Level 1 and the Level 9 moving-car challenge at
seed 42. The earlier control-panel revision evaluated all ten levels at seeds 42 and 43
using the panel pipeline with two isolated workers. See [the revision report](operator_validation.md)
for actual Chromium clicks, outcomes and repeatability comparisons; the earlier
privileged-map results remain a separate baseline.

## Earlier verified results (before route-crossing generation)

- Before implementation: **33 tests passed**.
- Combined regression suite: **54 tests passed** (including all prior tests).
- After the final camera/marker calibration: **21 panel tests passed**, including both
  full missions, target-motion recovery and exact checkpoint/prefix assertions.
- Default Level 1 / seed 42: seated at **36.846 s**; peak tension **14.43 N**,
  peak strain relief **16.86 N**, zero load violations or unsafe/obstacle contact episodes.
- Level 9 / seed 42: detected the moving port, stopped, replanned once and seated at
  **41.002 s**; peak tension **13.33 N**, peak strain relief **15.80 N**, zero load
  violations or unsafe/obstacle contact episodes. Both used the same fixed-length cable.
- Same-seed prefix equality was checked over 0.1 simulated seconds; complete checkpoint
  continuation equality was checked over 0.08 seconds. Longer/cross-platform exact
  determinism is not asserted.

The earlier revision's full-mission recordings and summary are in
`outputs/panel_verified_runs/` (`results.json` plus paired NPZ/MJB files under `runs/`).
To review these verified attempts directly in the panel:

```bash
.venv/bin/python scripts/run_control_panel.py --output outputs/panel_verified_runs
```

Select the saved attempt under Attempt B, then move the timeline. Restore checkpoint is
unavailable for disk-only saved attempts; this is explicitly visual playback.

## Current editor and batch revision

See [operator validation](operator_validation.md) for request-level editor diagnostics,
scene-tagged viewport frames, click-to-place with a 0.15 m projected-edge clearance,
ready-to-load layouts, the unchanged connector camera and separate operator side view,
retained checkpoint branching, and the shared panel/headless runner.

Save active layout saves only geometry already accepted by the physics scene, plus its
seed and relevant scenario flags. It does not silently build or save an invalid pending
draft. Active and draft counts are shown separately, and rejected builds retain the
previous active scene.

See [Master Challenge](master_challenge.md) for route-intersecting generation, the forward-car actor and verification details.

The latest requested Level 6, Level 9 on/off and Level 11 verification is reported in [Master Challenge](master_challenge.md). Earlier success rates do not apply to the new route-crossing generator.
