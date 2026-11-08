# Master Challenge and route-intersecting courses

All changes are local and uncommitted on `feature/charger-alignment`.

## Shared execution pipeline

The browser and default batch engine both use `Session.configure -> build_scenario`
and `Session.tick -> AdaptiveMission`. Level 11 is selectable in both. The legacy
privileged-map engine rejects Level 11 because it cannot demonstrate sensor-based
moving-target recovery.

Random and Hybrid avoidance levels place a generated obstacle on the empty-map route
that the robot would initially follow. The actual footprint contains a route sample,
not merely a surrounding safety radius. The scenario builder checks a detour and the
protected final corridor, then checks the real reset robot, connector and routed cable.
Manual entries are preserved. The remaining generated obstacles retain the level's
counts and types and keep the validated detour clear. Invalid builds retain the active
scene and report their geometry, path or cable reason in the panel.

Only scenario validation uses obstacle ground truth. Navigation receives finite-range
ray observations: 2.2 m, 240 degrees, 81 rays at three heights. Port pose is an ideal
calibrated visible-fiducial observation with range, camera FOV and occlusion checks;
this is an approximation rather than image-based computer vision. Navigation can use
the last observed port during distant avoidance turns. Final approach and arm motion
require a fresh visible port. Obstacle positions and actor destinations are not used
as navigation targets.

The navigation wrapper tries smooth bypasses that remain offset until the final gate,
then checked staging waypoints and grid paths. It brakes and turns before sharp path
changes. Its initial bypass-side preference is a simple cable-side heuristic from the
robot's own starting pose, not a coupled rope planner. The original driver, its safety
checks, insertion controller and tolerances are unchanged. The existing planar XY
servos and final docking trim remain approximations to wheel/traction dynamics.

## Vehicle actor

Levels 9 and 11 use a one-shot, four-second quintic forward translation of 0.10–0.14 m
along the vehicle's heading. Whole-body slide joints move the body and child socket
together using the existing position actuators; no static body is teleported. It is a
planar actuated vehicle, not a tyre/traction or rotating-wheel simulation. Level 10
retains its earlier translation/yaw challenge. With Car challenge disabled, automatic
motion is disabled and the vehicle is held at its original pose.

The actor starts stationary and triggers within 0.45 m of the original base docking
target, before insertion. Level 11 additionally requires an actual sensor avoidance
replan first. Actor command state provides an insertion interlock; measured port
motion causes braking. After measured stability for 0.6 s, the controller updates its
docking target and socket alignment estimate from the sensor and replans. Existing
contact, insertion-motion recovery and 60 N / 45 N cable safeguards remain active.

## Level 11 scene

The compact start is deliberately bounded by the fixed cable, not by an arbitrary
maximum distance. Its nominal base pose is (-2.24, 0.62, 12 degrees); seeded positions
use X in [-2.29, -2.24], Y in [0.57, 0.67], and heading in [10, 14] degrees. Nominal
start (randomization off) uses exactly that nominal pose. Cable length remains
4.545351328240237 m, with the original routing and material/attachment model.

The primary route obstacle is a low 0.10 m cube. The course also includes the existing
stone (0.06 m horizontal radius and height) and slope (1 m footprint, 0.18 m rise)
behaviour. The low box is deliberate: larger boxes and large detours exposed tether
wrapping and robot/cable entanglement in this model. This is a repeatable demonstration
of two decisions, not evidence that the robot can negotiate arbitrary large obstacles
with a trailing charging cable. The experimental alternate initial cable routing was
removed; no cable rest length, material or connector behaviour was changed.

## Verification artifacts

Final results, the selected recording and repeat comparison are recorded separately
from the development runs. Development `outputs/master_*` runs include rejected
geometry, cable-safe aborts and unsuccessful planning attempts; they are not successful
demonstrations. Event timestamps identify obstacle detection, avoidance replanning,
vehicle motion, measured braking, updated docking target and seating/safe abort.
Disk NPZ/MJB recordings are visual playback; live controller checkpoint restoration
is available only for in-process checkpoints. Deterministic re-simulation is claimed
only for cases explicitly compared by outcomes, event times and final integration hash.

## Final mission results

Counts below are MuJoCo contact-pair episodes, not distinct impacts. Expected wheel/floor
and cable/floor contacts are excluded. All reported robot/obstacle, robot-self and unsafe
robot/environment counts are zero. Startup cable-limit violations are zero in every row.

| Level | Seed | Car | Avoidance | Motion recovery | Docking | Robot/obstacle | Cable/obstacle | Cable-limit episodes (tension / strain) | Reason |
|---|---:|---|---|---|---|---:|---:|---|---|
| 6 | 42 | off / inactive | yes | not requested | safe stop | 0 | 0 | 0 / 1 | cable load limit: motion inhibited |
| 6 | 43 | off / inactive | yes | not requested | seated | 0 | 119 | 0 / 0 | — |
| 9 | 42 | on | yes | yes | seated | 0 | 159 | 0 / 0 | — |
| 9 | 42 | off / inactive | yes | not requested | seated | 0 | 159 | 0 / 0 | — |
| 11 | 42 | on | yes | yes | seated | 0 | 50 | 0 / 0 | — |
| 11 | 43 | on | yes | not triggered | safe stop | 0 | 15 | 0 / 1 | cable load limit: motion inhibited |
| 11 repeat | 42 | on | yes | yes | seated | 0 | 50 | 0 / 0 | — |

Five of seven episodes completed, including the repeated demo; four of six unique
configurations completed. Level 11 seed 42 completed twice; seed 43 did not reach the
vehicle trigger. The selected demo is repeatable, but robustness across arbitrary seeds
has not been established.

Both failed requested episodes are **simulated cable/contact constraints**: Level 6 seed 42
stopped after 19.010 s at the strain-relief threshold (peak 45.072 N); Level 11 seed 43
stopped after 29.830 s (peak 45.061 N). Those stops are expected safety behaviour. The
underlying snagging/routing problem needs better coupled cable planning and contact
modelling; it is not resolved by raising limits. The earlier moving-target planning
failure near the pedestal was fixed by restoring the original 0.50 m planning clearance
while retaining the driver's independent 0.43 m braking-path safety check.

## Verified judge demonstration

Open the already-running updated panel at **http://127.0.0.1:8772**. Existing panels on
other ports were not restarted or reloaded. To launch it again after that server stops:

```bash
cd /home/sebastianborisov676/Events/AgentVerse
.venv/bin/python scripts/run_control_panel.py --port 8772 --output outputs/master_demo
```

For a fresh live run select **Level 11 — Master Challenge**, **seed 42**, **Random**,
**Randomize start checked**, **Car challenge checked**, then Build scenario and Start.
No manual entries are required. Hybrid is supported and has a seeded/manual-preservation
regression test; the verified recording uses Random mode.

To review the saved run, select **attempt_1794134385345455365** in Attempt B. Its paired
NPZ/MJB files are in `outputs/master_demo/runs/`. Select External / orbit for the whole
sequence, or the connector operator view near seating. Disk playback has Restore disabled.

Judges can identify these two plan changes in **simulated seconds**:

1. **0.500 s:** actual sensor obstruction detection and avoidance replan (seek about **0.6 s**).
2. **33.250 s:** measured port motion triggers braking and insertion inhibition (seek about **33.4 s**).

The car starts its one-shot forward motion at **32.910 s**, travelling approximately
**0.111881 m**. At **37.750 s** the port is confirmed stationary and the target is updated;
the connector seats at **62.754 s**. The peak simulated strain-relief load is **43.117 N**,
so this demo has limited margin below the 45 N stop threshold. Cable/robot and
cable/obstacle contacts remain present and must not be described as collision-free physics.

Repeat the selected demo headlessly using the same pipeline:

```bash
cd /home/sebastianborisov676/Events/AgentVerse
.venv/bin/python scripts/run_course_batch.py --engine panel --levels 11 --seeds 42 \
  --mode random --workers 1 --max-time 90 --wall-time 300 \
  --output outputs/master_demo_judge.json
```

The two headless seed-42 runs matched exactly in outcome, all event timestamps, cable
and contact telemetry, obstacle positions, source hash and final integration-state SHA256.
That validates repeatability for this local software/assets/platform combination; it is
not a cross-platform determinism claim. Browser Build produced the same complete initial
integration state as the headless recording.

Artifacts:

- [Consolidated results](../outputs/master_verification.json)
- [Repeat comparison](../outputs/master_demo/repeat_comparison.json)
- [Browser control verification](../outputs/master_demo/browser/verification.json)
- [Avoidance viewport](../outputs/master_demo/browser/avoidance_view.png)
- [Braking viewport](../outputs/master_demo/browser/braking_view.png)
- [Updated target viewport](../outputs/master_demo/browser/updated_target_view.png)
- [Seated connector operator view](../outputs/master_demo/browser/connector_seated.png)


## Test results and changed files

**All 65 tests passed** across four isolated pytest groups (17 + 16 + 16 + 16).
The initial full run had 63 passes and one failure: its Level 9 mission test required
zero cable/obstacle contacts, an expectation of the earlier route-adjacent layouts.
The updated test retains zero robot/obstacle, self and unsafe-environment contacts and
zero cable-load violations; it additionally verifies avoidance -> vehicle motion ->
stationary target update -> insertion ordering. The final rerun covered every collected
test, including the new Hybrid/manual-preservation test. No runtime contact safeguards
or force limits were removed. Logs are `outputs/master_tests_group_0.log` through
`outputs/master_tests_group_3.log`, with [group results](../outputs/master_tests_final.json).
The initial failure log remains available as `outputs/master_regression.log`.

Python compilation passed. Actual Chromium checks passed for Level 11 selection, Build,
initial-state equality with headless execution, disk-record selection (including reopening
the page), timeline seeks, External/connector cameras and disabled disk Restore.
Whole requested mission outcomes and limits are in [the consolidated JSON](../outputs/master_verification.json).

Modified for this request:

- `src/autocharge/training/levels.py`, `src/autocharge/training/course.py`
- `src/autocharge/panel/scenario.py`, `adaptive.py`, `drive.py`, `session.py`
- `src/autocharge/panel/static/app.js`, `index.html`
- `scripts/run_course_batch.py`, new `scripts/verify_master_panel.py`
- `tests/test_panel.py`, `tests/test_course.py`
- `docs/control_panel.md`, new `docs/master_challenge.md`
- New local recordings, screenshots, test logs and experiment results under `outputs/`

The temporary cable-route experiment was reverted to the pre-request scene code.
Existing local changes in README, scene, mission, the original controller and other
training/panel files were preserved. No branch switch, commit, push, merge or PR occurred.
