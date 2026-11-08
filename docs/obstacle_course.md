# Obstacle course and training baseline

All work runs locally. The original untracked `training/levels.py` is preserved unchanged;
`course.py` consumes its ten level definitions and obstacle sizes. Levels progress from
empty, box, cones and barriers through mixed layouts, stones, slopes and a 16-obstacle mix.
Every configured type appears in a random course; seeded position and angle sampling is
independent of global random state. Large objects are packed first without overlapping
circumscribed footprints. Placement can fail explicitly if manual geometry leaves no room.

```bash
.venv/bin/python scripts/run_autocharge.py --level 2 --seed 0 --viewer
.venv/bin/python scripts/run_course_batch.py --engine legacy --levels 1 2 3 4 5 6 7 8 9 10 --seeds 0 1 2
.venv/bin/python -m pytest -q
```

Batch JSON includes exact layouts, seeds, software versions, events, terminal phase,
travel distance, warnings and truncation. Exit code 2 means at least one episode failed;
a safe stop is not counted as success. Reproduction assumes the same dependencies and
platform; cross-platform bit-identical floating-point trajectories are not guaranteed.

From Level 5, pass `--placement manual --manual-layout layout.json` to the demonstration,
or `--mode manual --manual layout.json` to the batch runner. Hybrid adds the level's full
random count to the supplied obstacles. Random mode rejects supplied manual obstacles.
Example layout (angles in degrees, size in metres):

```json
{"obstacles": [{"kind": "box", "x": -1.7, "y": 1.7, "size": 0.15, "angle": 30}]}
```

Course centres lie in x [-3.8, 0], y [-4.5, 4.5]. Start and docking regions are protected.
Manual geometry is validated for known types, finite coordinates, size, bounds and overlap.
The larger course width accommodates the original half-metre slopes at high levels.

## Physics and approximations

MuJoCo simulates static obstacle contact, robot articulated dynamics, connector/socket
contact, and the existing elastic cable with attachment constraints. Boxes and barriers
are box geoms; stones are ellipsoids; cones are 12-sided convex meshes; slopes are convex
wedge meshes. Obstacles are fixed, rather than movable or breakable.

The existing base remains planar x/y/yaw velocity servos with commanded unicycle motion;
there are no wheel/terrain dynamics, vertical travel, roll or pitch. Slopes and stones are
avoided, not climbed. This is not a validated rough-terrain vehicle simulation.

Navigation uses privileged known obstacle positions, grid A*, conservative circular
footprints and swept segment checks, with a straight final approach. The existing smooth
route is retained when clear; smooth bypass candidates precede the grid fallback. The robot footprint radius and braking guard are engineering
approximations, not full articulated or cable swept-volume planning. Pure pursuit can cut
corners; the clearance guard stops rather than replanning. Cable snagging, blocked routes,
tracking failures and timeout can prevent success, especially on higher levels. Random
layouts preserve a geometrically feasible route and final docking corridor in the default scene.
This is not a guarantee of dynamic feasibility for arbitrary manual layouts or changed robot/cable parameters. The charger footprint is included;
vehicle and port geometry remain protected by the docking corridor and existing physics.

## RL extension

`CourseEnv.reset(seed, alignment_offset_mm)` returns a state observation and metadata;
`step(action=None)` returns observation, reward, terminated, truncated and info. None runs
the baseline; [forward speed, yaw rate] applies bounded navigation actions while retaining
cable checks and an obstacle guard. After docking the existing insertion controller runs.
Actions are simulator velocity commands, not wheel torques. Observations contain privileged
base state, variable-length obstacle geometry, alignment errors, cable loads and phase.
This is a lightweight research API, not yet a Gymnasium environment. Reward is initial
progress shaping plus terminal bonuses/penalties. No policy has been trained. An adapter
will need padded/type-encoded obstacles, observation spaces, policy evaluation and richer
rewards. Alignment offsets exercise existing recovery logic; learned arm/alignment actions,
vision, port pose randomization and domain randomization remain future work.

## First evaluation (8 November 2026; superseded by reliability evaluation below)

- Full pytest suite: 29 passed (15 existing, 14 added).
- Ruff checks on the new modules, runner and tests: passed.
- All ten levels, seeds 0 and 1, 0.1-second smoke horizon: 20 episodes,
  no construction errors or MuJoCo warnings. These are smoke checks, not mission successes.
- Full missions on levels 1, 2, 5 and 10, seeds 0 and 1: 4/8 seated.
  Both Level 1 and Level 2 seeds seated. Level 5 stopped on cable tension / strain-relief
  limits; Level 10 rejected the blocked final docking corridor. No MuJoCo warnings.
- Reports: `outputs/course_final.json` and `outputs/course_all_levels_smoke.json`
  (local ignored output artifacts). Earlier development reports are superseded by these.


## Reliability improvements and full evaluation

The generator now reserves an entire feasible path with 0.65 m obstacle inflation and a
straight final docking corridor from (-0.8, 0) to the origin. It also rejects obstacle
footprints against a padded polygon enclosing the measured nominal cable sweep, including
its initial route. The charger retains its existing 0.55 m planner clearance separately
from the obstacle inflation; the larger obstacle inflation must not make the fixed,
validated docking pose appear blocked by the charger.

All counts, types, original sizes, and static contact-enabled geometry are retained.
At least one obstacle is sampled near the route boundary (0.67–0.80 m centreline-to-footprint
clearance), rather than sending all obstacles to the far edges. Large slopes and barriers
remain present at high levels. Rejection sampling is deterministic; there is no simulation
retry or selection of successful trajectories after evaluation. The untracked `levels.py`
is unchanged. New generator seeds intentionally produce different layouts than the first
evaluation; every report retains the exact evaluated coordinates.

These are conservative deployment layouts, rather than forced-detour mazes. All twenty
validated episodes follow the nominal route. The cable sweep polygon is a measured XY
approximation for the default 64-segment cable, start pose, carry pose and mission; it is
not a general rope planner or a formal safety certificate. If these parameters change,
re-measure the envelope and revalidate. Manual layouts can still cause cable snags or
tracking failures; unsafe reset intersections are reported and block the mission.

### Root causes in the saved original layouts

| Original case | Evidence | Classification and action |
|---|---|---|
| Level 5, seed 0 | Six initial cable/obstacle contacts, up to 24.5 mm penetration into barrier 0; load persists after a 1 s hold | Invalid initial geometry. Protect the initial and moving cable workspace. |
| Level 5, seed 1 | No initial obstacle contact; cable catches barrier 0 during driving, then strain-relief limit fires | Mechanical snag on the chosen route. Base-only planning did not account for the trailing cable. Protect its sweep; arbitrary detours still require rope planning. |
| Level 7, seed 0 | Fifteen initial cable/slope contacts, up to 90.7 mm penetration; settling tension peaks around 801 N and is still 189 N at 1 s | Invalid reset geometry causing large constraint forces. Waiting longer is not a fix; reject the overlap. |
| Blocked docking corridors | Start and goal discs were clear while the connecting final segment was obstructed | Generator/planner contract failure. Reserve the complete approach and witness path. |

The original ~225 N startup observation is consistent with this deeply intersecting cable;
replaying the saved layout gives approximately 221 N at 0.5 s and 189 N at 1 s. Exact values
vary with the final forward-dynamics evaluation; both are far over the unchanged 60 N limit.
The high forces in penetrating initial geometry are solver responses, not credible real
cable strength measurements. The historical safety stops were correct protective actions.
No connector geometry, insertion control, drive load threshold or insertion load threshold
was changed for this work.

Diagnostic traces are in `outputs/course_diagnosis.json` and
`outputs/level7_seed0_contact_replay.json`. Saved layouts can be investigated independently
of generator revisions:

```bash
.venv/bin/python scripts/inspect_course_contacts.py outputs/course_all_levels_smoke.json --level 7 --seed 0 --settle 1 --max-time 2
```

### Contact and load telemetry

`telemetry.py` reads MuJoCo contacts at every physics step, using `mj_contactForce`. Only
solver-active normal forces above 0.001 N count. This is a telemetry filter, not a changed
controller force limit. Reports contain contact samples, peak normal forces, maximum
penetration, and contact-pair episodes (onsets). Multiple manifold points on the same
geometry pair in the same step count as one active pair episode. Separation followed by
recontact starts another episode; counts are not estimates of discrete physical impacts.
Cable/floor contacts are grouped under the cable, rather than counted per cable vertex.

Categories distinguish wheel/floor, cable/floor, connector/socket, robot/obstacle,
cable/obstacle, other obstacle contacts, cable/robot, robot self-contact and unsafe robot
contacts with the environment. Connector/socket contact is expected during insertion and
reported separately; it is not counted as an unsafe robot collision. Unsafe robot contacts
stop the course mission. Settling contacts and cable violations are recorded in a separate
section, including initial penetrations, so they are not hidden by resetting statistics.
Cable reports include peak edge tension, peak strain-relief load, violation samples,
violation episodes and the first violation with its threshold. Docking outcome, final
pose error, truncation and failure category/reason are separate fields.

Wheel/floor counts are zero in these runs because the original planar base excludes
world/base contact. The wheels are visual/collision geometry without wheel dynamics.
Cable/floor contact is expected and occurred in every episode. Successful insertion with
clearance can also produce zero connector/socket contacts.

### Full mission results (seeds 0 and 1)

Every episode had a 70 s simulated-time limit and a 180 s wall-clock limit. Reports are
saved after each episode. Both limits include failures rather than silently retrying them.

| Level | Obstacles | Seated | Robot/obstacle episodes | Cable/obstacle episodes | Unsafe robot episodes | Cable-load violations | Docked |
|---|---|---|---|---|---|---|---|
| 1 | None | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 2 | 1 box | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 3 | 4 cones | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 4 | 4 barriers | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 5 | 5 boxes/cones/barriers | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 6 | 12 stones/boxes | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 7 | 5 slopes/boxes | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 8 | 7 slopes/stones | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 9 | 12 stones/slopes/barriers | 2/2 | 0 | 0 | 0 | 0 | 2/2 |
| 10 | 16, all five types | 2/2 | 0 | 0 | 0 | 0 | 2/2 |

There were no mission failures, timeouts, load violations or MuJoCo warnings. Settling also
had zero obstacle/unsafe robot contacts and zero cable-load violations. Each mission had
one continuous cable/floor pair episode; settling has its own cable/floor episode record.
Peak operating edge tension was 12.39 N; peak strain-relief load was 14.83 N. Peak settling
tension was 16.49 N. Final base error was at most 7.53 mm and 0.181 degrees. Missions finished
at approximately 30.416 s simulator time (including 0.5 s settling).

Primary report: `outputs/course_reliability_full.json`, combining the low/high level batch
partitions with their exact layouts and software versions. This supersedes the earlier
smoke and partial full-mission reports. To repeat the entire evaluation in one batch:

```bash
.venv/bin/python scripts/run_course_batch.py --engine legacy --levels 1 2 3 4 5 6 7 8 9 10 --seeds 0 1 --max-time 70 --wall-time 180 --output outputs/course_reliability_repeat.json
```

Validation: 33 tests passed (all 15 original tests plus 18 course tests); the 18 course
tests passed again after formatting. New telemetry tests check actual nonzero MuJoCo
contact forces and separation/recontact counting, match cable loads against the existing
metrics, and reproduce the original Level 7 invalid initialization. Seeded generator tests
check counts/types, route/corridor clearance, cable workspace and a nearby obstacle for
all ten levels at seeds 0, 1 and 42. Ruff checks and Git whitespace checks pass.

Remaining physical limitations are unchanged: planar servo motion, no slope climbing,
privileged static maps, approximate footprints and a nominal cable workspace. This result
establishes reliable completion for the tested layouts, not robust navigation through
arbitrary mazes or a validated terrain/mechanical model.


The current batch runner defaults to the operator panel's sensor-based pipeline. The
commands above explicitly select the historical privileged-map baseline. For current
parallel panel experiments and actual browser validation, see
[operator validation](operator_validation.md).
