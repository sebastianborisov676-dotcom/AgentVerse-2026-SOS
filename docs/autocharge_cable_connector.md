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
python -m pytest tests                                   # 13 validation tests (~2 min)
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

## Validation results (all measured, `pytest tests`: 13 passed)

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

## Next step: camera-based autonomy

The stability checks above pass, so perception can be added. Cameras `connector_cam` (on the
connector, looking along the plug axis), `socket_view` and `overview` are already in the scene.
Perception only has to produce `socket_estimate=(position, rotation)` for
`InsertionController`; the tests show the system tolerates ≤ 3 mm error and aborts safely above
that.
