"""Watch the autonomy tests run live: MuJoCo viewer + robot camera + decisions + verdict.

    python scripts/watch_tests.py                 # play every scenario in order
    python scripts/watch_tests.py car-moves flap  # play only these
    python scripts/watch_tests.py --list

Each scenario is the same experiment as an automated test. The verdict at the end comes from the
independent evaluator (physical socket pose), not from the robot. Close the viewer to stop.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import autocharge.perception as perception
from autocharge import build_model, reset_to_home
from autocharge.autonomy import AgentState, AutonomousAgent
from autocharge.evaluation import check_docking
from autocharge.overlay import robot_camera_overlay, status_panel
from autocharge.scenarios import CarMove, CarPose, Occlusion, ScenarioManager


@dataclasses.dataclass
class Show:
    key: str
    title: str
    proves: str
    expect: str  # "dock" | "no-dock" | "no-move" | "safe-stop"
    scenario: dict = dataclasses.field(default_factory=dict)
    max_time: float = 40.0
    poison: bool = False
    calib_error: bool = False
    move_when_engaged: bool = False


SHOWS = [
    Show("nominal", "1. Standard docking from the camera", "perception-driven action", "dock"),
    Show("random-park", "2. Car parked 8 cm sideways, 4 cm higher, turned 6 deg", "not tied to one pose", "dock",
         {"park": CarPose(0.03, -0.08, 0.04, -6.0)}),
    Show("car-moves", "3. Car rolls 3 cm + turns 3 deg during approach", "adaptation: re-plans", "dock",
         {"moves": (CarMove(at=3.0, to=CarPose(0.0, 0.03, 0.0, 3.0)),)}),
    Show("mid-block", "4. Camera view blocked mid-approach", "pauses when blind, resumes", "dock",
         {"occlusions": (Occlusion(1.5, 3.0),)}),
    Show("blindfold", "5. Camera view blocked from the start", "never moves without perception", "no-move",
         {"occlusions": (Occlusion(0.0, 1e9),)}, max_time=8.0),
    Show("dim", "6. Lights dimmed to 15%", "robust perception", "dock", {"light_scale": 0.15}),
    Show("dark", "7. Lights at 3% (too dark)", "refuses when unsure", "no-move", {"light_scale": 0.03}, max_time=8.0),
    Show("flap", "8. Charging flap closed", "no false success", "no-dock", {"flap_closed": True}, max_time=75.0),
    Show("engaged-move", "9. Car moves while plug is engaged", "safe stop, no chasing", "safe-stop",
         move_when_engaged=True),
    Show("poison", "10. Simulator answer key corrupted by 2 cm", "never reads hidden ground truth", "dock",
         poison=True),
    Show("calib-error", "11. Camera calibration off by 8 mm", "actions follow perception", "no-dock",
         calib_error=True, max_time=30.0),
]


def verdict(show: Show, agent, model, data, plug_moved: float) -> tuple[bool, str]:
    v = check_docking(model, data)
    if show.expect == "dock":
        return v.docked and agent.state == AgentState.SUCCESS, f"checker: {v.reason}"
    if show.expect == "no-dock":
        return not v.docked and agent.state != AgentState.SUCCESS, f"checker: {v.reason}; robot: {agent.state.value}"
    if show.expect == "no-move":
        return agent.ctrl is None, f"robot never planned a move (plug moved {plug_moved * 1000:.0f} mm by sagging)"
    return agent.state == AgentState.SAFE_STOP, f"robot: {agent.state.value}"


TRAIL_NOW = np.array([1.0, 0.55, 0.1, 1.0])
TRAIL_PREV = np.array([0.6, 0.6, 0.6, 0.8])


def draw_overlays(v, ghost, trail_now, trail_prev) -> None:
    """Translucent 'standard bay' car + inlet, and the plug's path (current orange, previous grey)."""
    scn = v.user_scn
    scn.ngeom = 0

    def add(gtype, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    for size, pos, mat, rgba in ghost:
        add(mujoco.mjtGeom.mjGEOM_BOX, size, pos, mat, rgba)
    for trail, rgba in ((trail_prev, TRAIL_PREV), (trail_now, TRAIL_NOW)):
        for a, b in zip(trail[:-1], trail[1:], strict=False):
            if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-4:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                rgba.astype(np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, a, b)
            scn.ngeom += 1


def ghost_geometry(model, data) -> list:
    """Where the car and inlet sit in the standard bay (captured before the scenario moves the car)."""
    out = []
    for name, rgba in (("car_body", (0.1, 0.9, 0.3, 0.12)), ("inlet_backplate", (0.1, 1.0, 0.3, 0.55))):
        g = model.geom(name).id
        out.append((model.geom_size[g].copy(), data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy(),
                    np.array(rgba)))
    return out


def banner(show: Show, width: int, result=None) -> np.ndarray:
    h = 74
    img = np.full((h, width, 3), (60, 40, 20), np.uint8)
    cv2.putText(img, show.title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(img, f"proves: {show.proves}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    if result is not None:
        ok, why = result
        col = (40, 180, 40) if ok else (40, 40, 220)
        cv2.rectangle(img, (width - 110, 8), (width - 10, 40), col, -1)
        cv2.putText(img, "PASS" if ok else "FAIL", (width - 92, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, why[:70], (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="scenario keys to play (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for s in SHOWS:
            print(f"{s.key:14s} {s.title}")
        return 0
    shows = [s for s in SHOWS if not args.only or s.key in args.only]

    model, _ = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    from mujoco import viewer as mj_viewer

    results = []
    trail_prev: list = []
    with mj_viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.3, -0.35, 0.7]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.1, 60.0, -22.0
        for show in shows:
            if not v.is_running():
                break
            sid = model.site("socket_face").id
            site0 = model.site_pos[sid].copy()
            marker0 = perception.MARKER_POS_S.copy()
            with v.lock():
                reset_to_home(model, data)
                ghost = ghost_geometry(model, data)  # standard bay, before the scenario moves the car
                if show.poison:
                    model.site_pos[sid] = site0 + np.array([0.02, 0.02, 0.0])
                if show.calib_error:
                    perception.MARKER_POS_S = marker0 + np.array([0.008, 0.0, 0.0])
                sc = ScenarioManager(model, data, **show.scenario)
                mujoco.mj_forward(model, data)
                agent = AutonomousAgent(model, data)
                plug0 = data.site_xpos[model.site("plug_face").id].copy()
            moved = False
            end_at = None
            result = None
            trail: list = []
            plug_site = model.site("plug_face").id
            while v.is_running():
                t0 = time.time()
                with v.lock():
                    for _ in range(int(round((1 / 30) / model.opt.timestep))):
                        if (show.move_when_engaged and not moved and agent.ctrl is not None
                                and agent.phase == "insert" and agent.ctrl.engaged(data)):
                            sc.moves = [CarMove(at=float(data.time), to=CarPose(0.0, 0.01, 0.0, 0.0), duration=0.5)]
                            moved = True
                        sc.step(data)
                        agent.step(data)
                        mujoco.mj_step(model, data)
                    p_now = data.site_xpos[plug_site].copy()
                    if not trail or np.linalg.norm(p_now - trail[-1]) > 0.004:
                        trail.append(p_now)
                    draw_overlays(v, ghost, trail, trail_prev)
                    if result is None and (agent.done or data.time >= show.max_time):
                        plug_moved = float(np.linalg.norm(data.site_xpos[model.site("plug_face").id] - plug0))
                        result = verdict(show, agent, model, data, plug_moved)
                        results.append((show, result))
                        end_at = time.time() + 3.0
                v.sync()
                cam = robot_camera_overlay(agent, data, 0.5)
                world = [f"world: {t:.1f}s {txt}" for t, txt in sc.log[-2:]]
                frame = np.vstack([banner(show, cam.shape[1], result), cam,
                                   status_panel(agent, data, cam.shape[1], 330, world)])
                cv2.imshow("AutoCharge AI - robot view, decisions, verdict", frame)
                cv2.waitKey(1)
                if end_at is not None and time.time() >= end_at:
                    break
                time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))
            trail_prev = trail
            with v.lock():
                sc.restore()
                model.site_pos[sid] = site0
                perception.MARKER_POS_S = marker0
    cv2.destroyAllWindows()
    print("\nResults (independent checker):")
    for show, (ok, why) in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {show.title:52s} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
