"""Autonomous charging demo: the robot finds the inlet with its camera and adapts live.

    python scripts/autonomy_demo.py                    # live: MuJoCo viewer + robot-camera window
    python scripts/autonomy_demo.py --seed 7           # random parking position
    python scripts/autonomy_demo.py --record outputs/demo.mp4 --move-at 3   # headless video

Live keys (click the MuJoCo viewer window first):
    arrow keys      push the car 1 cm sideways / along the inlet axis
    [  ]            turn the car 1 degree
    B               block / unblock the camera's view of the inlet
    E               emergency stop

The robot is never told about these changes; it has to notice them through its camera or force sensing.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from autocharge import build_model, reset_to_home
from autocharge.autonomy import AutonomousAgent
from autocharge.overlay import robot_camera_overlay, status_panel
from autocharge.scenarios import CarMove, CarPose, Occlusion, ScenarioManager


def compose(agent, data, overview: np.ndarray, extra) -> np.ndarray:
    cam = robot_camera_overlay(agent, data, scale=0.5)  # 480x360
    panel = status_panel(agent, data, cam.shape[1], overview.shape[0] - cam.shape[0], extra)
    right = np.vstack([cam, panel])
    return np.hstack([cv2.cvtColor(overview, cv2.COLOR_RGB2BGR), right])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="random parking position")
    ap.add_argument("--record", type=pathlib.Path, help="headless: write an MP4 instead of opening windows")
    ap.add_argument("--move-at", type=float, help="scenario manager rolls the car at this time (s)")
    ap.add_argument("--max-time", type=float, default=45.0)
    args = ap.parse_args()

    model, _ = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    park = CarPose.random(np.random.default_rng(args.seed)) if args.seed is not None else None
    moves = (CarMove(at=args.move_at, to=CarPose(0.0, 0.03, 0.0, 3.0)),) if args.move_at else ()
    sc = ScenarioManager(model, data, park=park, moves=moves)
    mujoco.mj_forward(model, data)
    agent = AutonomousAgent(model, data)
    steps_per_frame = int(round((1 / 30) / model.opt.timestep))

    def world_events():
        return [f"world: {t:.1f}s {txt}" for t, txt in sc.log[-2:]]

    if args.record:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.3, -0.35, 0.7]
        cam.distance, cam.azimuth, cam.elevation = 2.1, 60.0, -22.0
        ren = mujoco.Renderer(model, 720, 960)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        end_after = None
        while data.time < args.max_time:
            for _ in range(steps_per_frame):
                sc.step(data)
                agent.step(data)
                mujoco.mj_step(model, data)
            ren.update_scene(data, cam)
            frame = compose(agent, data, ren.render(), world_events())
            if writer is None:
                writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            if agent.done and end_after is None:
                end_after = data.time + 1.5
            if end_after is not None and data.time >= end_after:
                break
        writer.release()
        print(f"Wrote {args.record} ({data.time:.1f} s simulated). Final state: {agent.phase}")
        for t, e in sorted(agent.events + sc.log):
            print(f"  {t:6.2f}  {e}")
        return 0

    from mujoco import viewer as mj_viewer

    keys: list[int] = []
    with mj_viewer.launch_passive(model, data, key_callback=keys.append) as v:
        v.cam.lookat[:] = [0.3, -0.35, 0.7]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.1, 60.0, -22.0
        covered = False
        while v.is_running() and data.time < args.max_time + 600:
            t0 = time.time()
            while keys:
                k = keys.pop(0)
                with v.lock():
                    if k == 263:
                        sc.nudge(data, dy=0.01)
                    elif k == 262:
                        sc.nudge(data, dy=-0.01)
                    elif k == 265:
                        sc.nudge(data, dx=0.01)
                    elif k == 264:
                        sc.nudge(data, dx=-0.01)
                    elif k == ord("["):
                        sc.nudge(data, dyaw_deg=1.0)
                    elif k == ord("]"):
                        sc.nudge(data, dyaw_deg=-1.0)
                    elif k == ord("B"):
                        covered = not covered
                        sc.occlusions = (Occlusion(0.0, 1e9),) if covered else ()
                    elif k == ord("E") and agent.ctrl is not None:
                        agent.ctrl.emergency_stop(data)
            with v.lock():
                for _ in range(steps_per_frame):
                    sc.step(data)
                    agent.step(data)
                    mujoco.mj_step(model, data)
            v.sync()
            cv2.imshow("AutoCharge AI - robot view and decisions",
                       np.vstack([robot_camera_overlay(agent, data, 0.5),
                                  status_panel(agent, data, 480, 380, world_events())]))
            cv2.waitKey(1)
            time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
