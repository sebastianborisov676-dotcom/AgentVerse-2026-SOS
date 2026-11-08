#!/usr/bin/env python3
"""Replay a saved layout independently of generator revisions to inspect cable contacts.

Example: python scripts/inspect_course_contacts.py outputs/course_final.json --level 5 --seed 0
"""

import argparse
import dataclasses
import json
import pathlib
import sys

import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from autocharge.base_controller import DriveLimits
from autocharge.mission import DRIVE_IN_PARAMS, ChargingMission
from autocharge.scene import build_model, reset_to_home
from autocharge.training.levels import Obstacle
from autocharge.training.navigation import NoPathError
from autocharge.training.telemetry import EpisodeTelemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs/contact_diagnosis.json"),
    )
    args = parser.parse_args()
    if args.settle < 0 or args.max_time <= 0:
        parser.error("settle must be nonnegative and max-time positive")
    rows = json.loads(args.report.read_text())["episodes"]
    row = next(r for r in rows if (r["level"], r["seed"]) == (args.level, args.seed))
    obstacles = tuple(Obstacle(**o) for o in row["obstacles"])
    model, _ = build_model(dataclasses.replace(DRIVE_IN_PARAMS, obstacles=obstacles))
    data = mujoco.MjData(model)
    reset_to_home(model, data, settle=0)
    settling = EpisodeTelemetry(model)
    initial = []
    for contact in data.contact:
        a, b = map(int, contact.geom)
        category = settling.contacts.classify(a, b)
        if category.startswith("obstacle_"):
            initial.append(
                {
                    "category": category,
                    "geom_ids": [a, b],
                    "geom_names": [
                        model.geom(g).name if g >= 0 else "cable" for g in (a, b)
                    ],
                    "distance_m": float(contact.dist),
                }
            )
    snapshots = []
    for step in range(round(args.settle / model.opt.timestep) + 1):
        load = settling.sample(data, DriveLimits())
        if step % 25 == 0:
            snapshots.append(
                {
                    "time": float(data.time),
                    "tension_N": load.max_tension,
                    "strain_relief_N": load.strain_relief_load,
                }
            )
        if step < round(args.settle / model.opt.timestep):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    drive = EpisodeTelemetry(model)
    events = []
    try:
        mission = ChargingMission(model, data, obstacles=obstacles)
        while not mission.done and data.time < args.max_time:
            mission.step(data)
            drive.sample(data, mission.drive.lim)
            mujoco.mj_step(model, data)
        events = mission.events
    except NoPathError as exc:
        events = [(float(data.time), str(exc))]
    result = {
        "source_report": str(args.report),
        "level": args.level,
        "seed": args.seed,
        "obstacles": row["obstacles"],
        "initial_contacts": initial,
        "settling": settling.report(),
        "snapshots": snapshots,
        "mission": drive.report(),
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
