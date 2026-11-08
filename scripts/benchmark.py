"""Held-out benchmark: autonomous agent vs a pre-programmed (fixed-coordinate) controller.

    python scripts/benchmark.py              # 10 held-out trials, both controllers
    python scripts/benchmark.py --trials 20

Each trial draws a random parking pose (and, for half of them, a random car move during the
approach) from seeds that were never used during development. Success is judged by the
independent evaluator (pins >= 30 mm deep, within the cavity clearance), not by either
controller's own claim. Results go to outputs/benchmark/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from autocharge import InsertionController, build_model, reset_to_home
from autocharge.autonomy import AutonomousAgent
from autocharge.evaluation import check_docking
from autocharge.scenarios import CarMove, CarPose, ScenarioManager

HELD_OUT_SEED_BASE = 1000  # development used seeds < 100


def make_trial(i: int):
    rng = np.random.default_rng(HELD_OUT_SEED_BASE + i)
    park = CarPose.random(rng)
    moves = ()
    if i % 2 == 1:  # every second trial: the car moves during the approach
        to = CarPose(park.dx + rng.uniform(-0.02, 0.02), park.dy + rng.uniform(-0.03, 0.03),
                     park.dz + rng.uniform(-0.01, 0.01), park.dyaw_deg + rng.uniform(-3, 3))
        moves = (CarMove(at=float(rng.uniform(2.0, 4.0)), to=to),)
    return park, moves


def run(model, controller: str, park, moves, max_time=60.0) -> dict:
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    nominal = data.site_xpos[model.site("socket_face").id].copy(), data.site_xmat[model.site("socket_face").id].reshape(3, 3).copy()
    sc = ScenarioManager(model, data, park=park, moves=moves)
    mujoco.mj_forward(model, data)
    t0 = time.time()
    if controller == "autonomous":
        ctrl = AutonomousAgent(model, data)
    else:  # pre-programmed: drives to the nominal bay's socket coordinates, never looks
        ctrl = InsertionController(model, data, socket_estimate=nominal)
    try:
        while not ctrl.done and data.time < max_time:
            sc.step(data)
            ctrl.step(data)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                break
        v = check_docking(model, data)
        inner = ctrl.ctrl if controller == "autonomous" else ctrl
        events = ctrl.events
    finally:
        sc.restore()
    return {
        "controller": controller,
        "docked": v.docked,
        "verdict": v.reason,
        "lateral_mm": round(v.lateral_mm, 2),
        "depth_mm": round(v.depth_mm, 1),
        "final_state": ctrl.phase if controller == "autonomous" else ctrl.phase.value,
        "sim_time_s": round(float(data.time), 1),
        "wall_s": round(time.time() - t0, 1),
        "peak_force_n": round(inner.peak["contact_force"], 1) if inner else 0.0,
        "retries": inner.retries if inner else 0,
        "unsafe_contact": bool(inner and inner.peak["contact_force"] > 70.0),
        "last_event": events[-1][1] if events else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/benchmark"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, _ = build_model()
    rows = []
    for i in range(args.trials):
        park, moves = make_trial(i)
        for controller in ("autonomous", "pre-programmed"):
            r = run(model, controller, park, moves)
            r.update(trial=i, seed=HELD_OUT_SEED_BASE + i, car_moves=bool(moves),
                     park=[round(park.dx * 100, 1), round(park.dy * 100, 1), round(park.dz * 100, 1),
                           round(park.dyaw_deg, 1)])
            rows.append(r)
            print(f"trial {i} {'move' if moves else 'park'} {controller:14s} docked={r['docked']!s:5} "
                  f"{r['verdict']:28s} state={r['final_state']:9s} retries={r['retries']} "
                  f"peakF={r['peak_force_n']}N sim={r['sim_time_s']}s")
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| Controller | Scenario | Trials | Docked | Unsafe contacts | Median sim time (docked) |", "|---|---|---|---|---|---|"]
    for controller in ("autonomous", "pre-programmed"):
        for label, moving in (("random parking", False), ("random parking + car moves", True)):
            sel = [r for r in rows if r["controller"] == controller and r["car_moves"] == moving]
            ok = [r for r in sel if r["docked"]]
            med = f"{np.median([r['sim_time_s'] for r in ok]):.1f} s" if ok else "–"
            lines.append(f"| {controller} | {label} | {len(sel)} | {len(ok)}/{len(sel)} | "
                         f"{sum(r['unsafe_contact'] for r in sel)} | {med} |")
    table = "\n".join(lines)
    (args.out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
