"""Run the AutoCharger AI cable + connector insertion demo.

Examples (from the project root, with the virtual environment active):

    python scripts/run_autocharge.py                     # headless run, prints a report
    python scripts/run_autocharge.py --viewer            # interactive MuJoCo viewer
    python scripts/run_autocharge.py --gif outputs/run.gif
    python scripts/run_autocharge.py --perception-error-mm 4   # simulate a bad socket estimate
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from autocharge import (
    InsertionController,
    InsertionLimits,
    build_model,
    cable_metrics,
    reset_to_home,
)


def make_controller(model, data, perception_error_mm: float, limits: InsertionLimits):
    sid = model.site("socket_face").id
    pos = data.site_xpos[sid].copy()
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    # Lateral perception error along the socket's +x ("up") axis.
    pos += rot[:, 0] * perception_error_mm * 1e-3
    return InsertionController(model, data, limits=limits, socket_estimate=(pos, rot))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--gif", type=pathlib.Path, help="save an animated GIF of the run")
    ap.add_argument("--csv", type=pathlib.Path, help="save telemetry as CSV")
    ap.add_argument("--perception-error-mm", type=float, default=0.0)
    ap.add_argument("--max-time", type=float, default=45.0, help="simulated seconds")
    args = ap.parse_args()

    t0 = time.time()
    model, info = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    print(f"Scene built in {time.time() - t0:.1f} s  (nq={model.nq}, bodies={model.nbody})")
    print(
        f"Cable: {info.cable_length:.2f} m in {info.params.cable_segments} segments, "
        f"anchor->connector span {info.anchor_to_connector:.2f} m "
        f"(slack ratio {info.cable_length / info.anchor_to_connector:.2f})"
    )
    ctrl = make_controller(model, data, args.perception_error_mm, InsertionLimits())

    frames = []
    renderer = mujoco.Renderer(model, 480, 640) if args.gif else None

    def tick():
        ctrl.step(data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"simulation diverged at t={data.time:.3f}")
        if renderer is not None and len(frames) < data.time * 10:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as v:
            while v.is_running() and data.time < args.max_time:
                start = time.time()
                for _ in range(int(0.02 / model.opt.timestep)):
                    if not ctrl.done:
                        tick()
                    else:
                        ctrl.step(data)
                        mujoco.mj_step(model, data)
                v.sync()
                time.sleep(max(0.0, 0.02 - (time.time() - start)))
    else:
        t1 = time.time()
        while not ctrl.done and data.time < args.max_time:
            tick()
        # Hold for one more second after finishing to show it stays put.
        end = data.time + 1.0
        while data.time < end:
            tick()
        print(f"Simulated {data.time:.1f} s in {time.time() - t1:.1f} s wall time")

    cab = cable_metrics(model, data)
    print("\nEvents:")
    for t, msg in ctrl.events:
        print(f"  {t:7.3f} s  {msg}")
    print(f"\nFinal phase: {ctrl.phase.value}   retries: {ctrl.retries}")
    print(
        "Peaks: contact force {contact_force:.1f} N | cable tension {cable_tension:.1f} N | "
        "strain-relief load {strain_relief:.1f} N".format(**ctrl.peak)
    )
    print(
        f"Cable now: max tension {cab.max_tension:.1f} N, edge strain {cab.max_edge_strain * 100:.2f} %, "
        f"floor clearance {cab.min_clearance * 1e3:.1f} mm"
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            fields = [fl.name for fl in dataclasses.fields(ctrl.telemetry[0])]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in ctrl.telemetry:
                w.writerow(dataclasses.asdict(row))
        print(f"Telemetry written to {args.csv}")
    if args.gif and frames:
        from PIL import Image

        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames[::2]]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
        print(f"GIF written to {args.gif} ({len(imgs)} frames, 5x speed)")
    return 0 if ctrl.phase.value == "seated" else 2


if __name__ == "__main__":
    sys.exit(main())
