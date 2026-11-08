#!/usr/bin/env python3
"""Bounded process-isolated experiments using the same Session pipeline as the panel.

The previous privileged-map baseline remains available only with --engine legacy.
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import multiprocessing
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from autocharge.training.course import load_manual


def run_episode(case):
    started = time.monotonic()
    folder = pathlib.Path(case["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    row = dict(
        level=case["level"],
        seed=case["seed"],
        engine=case["engine"],
        directory=str(folder),
    )
    (folder / "scenario.json").write_text(json.dumps(case, indent=2))
    session = None
    try:
        with (
            (folder / "console.log").open("w") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            if case["engine"] == "legacy":
                from autocharge.training.environment import CourseEnv

                env = CourseEnv(
                    case["level"],
                    case["max_time"],
                    load_manual(case["manual"]) if case["manual"] else (),
                    case["mode"],
                )
                env.reset(case["seed"], case["alignment_offset_mm"])
                while not (env.terminated or env.truncated):
                    if time.monotonic() - started > case["wall_time"]:
                        env.truncated = True
                        env.truncation_reason = "episode wall-clock limit"
                        break
                    env.step()
                row.update(
                    phase=env.mission.phase,
                    success=env.mission.phase == "seated",
                    truncated=env.truncated,
                    time=float(env.data.time),
                    events=env.mission.events,
                    obstacles=[dataclasses.asdict(o) for o in env.obstacles],
                    warnings=[int(w.number) for w in env.data.warning],
                    travelled=env.mission.drive.travelled,
                    **env.report(),
                )
            else:
                from autocharge.panel.session import Session, STATE

                session = Session(folder)
                config = dict(
                    level=case["level"],
                    seed=case["seed"],
                    mode=case["mode"],
                    manual=[dataclasses.asdict(o) for o in load_manual(case["manual"])]
                    if case["manual"]
                    else [],
                    moving_car=case["moving_car"],
                    random_start=case["random_start"],
                )
                session.configure(config)
                session.action("start", {})
                truncated = False
                while session.status == "running":
                    reason = None
                    if time.monotonic() - started > case["wall_time"]:
                        reason = "episode wall-clock limit"
                    elif session.data.time >= case["max_time"]:
                        reason = "episode simulated-time limit"
                    if reason:
                        truncated = True
                        session.controller.stop(session.data, reason)
                        session.status = "finished"
                        session.record.capture(session, checkpoint=True)
                        break
                    # Identical physics/control stepping to panel.server.Worker (no renderer).
                    session.tick(10)
                final = session.snapshot()
                integration = np.empty(mujoco.mj_stateSize(session.model, STATE))
                mujoco.mj_getState(session.model, session.data, integration, STATE)
                row.update(
                    phase=final["phase"],
                    success=final["phase"] == "seated",
                    truncated=truncated,
                    time=final["time"],
                    events=final["events"],
                    obstacles=final["obstacles"],
                    config=final["config"],
                    start_pose=final["start_pose"],
                    cable_length=final["cable_length"],
                    contacts=final["contacts"],
                    startup=session.startup,
                    cable=final["cable"],
                    failure_reason=final["failure"],
                    failure_category=None
                    if final["phase"] == "seated"
                    else "safety_or_navigation",
                    docking=dict(
                        reached=any(
                            e.startswith("docked:") for _, e in final["events"]
                        ),
                        distance_m=session.controller.mission.drive.errors(
                            session.data
                        )[0],
                    ),
                    replans=final["replans"],
                    vehicle_started=session.controller.vehicle.started,
                    avoidance_replans=session.controller.avoidance_replans,
                    vehicle_displacement=session.data.qpos[session.controller.vehicle.qadr].tolist() if session.controller.vehicle.available else None,
                    travelled=session.controller.mission.drive.travelled,
                    warnings=[int(w.number) for w in session.data.warning],
                    pipeline="autocharge.panel.Session -> build_scenario + AdaptiveMission",
                    final_state_sha256=hashlib.sha256(
                        integration.tobytes()
                    ).hexdigest(),
                    recording=str(folder / "runs" / (session.record.id + ".npz")),
                )
                if any(row["warnings"]):
                    row.update(
                        success=False,
                        failure_reason="MuJoCo warning",
                        failure_category="simulation",
                    )
    except Exception as exc:
        row.update(
            success=False,
            phase="setup_failed",
            failure_reason=str(exc),
            failure_category="setup_or_configuration",
            docking=None,
            contacts=None,
            cable=None,
            error_type=type(exc).__name__,
        )
    finally:
        if session is not None:
            session.close()
    if not row.get("success") and row.get("failure_category") != "simulation":
        reason = (row.get("failure_reason") or "").lower()
        if "cable" in reason or "strain relief" in reason:
            row["failure_category"] = "cable_contact_constraint"
        elif "time limit" in reason or "timeout" in reason:
            row["failure_category"] = "episode_timeout"
        elif "observation lost" in reason:
            row["failure_category"] = "perception"
        elif "unsafe robot contact" in reason:
            row["failure_category"] = "contact_safety"
        elif "route" in reason or "replanning" in reason or "obstacle safety" in reason:
            row["failure_category"] = "navigation"
    events = row.get("events", [])
    patterns = {
        "obstacle_detection": "obstacle detected:",
        "avoidance_replan": "avoidance replan:",
        "vehicle_motion": "vehicle motion started:",
        "braking": "port motion detected by visible fiducial; braking",
        "updated_target": "stationary port confirmed; updated docking pose",
        "seated": "seated:",
    }
    row["milestones"] = {
        name: next((t for t, event in events if event.startswith(prefix)), None)
        for name, prefix in patterns.items()
    }
    row["milestones"]["safe_abort"] = next((t for t, event in events if event == row.get("failure_reason")), row.get("time")) if not row.get("success") else None
    row["both_decisions_observed"] = all(
        row["milestones"][k] is not None
        for k in ("avoidance_replan", "braking", "updated_target")
    )
    row["elapsed_wall_s"] = time.monotonic() - started
    (folder / "result.json").write_text(json.dumps(row, indent=2))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    ap.add_argument("--engine", choices=["panel", "legacy"], default="panel")
    ap.add_argument("--max-time", type=float, default=90)
    ap.add_argument("--wall-time", type=float, default=180)
    ap.add_argument("--mode", choices=["random", "manual", "hybrid"], default="random")
    ap.add_argument("--manual", type=pathlib.Path)
    ap.add_argument("--alignment-offset-mm", type=float, default=0)
    ap.add_argument("--no-moving-car", action="store_true")
    ap.add_argument(
        "--nominal-start", action="store_true", help="disable seeded pose perturbations"
    )
    ap.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("outputs/course_batch.json")
    )
    args = ap.parse_args()
    if args.max_time <= 0 or args.wall_time <= 0:
        ap.error("Timeouts must be positive")
    if any(level not in range(1, 12) for level in args.levels):
        ap.error("Levels must be 1–11")
    if args.engine == "panel" and args.alignment_offset_mm:
        ap.error(
            "Panel uses observed fiducials; artificial offsets require --engine legacy"
        )
    if args.engine == "legacy" and 11 in args.levels:
        ap.error("Master Challenge requires --engine panel for sensor-based avoidance and moving-target recovery")
    root = args.output.with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dict(
        output=str(args.output), manual=str(args.manual) if args.manual else None
    )
    cases = []
    for level in args.levels:
        for seed in args.seeds:
            cases.append(
                dict(
                    level=level,
                    seed=seed,
                    engine=args.engine,
                    max_time=args.max_time,
                    wall_time=args.wall_time,
                    mode=args.mode,
                    manual=str(args.manual.resolve()) if args.manual else None,
                    moving_car=not args.no_moving_car,
                    random_start=not args.nominal_start,
                    alignment_offset_mm=args.alignment_offset_mm,
                    directory=str(
                        root / f"level_{level:02d}_seed_{seed}_case_{len(cases):03d}"
                    ),
                )
            )
    results = []

    def save():
        ordered = sorted(results, key=lambda r: (r["level"], r["seed"]))
        args.output.write_text(
            json.dumps(
                dict(
                    schema_version=3,
                    mujoco=mujoco.__version__,
                    numpy=np.__version__,
                    configuration=config,
                    episodes=ordered,
                ),
                indent=2,
            )
        )

    # spawn avoids sharing MuJoCo state or GL contexts. Every case has its own directory.
    if args.workers == 1:
        for case in cases:
            row = run_episode(case)
            results.append(row)
            save()
            print(
                json.dumps(
                    dict(
                        level=row["level"],
                        seed=row["seed"],
                        phase=row["phase"],
                        success=row["success"],
                        failure=row.get("failure_reason"),
                    )
                ),
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            pending = {pool.submit(run_episode, case): case for case in cases}
            for future in concurrent.futures.as_completed(pending):
                case = pending[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(
                        level=case["level"],
                        seed=case["seed"],
                        phase="worker_failed",
                        success=False,
                        failure_reason=str(exc),
                    )
                results.append(row)
                save()
                print(
                    json.dumps(
                        dict(
                            level=row["level"],
                            seed=row["seed"],
                            phase=row["phase"],
                            success=row["success"],
                            failure=row.get("failure_reason"),
                        )
                    ),
                    flush=True,
                )
    return 0 if all(row["success"] for row in results) else 2


if __name__ == "__main__":
    sys.exit(main())
