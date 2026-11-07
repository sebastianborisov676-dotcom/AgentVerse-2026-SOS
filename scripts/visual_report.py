"""Record every validation scenario (frames + telemetry) and build an interactive HTML report.

    python scripts/visual_report.py                 # writes outputs/test_report/index.html
    python scripts/visual_report.py --skip-pytest   # reuse no pytest run, scenarios only

Each scenario mirrors one of the automated tests in ``tests/``. The report shows the robot
from three cameras, a status panel, and synced charts of alignment, contact force and cable
load against the controller's limits, plus the pass/fail criteria measured in the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import io
import json
import math
import pathlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocharge import (
    InsertionController,
    InsertionLimits,
    Phase,
    build_model,
    cable_metrics,
    connector_contact_force,
    plug_alignment_error,
    reset_to_home,
)
from autocharge.scene import ARM_JOINTS, PIN_LENGTH

FRAME_HZ = 4.0
SAMPLE_HZ = 10.0
MAX_FRAMES = 64
SPRITE_COLS = 8
CAMERAS = {
    "overview": (384, 288),
    "closeup": (320, 240),
    "robot": (256, 192),
}


@dataclasses.dataclass
class Scenario:
    id: str
    title: str
    kind: str  # insertion | hold | swing
    question: str  # what the test proves
    expected: str
    offset_mm: float = 0.0
    limits: dict | None = None
    estop: bool = False
    tests: tuple[str, ...] = ()


SCENARIOS = [
    Scenario(
        "nominal", "Nominal insertion", "insertion",
        "Does the robot align, insert slowly and seat the connector without touching the socket walls?",
        "Seated on the first attempt",
        tests=("test_nominal_insertion_seats_without_jamming",),
    ),
    Scenario(
        "error-3mm", "3 mm socket estimate error", "insertion",
        "With the socket estimate 3 mm off (more than the 2 mm clearance), do the pin chamfers guide it in?",
        "Seated, contact force below the hard limit",
        offset_mm=3.0, tests=("test_small_perception_error_is_absorbed_by_pin_chamfers",),
    ),
    Scenario(
        "error-4mm", "4 mm socket estimate error", "insertion",
        "When contact pushes the plug more than 2.5 mm off the expected axis, does the robot back out?",
        "Retracts on lost alignment, aborts after 2 retries",
        offset_mm=4.0, tests=("test_unexpected_deflection_aborts",),
    ),
    Scenario(
        "error-6mm", "6 mm socket estimate error (jam)", "insertion",
        "When the pins land on the socket rim, does the force limit stop the push and back out?",
        "Jam detected, retracts, aborts, never seated",
        offset_mm=6.0, tests=("test_large_perception_error_jams_and_aborts_safely",),
    ),
    Scenario(
        "hard-force-limit", "Hard contact-force limit", "insertion",
        "With the soft pause disabled, does the hard force limit alone stop a jammed insertion?",
        "Retracts when filtered force passes 15 N, aborts",
        offset_mm=6.0,
        limits={"contact_force_soft": 1e9, "contact_force_max": 15.0, "max_retries": 0},
        tests=("test_hard_contact_force_limit_triggers_retract",),
    ),
    Scenario(
        "cable-tension-limit", "Cable tension limit", "insertion",
        "Is cable tension monitored, and does exceeding the limit stop the robot?",
        "Retracts when tension passes 3 N (limit lowered for the test)",
        limits={"cable_tension_max": 3.0, "max_retries": 0},
        tests=("test_cable_tension_limit_triggers_retract",),
    ),
    Scenario(
        "strain-relief-limit", "Strain-relief load limit", "insertion",
        "Is the load on the connector's cable boot monitored, and does exceeding it stop the robot?",
        "Retracts when the boot load passes 2 N (limit lowered for the test)",
        limits={"strain_relief_max": 2.0, "max_retries": 0},
        tests=("test_strain_relief_limit_triggers_retract",),
    ),
    Scenario(
        "estop", "Emergency stop", "insertion",
        "When the operator hits emergency stop during insertion, does the arm freeze in place?",
        "Arm holds position, drift under 3 mm over 1 s",
        estop=True, tests=("test_emergency_stop_freezes_arm",),
    ),
    Scenario(
        "hold", "10 s hold with cable", "hold",
        "Does the scene stay physically stable while the arm holds still with the cable attached?",
        "No divergence or warnings, cable settles",
        tests=("test_hold_is_stable",),
    ),
    Scenario(
        "swing", "Large arm swing dragging the cable", "swing",
        "Does the cable stay attached and stable through large, fast arm motions?",
        "Stable, cable stretch under 3 %, no deep floor penetration",
        tests=("test_large_arm_motion_with_cable",),
    ),
]
STATIC_TESTS = {
    "test_original_robot_files_unmodified": "Google Robot model files are byte-identical to upstream",
    "test_connector_is_rigidly_mounted": "Connector is a jointless child of the gripper; home IK error < 0.1 mm",
    "test_cable_anchor_and_slack": "Cable root is clamped in the charger outlet; length is > 2x the span",
}


def _warnings(data) -> dict[str, int]:
    return {
        mujoco.mjtWarning(i).name: int(data.warning[i].number)
        for i in range(len(data.warning))
        if data.warning[i].number
    }


class Recorder:
    def __init__(self, model):
        self.m = model
        self.renderers = {k: mujoco.Renderer(model, h, w) for k, (w, h) in CAMERAS.items()}
        self.close_cam = mujoco.MjvCamera()
        self.frames: dict[str, list[np.ndarray]] = {k: [] for k in CAMERAS}
        self.frame_t: list[float] = []
        self.samples: list[dict] = []
        self._raw_max = 0.0
        self._next_frame = 0.0
        self._next_sample = 0.0

    def _render(self, data):
        sp = data.site_xpos[self.m.site("socket_face").id]
        self.close_cam.lookat[:] = sp + np.array([-0.07, 0, -0.01])
        self.close_cam.distance, self.close_cam.azimuth, self.close_cam.elevation = 0.42, 90, -8
        for k, r in self.renderers.items():
            if k == "overview":
                r.update_scene(data, camera="overview")
            elif k == "robot":
                r.update_scene(data, camera="connector_cam")
            else:
                r.update_scene(data, self.close_cam)
            self.frames[k].append(r.render().copy())
        self.frame_t.append(round(float(data.time), 3))

    def tick(self, data, phase: str, force_filt: float = 0.0):
        cf = connector_contact_force(self.m, data)
        self._raw_max = max(self._raw_max, cf.total_normal)
        if data.time + 1e-9 >= self._next_sample:
            al = plug_alignment_error(self.m, data)
            cab = cable_metrics(self.m, data)
            self.samples.append(
                {
                    "t": round(float(data.time), 3),
                    "phase": phase,
                    "lat": round(al.lateral * 1e3, 3),
                    "ax": round(al.axial * 1e3, 2),
                    "ang": round(math.degrees(al.angle), 3),
                    "f": round(force_filt, 2),
                    "fr": round(self._raw_max, 1),
                    "ten": round(cab.max_tension, 2),
                    "sr": round(cab.strain_relief_load, 2),
                    "strain": round(cab.max_edge_strain * 100, 3),
                    "clr": round(cab.min_clearance * 1e3, 2),
                    "spd": round(cab.max_speed, 3),
                }
            )
            self._raw_max = 0.0
            self._next_sample += 1.0 / SAMPLE_HZ
        if data.time + 1e-9 >= self._next_frame:
            self._render(data)
            self._next_frame += 1.0 / FRAME_HZ

    def sprites(self, out_dir: pathlib.Path, sid: str) -> dict:
        n = len(self.frame_t)
        keep = np.unique(np.linspace(0, n - 1, min(n, MAX_FRAMES)).round().astype(int))
        info = {"times": [self.frame_t[i] for i in keep], "cols": SPRITE_COLS, "files": {}, "size": {}}
        rows = math.ceil(len(keep) / SPRITE_COLS)
        for cam, frames in self.frames.items():
            w, h = CAMERAS[cam]
            sheet = Image.new("RGB", (w * SPRITE_COLS, h * rows), (20, 24, 30))
            for j, i in enumerate(keep):
                sheet.paste(Image.fromarray(frames[i]), ((j % SPRITE_COLS) * w, (j // SPRITE_COLS) * h))
            rel = f"frames/{sid}-{cam}.jpg"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            sheet.save(buf, "JPEG", quality=74, optimize=True, progressive=True)
            (out_dir / rel).write_bytes(buf.getvalue())
            info["files"][cam] = rel
            info["size"][cam] = [w, h]
        return info


def check(label: str, measured: float, op: str, limit: float, unit: str, fmt: str = ".2f") -> dict:
    ok = {"<": measured < limit, ">": measured > limit, ">=": measured >= limit}[op]
    return {
        "label": label,
        "measured": f"{measured:{fmt}} {unit}".strip(),
        "limit": f"{op} {limit:{fmt}} {unit}".strip(),
        "pass": bool(ok),
    }


def flag(label: str, ok: bool, measured: str, expected: str) -> dict:
    return {"label": label, "measured": measured, "limit": expected, "pass": bool(ok)}


def run_insertion(model, sc: Scenario) -> tuple[Recorder, dict, list, list]:
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    lim = dataclasses.replace(InsertionLimits(), **(sc.limits or {}))
    sid = model.site("socket_face").id
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    pos = data.site_xpos[sid].copy() + rot[:, 0] * sc.offset_mm * 1e-3
    ctrl = InsertionController(model, data, limits=lim, socket_estimate=(pos, rot))
    rec = Recorder(model)
    rec.tick(data, ctrl.phase.value)
    estop_pos = None
    max_time = 60.0
    while not ctrl.done and data.time < max_time:
        ctrl.step(data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"{sc.id}: diverged at t={data.time:.3f}")
        if sc.estop and ctrl.phase == Phase.INSERT and estop_pos is None:
            ctrl.emergency_stop(data)
            estop_pos = data.site_xpos[model.site("plug_face").id].copy()
        rec.tick(data, ctrl.phase.value, ctrl.force_filt)
    end = data.time + 1.0
    while data.time < end:
        ctrl.step(data)
        mujoco.mj_step(model, data)
        rec.tick(data, ctrl.phase.value, ctrl.force_filt)
    rec._render(data)

    al = plug_alignment_error(model, data)
    events_text = " | ".join(e for _, e in ctrl.events)
    checks: list[dict] = []
    retract = [e for _, e in ctrl.events if e.startswith("retract")]
    pk = ctrl.peak
    if sc.id == "nominal":
        t_ins = next(t for t, e in ctrl.events if e.startswith("insert"))
        t_seat = next((t for t, e in ctrl.events if e.startswith("seated")), float("nan"))
        checks += [
            flag("Final state", ctrl.phase == Phase.SEATED, ctrl.phase.value, "seated"),
            flag("Retries", ctrl.retries == 0, str(ctrl.retries), "0"),
            check("Final lateral error", al.lateral * 1e3, "<", 1.0, "mm"),
            check("Final angle error", math.degrees(al.angle), "<", 1.0, "deg"),
            check("Peak contact force (filtered)", pk["contact_force"], "<", lim.contact_force_soft, "N", ".1f"),
            check("Peak cable tension", pk["cable_tension"], "<", lim.cable_tension_max, "N", ".1f"),
            check("Peak strain-relief load", pk["strain_relief"], "<", lim.strain_relief_max, "N", ".1f"),
            check("Insertion time (proves slow insert)", t_seat - t_ins, ">=",
                  lim.seat_depth / lim.insert_speed, "s", ".1f"),
        ]
    elif sc.id == "error-3mm":
        checks += [
            flag("Final state", ctrl.phase == Phase.SEATED, ctrl.phase.value, "seated"),
            check("Peak contact force (filtered)", pk["contact_force"], "<", lim.contact_force_max, "N", ".1f"),
        ]
    elif sc.id == "error-4mm":
        checks += [
            flag("Retract reason", any("alignment lost" in r for r in retract),
                 "; ".join(r.split(": ", 1)[1] for r in retract[:1]) or "none", "alignment lost"),
            flag("Final state", ctrl.phase == Phase.ABORTED, ctrl.phase.value, "aborted"),
        ]
    elif sc.id == "error-6mm":
        checks += [
            flag("Retract reason", any("jam" in r for r in retract),
                 "; ".join(r.split(": ", 1)[1] for r in retract[:1]) or "none", "jam at soft limit"),
            flag("Final state", ctrl.phase == Phase.ABORTED, ctrl.phase.value, "aborted"),
            check("Peak contact force (filtered)", pk["contact_force"], "<", lim.contact_force_max, "N", ".1f"),
            check("Plug position (never seated)", al.axial * 1e3, "<", -30.0, "mm", ".1f"),
        ]
    elif sc.id == "hard-force-limit":
        checks += [
            flag("Retract reason", any("contact force" in r for r in retract),
                 "; ".join(r.split(": ", 1)[1] for r in retract[:1]) or "none", "contact force > 15 N"),
            flag("Final state", ctrl.phase == Phase.ABORTED, ctrl.phase.value, "aborted"),
            check("Peak contact force (filtered)", pk["contact_force"], "<", 25.0, "N", ".1f"),
        ]
    elif sc.id == "cable-tension-limit":
        checks += [
            flag("Retract reason", any("cable tension" in r for r in retract),
                 "; ".join(r.split(": ", 1)[1] for r in retract[:1]) or "none", "cable tension > 3 N"),
            flag("Final state", ctrl.phase == Phase.ABORTED, ctrl.phase.value, "aborted"),
        ]
    elif sc.id == "strain-relief-limit":
        checks += [
            flag("Retract reason", any("strain relief" in r for r in retract),
                 "; ".join(r.split(": ", 1)[1] for r in retract[:1]) or "none", "strain relief > 2 N"),
        ]
    elif sc.id == "estop":
        drift = float(np.linalg.norm(data.site_xpos[model.site("plug_face").id] - estop_pos)) * 1e3
        checks += [
            flag("Final state", ctrl.phase == Phase.ESTOP, ctrl.phase.value, "estop"),
            check("Drift 1 s after stop", drift, "<", 3.0, "mm"),
        ]
    checks.append(flag("No simulator warnings", not _warnings(data), str(_warnings(data) or "none"), "none"))
    summary = {
        "outcome": ctrl.phase.value,
        "retries": ctrl.retries,
        "peaks": {k: round(float(v), 2) for k, v in pk.items()},
        "events_text": events_text,
    }
    events = [{"t": t, "text": e} for t, e in ctrl.events]
    return rec, summary, checks, events


def _robot_bodies(model) -> set[int]:
    root = model.body("base_link").id
    out = set()
    for b in range(model.nbody):
        a = b
        while a not in (0, root):
            a = model.body_parentid[a]
        if a == root:
            out.add(b)
    return out


def run_passive(model, info, sc: Scenario) -> tuple[Recorder, dict, list, list]:
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    rec = Recorder(model)
    events = [{"t": round(float(data.time), 3), "text": f"{sc.kind}: start"}]
    peak_strain = peak_t = 0.0
    min_clear = 1.0
    robot_contacts = 0
    robot_b = _robot_bodies(model)
    if sc.kind == "hold":
        label = "holding"
        steps = int(10.0 / model.opt.timestep)
        plan = [None] * steps
    else:
        label = "swinging"
        acts = [
            i for i in range(model.nu)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) in ARM_JOINTS
        ]
        q0 = info.home_qpos_arm
        wps = [q0, q0 + [0.6, -0.3, 0.4, -0.5, 0.5, 0.4, 1.0], q0 + [-0.4, -0.5, -0.3, 0.3, -0.6, -0.4, -1.0], q0]
        seg = int(2.5 / model.opt.timestep)
        plan = []
        for k in range(3):
            for i in range(seg):
                s = 0.5 - 0.5 * math.cos(math.pi * i / seg)
                plan.append((1 - s) * wps[k] + s * wps[k + 1])
        for k in range(3):
            events.append({"t": round(float(data.time) + 2.5 * k, 3), "text": f"swing: waypoint {k + 1}"})
    rec.tick(data, label)
    for i, target in enumerate(plan):
        if target is not None:
            data.ctrl[acts] = target
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"{sc.id}: diverged at t={data.time:.3f}")
        rec.tick(data, label)
        if i % 25 == 0:
            c = cable_metrics(model, data)
            peak_strain = max(peak_strain, c.max_edge_strain)
            min_clear = min(min_clear, c.min_clearance)
            peak_t = max(peak_t, c.max_tension)
            for j in range(data.ncon):
                con = data.contact[j]
                if (con.flex[0] >= 0) != (con.flex[1] >= 0):
                    g = con.geom[1] if con.flex[0] >= 0 else con.geom[0]
                    if g >= 0 and model.geom_bodyid[g] in robot_b:
                        robot_contacts += 1
    rec._render(data)
    c = cable_metrics(model, data)
    warn = _warnings(data)
    events.append({"t": round(float(data.time), 3), "text": f"{sc.kind}: end"})
    checks = [flag("No divergence (finite state)", True, "finite", "finite"),
              flag("No simulator warnings", not warn, str(warn or "none"), "none")]
    if sc.kind == "hold":
        checks += [
            check("Peak cable stretch", peak_strain * 100, "<", 1.5, "%"),
            check("Worst floor penetration", min_clear * 1e3, ">", -2.0, "mm"),
            check("Peak cable tension", peak_t, "<", 15.0, "N", ".1f"),
            check("Fastest cable point at the end", c.max_speed, "<", 0.2, "m/s", ".3f"),
        ]
    else:
        checks += [
            check("Peak cable stretch", peak_strain * 100, "<", 3.0, "%"),
            check("Worst floor penetration", min_clear * 1e3, ">", -4.0, "mm"),
            flag("Cable touched the robot body", True, f"{robot_contacts} contact samples", "informational"),
        ]
    summary = {"outcome": "stable" if all(ch["pass"] for ch in checks) else "unstable", "retries": 0,
               "peaks": {"cable_tension": round(peak_t, 2), "strain_pct": round(peak_strain * 100, 3)}}
    return rec, summary, checks, events


def run_pytest(out_dir: pathlib.Path) -> dict:
    xml_path = out_dir / "junit.xml"
    t = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={xml_path}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-500:])
    tests = []
    root = ET.parse(xml_path).getroot()
    for case in root.iter("testcase"):
        failed = case.find("failure") is not None or case.find("error") is not None
        skipped = case.find("skipped") is not None
        tests.append({
            "name": case.get("name"),
            "file": (case.get("classname") or "").replace(".", "/") + ".py",
            "seconds": round(float(case.get("time", 0)), 2),
            "status": "failed" if failed else "skipped" if skipped else "passed",
        })
    return {"tests": tests, "wall_seconds": round(time.time() - t, 1), "exit_code": proc.returncode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "outputs" / "test_report")
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--html-only", action="store_true", help="rebuild index.html from report.json")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.html_only:
        write_html(out, json.loads((out / "report.json").read_text(encoding="utf-8")))
        return 0

    model, info = build_model()
    lim = InsertionLimits()
    report = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mujoco": mujoco.__version__,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "limits": {
            "align_lateral_mm": lim.align_lateral_tol * 1e3,
            "insert_lateral_max_mm": lim.insert_lateral_max * 1e3,
            "force_soft": lim.contact_force_soft,
            "force_max": lim.contact_force_max,
            "tension_max": lim.cable_tension_max,
            "strain_relief_max": lim.strain_relief_max,
            "standoff_mm": -lim.standoff * 1e3,
            "seat_mm": -(PIN_LENGTH - lim.seat_depth) * 1e3,
            "pin_contact_mm": -PIN_LENGTH * 1e3,
            "insert_speed_mm_s": lim.insert_speed * 1e3,
        },
        "scene": {
            "cable_length_m": round(info.cable_length, 2),
            "span_m": round(info.anchor_to_connector, 2),
            "segments": info.params.cable_segments,
        },
        "static_tests": STATIC_TESTS,
        "scenarios": [],
    }
    for sc in SCENARIOS:
        t = time.time()
        if sc.kind == "insertion":
            rec, summary, checks, events = run_insertion(model, sc)
        else:
            rec, summary, checks, events = run_passive(model, info, sc)
        sprites = rec.sprites(out, sc.id)
        lims = dict(report["limits"])
        for k, v in (sc.limits or {}).items():
            key = {"contact_force_soft": "force_soft", "contact_force_max": "force_max",
                   "cable_tension_max": "tension_max", "strain_relief_max": "strain_relief_max"}.get(k)
            if key:
                lims[key] = v
        report["scenarios"].append({
            **{k: v for k, v in dataclasses.asdict(sc).items() if k != "limits"},
            "limits": lims,
            "summary": summary,
            "checks": checks,
            "passed": all(c["pass"] for c in checks),
            "events": events,
            "samples": rec.samples,
            "sprites": sprites,
            "duration": rec.samples[-1]["t"] if rec.samples else 0,
        })
        print(f"{sc.id:22s} {summary['outcome']:9s} checks {'PASS' if all(c['pass'] for c in checks) else 'FAIL'}"
              f"  ({time.time() - t:.0f} s, {len(sprites['times'])} frames)")

    report["pytest"] = None if args.skip_pytest else run_pytest(out)
    (out / "report.json").write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    write_html(out, report)
    return 0


def write_html(out: pathlib.Path, report: dict) -> None:
    template = (ROOT / "scripts" / "visual_report_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__REPORT_DATA__*/null", json.dumps(report, separators=(",", ":")))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"Report written to {out / 'index.html'}")


if __name__ == "__main__":
    sys.exit(main())
