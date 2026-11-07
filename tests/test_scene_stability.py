"""Scene construction and physics stability checks (run before adding camera autonomy)."""

import subprocess

import mujoco
import numpy as np

from autocharge import cable_metrics, reset_to_home
from autocharge.scene import ARM_JOINTS, ROBOT_DIR


def _warnings(data):
    return {i: data.warning[i].number for i in range(len(data.warning)) if data.warning[i].number}


def test_original_robot_files_unmodified(scene):
    """The Menagerie Google Robot files must be byte-identical to upstream."""
    repo = ROBOT_DIR.parent
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "google_robot"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == ""


def test_connector_is_rigidly_mounted(scene):
    model, info = scene
    conn = model.body("connector").id
    assert model.body_parentid[conn] == model.body("link_gripper").id
    assert model.body_jntnum[conn] == 0  # welded by construction, no joint
    assert info.ik_error_pos < 1e-4 and info.ik_error_rot < 1e-3


def test_cable_anchor_and_slack(scene):
    model, info = scene
    # Clamped root: the first two cable points are pinned to the charger outlet body.
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    assert "cable_0" not in names and "cable_1" not in names
    assert model.body_parentid[model.body("cable_2").id] == model.body("cable_outlet").id
    assert info.cable_length / info.anchor_to_connector > 2.0


def test_hold_is_stable(scene):
    """10 s with the arm servoing to the home pose: cable settles, no divergence or warnings."""
    model, _ = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    peak_strain, min_clear, peak_t = 0.0, 1.0, 0.0
    steps = int(10.0 / model.opt.timestep)
    for i in range(steps):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), f"diverged at t={data.time}"
        if i % 25 == 0:
            c = cable_metrics(model, data)
            peak_strain = max(peak_strain, c.max_edge_strain)
            min_clear = min(min_clear, c.min_clearance)
            peak_t = max(peak_t, c.max_tension)
    c = cable_metrics(model, data)
    assert _warnings(data) == {}
    assert peak_strain < 0.015  # inextensible to within 1.5 %
    assert min_clear > -0.002  # no more than 2 mm floor penetration
    assert peak_t < 15.0
    assert c.max_speed < 0.2  # settled (residual creep from the soft servos only)


def test_large_arm_motion_with_cable(scene):
    """Swing the arm through large joint motions dragging the cable: must stay stable."""
    model, info = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    acts = [
        i
        for i in range(model.nu)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) in ARM_JOINTS
    ]
    q0 = info.home_qpos_arm
    waypoints = [
        q0,
        q0 + [0.6, -0.3, 0.4, -0.5, 0.5, 0.4, 1.0],
        q0 + [-0.4, -0.5, -0.3, 0.3, -0.6, -0.4, -1.0],
        q0,
    ]
    seg_steps = int(2.5 / model.opt.timestep)
    root = model.body("base_link").id
    robot_bodies = set()
    for b in range(model.nbody):
        a = b
        while a not in (0, root):
            a = model.body_parentid[a]
        if a == root:
            robot_bodies.add(b)
    peak_strain, min_clear, robot_contacts = 0.0, 1.0, 0
    for k in range(len(waypoints) - 1):
        for i in range(seg_steps):
            s = 0.5 - 0.5 * np.cos(np.pi * i / seg_steps)
            data.ctrl[acts] = (1 - s) * waypoints[k] + s * waypoints[k + 1]
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all(), f"diverged at t={data.time}"
            if i % 20 == 0:
                c = cable_metrics(model, data)
                peak_strain = max(peak_strain, c.max_edge_strain)
                min_clear = min(min_clear, c.min_clearance)
                for j in range(data.ncon):
                    con = data.contact[j]
                    if (con.flex[0] >= 0) != (con.flex[1] >= 0):
                        g = con.geom[1] if con.flex[0] >= 0 else con.geom[0]
                        if g >= 0 and model.geom_bodyid[g] in robot_bodies:
                            robot_contacts += 1
    assert _warnings(data) == {}
    assert peak_strain < 0.03
    assert min_clear > -0.004
    print(f"cable-robot contact samples during swing: {robot_contacts}")
