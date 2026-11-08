"""Insertion controller behaviour: nominal success and every safety limit firing."""

import dataclasses

import mujoco
import numpy as np

from autocharge import InsertionController, InsertionLimits, Phase, plug_alignment_error, reset_to_home


def _run(model, limits=None, offset_mm=0.0, max_time=45.0, on_step=None):
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    sid = model.site("socket_face").id
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    pos = data.site_xpos[sid].copy() + rot[:, 0] * offset_mm * 1e-3
    ctrl = InsertionController(model, data, limits=limits, socket_estimate=(pos, rot))
    while not ctrl.done and data.time < max_time:
        ctrl.step(data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        if on_step is not None:
            on_step(ctrl, data)
    return ctrl, data


def test_nominal_insertion_seats_without_jamming(scene):
    model, _ = scene
    ctrl, data = _run(model)
    lim = InsertionLimits()
    assert ctrl.phase == Phase.SEATED, ctrl.events
    assert ctrl.retries == 0
    al = plug_alignment_error(model, data)
    assert al.lateral < 1.0e-3 and np.degrees(al.angle) < 1.0
    assert al.axial > -(0.036 - lim.seat_depth) - 1e-4
    assert ctrl.peak["contact_force"] < lim.contact_force_soft
    assert ctrl.peak["cable_tension"] < lim.cable_tension_max
    assert ctrl.peak["strain_relief"] < lim.strain_relief_max
    # Insertion really was slow: pin engagement lasted at least depth / speed.
    t_insert = next(t for t, e in ctrl.events if e.startswith("insert"))
    t_seat = next(t for t, e in ctrl.events if e.startswith("seated"))
    assert t_seat - t_insert >= lim.seat_depth / lim.insert_speed


def test_small_perception_error_is_absorbed_by_pin_chamfers(scene):
    """3 mm estimate error (> 2 mm clearance): the domed pin tips guide the plug in."""
    model, _ = scene
    ctrl, _ = _run(model, offset_mm=3.0)
    assert ctrl.phase == Phase.SEATED, ctrl.events
    assert ctrl.peak["contact_force"] < InsertionLimits().contact_force_max


def test_large_perception_error_jams_and_aborts_safely(scene):
    """6 mm estimate error: pins land on the socket rim, soft force limit detects the jam."""
    model, _ = scene
    ctrl, data = _run(model, offset_mm=6.0, max_time=60.0)
    lim = InsertionLimits()
    reasons = [e for _, e in ctrl.events if e.startswith("retract")]
    assert any("jam" in r for r in reasons), ctrl.events
    assert ctrl.phase == Phase.ABORTED
    assert ctrl.peak["contact_force"] < lim.contact_force_max
    assert plug_alignment_error(model, data).axial < -0.03  # backed out, never seated


def test_unexpected_deflection_aborts(scene):
    """4 mm error: the chamfers push the plug >2.5 mm off the believed axis -> treated as unsafe."""
    model, _ = scene
    ctrl, _ = _run(model, offset_mm=4.0, max_time=60.0)
    assert any("alignment lost" in e for _, e in ctrl.events), ctrl.events
    assert ctrl.phase == Phase.ABORTED


def test_hard_contact_force_limit_triggers_retract(scene):
    """With the soft pause disabled, the hard force limit alone must stop a jammed insertion."""
    model, _ = scene
    limits = dataclasses.replace(
        InsertionLimits(), contact_force_soft=1e9, contact_force_max=15.0, max_retries=0
    )
    ctrl, _ = _run(model, limits=limits, offset_mm=6.0, max_time=30.0)
    assert any("contact force" in e for _, e in ctrl.events), ctrl.events
    assert ctrl.phase == Phase.ABORTED
    # Filtered force overshoots the limit by at most a few control steps.
    assert ctrl.peak["contact_force"] < 25.0


def test_cable_tension_limit_triggers_retract(scene):
    model, _ = scene
    limits = dataclasses.replace(InsertionLimits(), cable_tension_max=3.0, max_retries=0)
    ctrl, _ = _run(model, limits=limits, max_time=20.0)
    assert any("cable tension" in e for _, e in ctrl.events), ctrl.events
    assert ctrl.phase == Phase.ABORTED


def test_strain_relief_limit_triggers_retract(scene):
    model, _ = scene
    limits = dataclasses.replace(InsertionLimits(), strain_relief_max=2.0, max_retries=0)
    ctrl, _ = _run(model, limits=limits, max_time=20.0)
    assert any("strain relief" in e for _, e in ctrl.events), ctrl.events


def test_emergency_stop_freezes_arm(scene):
    model, _ = scene
    state = {}

    def on_step(ctrl, data):
        if ctrl.phase == Phase.INSERT and "stopped" not in state:
            ctrl.emergency_stop(data)
            state["stopped"] = data.site_xpos[model.site("plug_face").id].copy()
            state["t"] = data.time

    ctrl, data = _run(model, max_time=30.0, on_step=on_step)
    assert ctrl.phase == Phase.ESTOP
    for _ in range(int(1.0 / model.opt.timestep)):
        ctrl.step(data)
        mujoco.mj_step(model, data)
    drift = np.linalg.norm(data.site_xpos[model.site("plug_face").id] - state["stopped"])
    assert drift < 0.003
