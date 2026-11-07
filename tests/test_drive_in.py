"""Drive-in mission: the robot starts 3 m away, drives to the car, docks, aligns and plugs in."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from autocharge import cable_metrics, plug_alignment_error, reset_to_home
from autocharge.base_controller import DriveLimits, DrivePhase
from autocharge.mission import DRIVE_IN_PARAMS, ChargingMission
from autocharge.scene import build_model


@pytest.fixture(scope="module")
def mobile_scene():
    return build_model(DRIVE_IN_PARAMS)


def _base_cable_contacts(model, data, base_id) -> int:
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.flex[0] >= 0) != (c.flex[1] >= 0):
            g = c.geom[1] if c.flex[0] >= 0 else c.geom[0]
            if g >= 0 and model.geom_bodyid[g] == base_id:
                n += 1
    return n


def test_drive_in_from_3m_docks_aligns_and_seats(mobile_scene):
    model, _ = mobile_scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    mission = ChargingMission(model, data)
    start_dist, _, _ = mission.drive.errors(data)
    assert start_dist >= 3.0

    end_site, sr_site = model.site("cable_end").id, model.site("strain_relief").id
    base_id = model.body("base_link").id
    max_gap = peak_t = peak_sr = 0.0
    base_cable = 0
    step = 0
    while not mission.done and data.time < 70.0:
        mission.step(data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), f"diverged at t={data.time}"
        step += 1
        if step % 25 == 0:
            max_gap = max(max_gap, float(np.linalg.norm(data.site_xpos[end_site] - data.site_xpos[sr_site])))
            c = cable_metrics(model, data)
            peak_t, peak_sr = max(peak_t, c.max_tension), max(peak_sr, c.strain_relief_load)
            base_cable += _base_cable_contacts(model, data, base_id)

    events = [e for _, e in mission.events]
    assert mission.drive.phase == DrivePhase.DOCKED, events
    assert mission.drive.travelled >= 3.0
    dist, _, yaw_err = mission.drive.errors(data)
    assert dist < 0.012 and abs(math.degrees(yaw_err)) < 0.6  # parked accurately

    assert mission.phase == "seated", events
    al = plug_alignment_error(model, data)
    assert al.lateral < 1.5e-3 and math.degrees(al.angle) < 1.0
    assert max_gap < 0.003  # cable never pulled off the strain relief
    assert peak_t < 60.0 and peak_sr < 45.0
    assert not any(data.warning[i].number for i in range(len(data.warning)))
    print(f"base-cable contact samples while driving: {base_cable}")


def test_drive_stops_when_cable_pulls_too_hard(mobile_scene):
    model, _ = mobile_scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    limits = dataclasses.replace(DriveLimits(), cable_tension_max=6.0)
    mission = ChargingMission(model, data, drive_limits=limits)
    while not mission.done and data.time < 30.0:
        mission.step(data)
        mujoco.mj_step(model, data)
    assert mission.drive.phase == DrivePhase.STOPPED
    assert any("cable tension" in e for _, e in mission.events)
    assert mission.insertion is None  # the arm never starts
    # Brakes hold: the base barely moves in the next second.
    x0 = np.array(mission.drive.pose(data)[:2])
    for _ in range(int(1.0 / model.opt.timestep)):
        mission.step(data)
        mujoco.mj_step(model, data)
    assert np.linalg.norm(np.array(mission.drive.pose(data)[:2]) - x0) < 0.01
