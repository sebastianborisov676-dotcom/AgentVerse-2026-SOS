"""Perception-driven autonomy: the robot finds the inlet with its camera and adapts to changes.

The agent never reads the simulator's socket pose; these tests use ground truth only to score.
"""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from autocharge import InsertionLimits, plug_alignment_error, reset_to_home
from autocharge.autonomy import AgentState, AutonomousAgent, socket_estimate_error
from autocharge.perception import PortPerception, evaluate
from autocharge.scenarios import CarMove, CarPose, Occlusion, ScenarioManager


def _run(model, park=None, moves=(), occlusions=(), max_time=45.0, agent_kwargs=None):
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    sc = ScenarioManager(model, data, park=park, moves=moves, occlusions=occlusions)
    mujoco.mj_forward(model, data)
    agent = AutonomousAgent(model, data, **(agent_kwargs or {}))
    while not agent.done and data.time < max_time:
        sc.step(data)
        agent.step(data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
    sc.cover_marker(False)
    return agent, data


def test_camera_pose_estimate_matches_ground_truth(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    est = PortPerception(model).observe(data)
    assert est.detected and est.confidence > 0.5
    pos_err, ang_err = evaluate(model, data, est)
    assert pos_err < 0.002 and math.degrees(ang_err) < 1.0


def test_agent_seats_from_camera_only(scene):
    model, _ = scene
    agent, data = _run(model)
    assert agent.state == AgentState.SUCCESS, agent.events
    assert plug_alignment_error(model, data).lateral < 1.5e-3
    pe, ae = socket_estimate_error(model, data, agent)
    assert pe < 2.0 and ae < 1.0


@pytest.mark.parametrize("seed", [3, 11])
def test_random_parking_positions(scene, seed):
    model, _ = scene
    agent, data = _run(model, park=CarPose.random(np.random.default_rng(seed)))
    assert agent.state == AgentState.SUCCESS, agent.events
    assert plug_alignment_error(model, data).lateral < 1.5e-3


def test_car_moves_during_approach_robot_replans(scene):
    model, _ = scene
    agent, data = _run(model, moves=(CarMove(at=3.0, to=CarPose(0.0, 0.03, 0.0, 3.0)),))
    events = [e for _, e in agent.events]
    assert any("car moved" in e for e in events), events
    assert agent.state == AgentState.SUCCESS, events
    assert plug_alignment_error(model, data).lateral < 1.5e-3


def test_blindfold_never_moves_and_fails_safely(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    plug, sock = model.site("plug_face").id, model.site("socket_face").id
    start_gap = np.linalg.norm(data.site_xpos[plug] - data.site_xpos[sock])
    agent, data = _run(model, occlusions=(Occlusion(0.0, 1e9),), max_time=10.0)
    assert agent.state == AgentState.FAILED
    assert agent.ctrl is None  # never planned a motion
    # The arm only sags passively under the cable's weight; it never closes in on the socket.
    assert np.linalg.norm(data.site_xpos[plug] - data.site_xpos[sock]) > start_gap - 0.005


def test_wrong_calibration_misleads_robot(scene, monkeypatch):
    """Causality check: shift the marker-to-socket calibration by 8 mm and the robot follows its
    (now wrong) perception instead of the true socket, so it never seats."""
    import autocharge.perception as perception

    monkeypatch.setattr(perception, "MARKER_POS_S", perception.MARKER_POS_S + np.array([0.008, 0.0, 0.0]))
    model, _ = scene
    limits = dataclasses.replace(InsertionLimits(), max_retries=1)
    agent, data = _run(model, max_time=30.0, agent_kwargs={"limits": limits})
    assert agent.state != AgentState.SUCCESS, agent.events
    assert plug_alignment_error(model, data).axial < -0.03


def test_car_moves_while_engaged_safe_stop(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    sc = ScenarioManager(model, data)
    mujoco.mj_forward(model, data)
    agent = AutonomousAgent(model, data)
    moved = False
    while not agent.done and data.time < 40.0:
        if not moved and agent.ctrl is not None and agent.ctrl.engaged(data) and agent.phase == "insert":
            sc.moves = [CarMove(at=float(data.time), to=CarPose(0.0, 0.01, 0.0, 0.0), duration=0.5)]
            moved = True
        sc.step(data)
        agent.step(data)
        mujoco.mj_step(model, data)
    assert moved
    assert agent.state == AgentState.SAFE_STOP, agent.events
