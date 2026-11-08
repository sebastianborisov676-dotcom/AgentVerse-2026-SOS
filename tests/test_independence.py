"""Proof-of-independence experiments: the robot acts on what it senses, not on hidden answers.

Success is judged by the independent evaluator (``autocharge.evaluation``), which reads the
physical socket body pose; the robot never imports it.
"""

import mujoco
import numpy as np

from autocharge import reset_to_home
from autocharge.autonomy import AgentState, AutonomousAgent
from autocharge.evaluation import check_docking
from autocharge.scenarios import Occlusion, ScenarioManager


def _run(model, max_time=45.0, **scenario):
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    sc = ScenarioManager(model, data, **scenario)
    mujoco.mj_forward(model, data)
    agent = AutonomousAgent(model, data)
    try:
        while not agent.done and data.time < max_time:
            sc.step(data)
            agent.step(data)
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all()
    finally:
        sc.restore()
    return agent, data


def test_ground_truth_poisoning_does_not_change_behaviour(scene):
    """Corrupt the simulator's socket 'answer key' site by 2 cm. A robot that secretly read it would
    miss; ours docks anyway because it only uses its camera."""
    model, _ = scene
    sid = model.site("socket_face").id
    original = model.site_pos[sid].copy()
    model.site_pos[sid] = original + np.array([0.02, 0.02, 0.0])
    try:
        agent, data = _run(model)
        verdict = check_docking(model, data)
    finally:
        model.site_pos[sid] = original
    assert agent.state == AgentState.SUCCESS, agent.events
    assert verdict.docked, verdict


def test_success_claim_matches_independent_checker(scene):
    model, _ = scene
    agent, data = _run(model)
    verdict = check_docking(model, data)
    assert agent.state == AgentState.SUCCESS and verdict.docked, (agent.events, verdict)
    assert verdict.depth_mm > 30 and verdict.lateral_mm < 2.0


def test_camera_blocked_mid_approach_holds_then_reacquires(scene):
    """Ablation: withhold observations during the approach. The robot pauses instead of carrying
    on blind, then resumes when the inlet is visible again."""
    model, _ = scene
    agent, data = _run(model, occlusions=(Occlusion(1.5, 3.0),))
    decisions = [e for _, e in agent.decisions]
    assert any(e.startswith("hold") for e in decisions), decisions
    assert any(e.startswith("resume") for e in decisions), decisions
    assert agent.state == AgentState.SUCCESS and check_docking(model, data).docked


def test_dim_lighting_still_docks(scene):
    model, _ = scene
    agent, data = _run(model, light_scale=0.15)
    assert agent.state == AgentState.SUCCESS, agent.events
    assert check_docking(model, data).docked


def test_too_dark_refuses_to_move(scene):
    model, _ = scene
    agent, _ = _run(model, light_scale=0.03, max_time=10.0)
    assert agent.state == AgentState.FAILED and agent.ctrl is None


def test_closed_flap_refuses_insertion_without_false_success(scene):
    """The inlet is visible but the flap is shut: pins meet the flap, force sensing detects the jam,
    the robot backs out, retries a bounded number of times and never claims success."""
    model, _ = scene
    agent, data = _run(model, flap_closed=True, max_time=80.0)
    assert agent.state in (AgentState.FAILED, AgentState.SAFE_STOP), agent.events
    assert not check_docking(model, data).docked
    assert any("jam" in e or "contact force" in e for _, e in agent.events), agent.events
    assert agent.ctrl.peak["contact_force"] < 70.0 * 1.5
