"""LiDAR-based obstacle avoidance for the mobile base (car park scenarios).

The robot senses obstacles only through its simulated planar LiDAR (ray ranges); the bag and
neighbour-car positions are known only to the scenario manager and to these assertions.
"""

import mujoco
import numpy as np
import pytest

from autocharge import reset_to_home
from autocharge.autonomy import AgentState, AutonomousAgent
from autocharge.evaluation import check_docking
from autocharge.scenarios import CarPose, ScenarioManager
from autocharge.scene import SceneParams, build_model

OBSTACLES = ("bin_bag", "bin_bag_2", "nb_front_body", "nb_back_body", "nb_side_body")


@pytest.fixture(scope="module")
def mobile():
    return build_model(SceneParams(mobile_base=True, base_start=(0.0, 0.0, 0.0), cable_segments=64))


def _run(model, park, bags=(), neighbours=False, max_time=120.0):
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    sc = ScenarioManager(model, data, park=park)
    sc.set_bags(*bags)
    sc.set_neighbours(data, neighbours)
    mujoco.mj_forward(model, data)
    agent = AutonomousAgent(model, data)
    ids = {model.geom(n).id for n in OBSTACLES}
    contacts = 0
    base0 = data.xpos[model.body("base_link").id][:2].copy()
    try:
        while not agent.done and data.time < max_time:
            sc.step(data)
            agent.step(data)
            mujoco.mj_step(model, data)
            contacts += sum(1 for c in data.contact[:data.ncon] if c.geom[0] in ids or c.geom[1] in ids)
        moved = float(np.linalg.norm(data.xpos[model.body("base_link").id][:2] - base0))
    finally:
        sc.restore()
    return agent, data, contacts, moved


def test_bin_bag_detour_without_contact(mobile):
    model, _ = mobile
    agent, data, contacts, _ = _run(model, CarPose(1.5, 0.0, 0.0, 0.0), bags=((0.8, 0.0),))
    events = [e for _, e in agent.events]
    assert any("LiDAR: obstacle on the direct route" in e for e in events), events
    assert agent.state == AgentState.SUCCESS and check_docking(model, data).docked, events
    assert contacts == 0


def test_same_spot_without_bag_drives_direct(mobile):
    model, _ = mobile
    agent, data, _, _ = _run(model, CarPose(1.5, 0.0, 0.0, 0.0))
    events = [e for _, e in agent.events]
    assert any("direct route clear" in e for e in events), events
    assert agent.state == AgentState.SUCCESS


def test_three_neighbour_cars_no_contact(mobile):
    model, _ = mobile
    agent, data, contacts, _ = _run(model, CarPose(0.9, -0.6, 0.0, 0.0), neighbours=True)
    assert agent.state == AgentState.SUCCESS and check_docking(model, data).docked, agent.events
    assert contacts == 0


def test_bag_on_parking_spot_refuses_without_driving(mobile):
    model, _ = mobile
    agent, _, contacts, moved = _run(model, CarPose(1.5, 0.0, 0.0, 0.0), bags=((1.5, -0.1),), max_time=20.0)
    assert agent.state == AgentState.FAILED and agent.unreachable, agent.events
    assert moved < 0.05 and contacts == 0
