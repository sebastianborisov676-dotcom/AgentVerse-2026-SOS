import dataclasses
import itertools

import mujoco
import numpy as np
import pytest

from autocharge.mission import DRIVE_IN_PARAMS
from autocharge.scene import build_model, reset_to_home
from autocharge.training.course import (
    GOAL,
    START,
    cable_lane_clear,
    make_course,
    radius,
    validate,
)
from autocharge.training.environment import CourseEnv
from autocharge.training.levels import LEVELS, Obstacle
from autocharge.training.navigation import (
    NoPathError,
    course_path,
    path_clear,
    plan_path,
    segment_clear,
)


@pytest.mark.parametrize("level", range(1, 11))
def test_seeded_levels(level):
    for seed in (0, 1, 42):
        a = make_course(level, seed)
        assert a == make_course(level, seed)
        assert len(a) == LEVELS[level]["count"]
        validate(a)
        assert {o.kind for o in a} == set(LEVELS[level]["types"])
        path = course_path(START, np.radians(15), GOAL, 0.0, a)
        assert path_clear(path, a, 0.65)
        assert segment_clear((-0.8, 0), GOAL, a, 0.65)
        assert all(cable_lane_clear(o) for o in a)
        if a:
            # At least one obstacle constrains the lane, even at sparse levels.
            assert (
                min(
                    np.min(np.linalg.norm(path - [o.x, o.y], axis=1)) - radius(o)
                    for o in a
                )
                < 0.81
            )


def test_manual_and_hybrid_validation():
    manual = (Obstacle("box", -1.7, 1.7, 0.1),)
    assert make_course(5, 0, manual, "manual") == manual
    assert make_course(5, 0, manual, "hybrid")[0] == manual[0]
    with pytest.raises(ValueError):
        make_course(4, 0, manual, "hybrid")
    with pytest.raises(ValueError):
        validate([Obstacle("box", -3, 0.5, 0.2)])
    with pytest.raises(ValueError):
        validate([Obstacle("box", float("nan"), 0, 0.2)])


def test_planner_detours_and_rejects_blocked_start():
    obs = (Obstacle("box", -1.7, 0.3, 0.22),)
    path = plan_path((-3, 0.5), (0, 0), obs)
    assert np.allclose(path[0], (-3, 0.5)) and np.allclose(path[-1], (0, 0))
    assert all(segment_clear(a, b, obs) for a, b in itertools.pairwise(path))
    with pytest.raises(NoPathError):
        plan_path((-3, 0.5), (0, 0), (Obstacle("box", -3, 0.5, 0.2),))


def test_all_obstacle_shapes_have_physical_contacts():
    obs = tuple(
        Obstacle(kind, -1.7, y, 0.1)
        for kind, y in zip(
            ("box", "cone", "barrier", "stone", "slope"), (-2, -1, 0, 1, 2)
        )
    )
    model, _ = build_model(dataclasses.replace(DRIVE_IN_PARAMS, obstacles=obs))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    for i in range(5):
        geom = model.geom(f"obstacle_{i}_geom").id
        assert model.geom_contype[geom] == 3 and model.geom_conaffinity[geom] == 3
        assert model.geom_bodyid[geom] != 0
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


def test_environment_seed_reset_and_action_validation():
    env = CourseEnv(1, max_time=0.04)
    obs, info = env.reset(7)
    assert info["seed"] == 7 and obs["obstacles"].shape == (0, 4)
    with pytest.raises(ValueError):
        env.step([float("nan"), 0])
    while not (env.terminated or env.truncated):
        env.step([0, 0])
    assert env.truncated
    assert env.report()["failure_reason"] == "episode simulated-time limit"
    env.truncation_reason = "episode wall-clock limit"
    assert env.report()["failure_reason"] == "episode wall-clock limit"
    with pytest.raises(RuntimeError):
        env.step()


def test_manual_blocked_final_approach_is_rejected():
    with pytest.raises(ValueError, match="final docking corridor"):
        make_course(5, 0, (Obstacle("barrier", -1.0, -0.1, 0.2),), "manual")


def test_startup_cable_overlap_is_detected(monkeypatch):
    # The actual offending Level 7 seed 0 slope from the first evaluation.
    import autocharge.training.environment as module

    monkeypatch.setattr(
        module,
        "make_course",
        lambda *args: (Obstacle("slope", -0.858, -1.164, 0.5, 15),),
    )
    env = CourseEnv(7)
    env.reset(0)
    assert env.terminated
    report = env.report()
    assert report["failure_category"] == "initial_geometry"
    assert report["initial_obstacle_penetrations"]
    assert report["settling"]["contacts"]["pair_episodes"]["obstacle_cable"] > 0
    assert report["settling"]["cable"]["peak_tension_N"] > 60
    with pytest.raises(RuntimeError):
        env.step()


def test_contact_episodes_measure_actual_forces_and_reentry():
    from autocharge.training.telemetry import ContactTelemetry

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <body name="base_link" pos="0 0 .3"><freejoint/>
        <geom name="robot" type="sphere" size=".2"/>
        <geom name="wheel" type="cylinder" size=".06 .01" pos="0 .5 0"/>
        <body name="connector"><geom name="plug" size=".01" contype="0" conaffinity="0"/></body>
      </body>
      <body name="obstacle_0" pos=".2 0 .3"><geom name="obstacle_0_geom" size=".15"/></body>
      <body name="vehicle" pos="10 0 0"><geom name="car" size=".1"/><body name="socket"><geom name="inlet" size=".05"/></body></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    telemetry = ContactTelemetry(model)
    mujoco.mj_forward(model, data)
    telemetry.sample(data)
    assert telemetry.episodes["obstacle_robot"] == 1
    assert telemetry.peaks["obstacle_robot"] > 0
    telemetry.sample(data)
    assert telemetry.episodes["obstacle_robot"] == 1
    data.qpos[0] = -2
    mujoco.mj_forward(model, data)
    telemetry.sample(data)
    data.qpos[0] = 0
    mujoco.mj_forward(model, data)
    telemetry.sample(data)
    assert telemetry.episodes["obstacle_robot"] == 2
    assert (
        telemetry.classify(model.geom("floor").id, model.geom("wheel").id)
        == "wheel_floor"
    )
    assert (
        telemetry.classify(model.geom("plug").id, model.geom("inlet").id)
        == "connector_socket"
    )

    assert (
        telemetry.classify(model.geom("plug").id, model.geom("car").id)
        == "unsafe_robot_environment"
    )


def test_load_telemetry_matches_existing_metrics(scene):
    from autocharge.base_controller import DriveLimits
    from autocharge.metrics import cable_metrics
    from autocharge.training.telemetry import EpisodeTelemetry

    model, _ = scene
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    telemetry = EpisodeTelemetry(model)
    for _ in range(5):
        cab = telemetry.sample(data, DriveLimits())
        expected = cable_metrics(model, data)
        assert cab.max_tension == pytest.approx(expected.max_tension)
        assert cab.strain_relief_load == pytest.approx(expected.strain_relief_load)
        mujoco.mj_step(model, data)


def test_editor_errors_identify_geometry_and_conflicting_entries():
    from autocharge.training.course import validate
    from autocharge.training.levels import Obstacle

    with pytest.raises(ValueError, match=r"Obstacle 1 .*protected start/dock"):
        validate([Obstacle("barrier", -0.3, 0.1, 0.35, -87)], protect_start=False)
    with pytest.raises(ValueError, match=r"Obstacle 2 .*overlaps obstacle 1 .*0.081 m"):
        validate([
            Obstacle("slope", -1.54, 0.26, 0.5, -87),
            Obstacle("barrier", -1.46, 0.27, 0.35, -87),
        ], protect_start=False)


def test_master_hybrid_preserves_manual_and_repeats_seed():
    import math
    from autocharge.panel.scenario import Config, start_pose
    from autocharge.panel.drive import operator_path
    from autocharge.panel.placement import footprint, cross

    manual = Obstacle('box', -3.4, 3.8, 0.10)
    pose = start_pose(Config(level=11, seed=42))
    kwargs = dict(level=11, seed=42, manual=(manual,), mode='hybrid',
                  protect_start=False, approach=(pose[:2], math.radians(pose[2])))
    course = make_course(**kwargs)
    assert course == make_course(**kwargs)
    assert course[0] == manual
    assert len(course) == 4
    assert {o.kind for o in course[1:]} == {'box', 'stone', 'slope'}
    initial = operator_path(pose[:2], math.radians(pose[2]), GOAL, 0, [])
    polygon = footprint(course[1])
    edges = np.roll(polygon, -1, axis=0) - polygon
    signs = cross(edges[None], initial[:, None] - polygon)
    assert np.any(np.all(signs >= -1e-9, axis=1) | np.all(signs <= 1e-9, axis=1))
    assert path_clear(operator_path(pose[:2], math.radians(pose[2]), GOAL, 0, course), course, .5)
