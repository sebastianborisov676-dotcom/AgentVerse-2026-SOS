"""Operator experiments: physical budget, sensor boundaries, safety, and replay."""

import dataclasses

import mujoco
import numpy as np
import pytest

from autocharge import reset_to_home
from autocharge.controller import InsertionController, Phase
from autocharge.panel.adaptive import AdaptiveMission
from autocharge.panel.scenario import CABLE_LENGTH, Config, build_scenario, start_pose
from autocharge.panel.sensors import Sensors
from autocharge.panel.session import Recording, Session
from autocharge.mission import DRIVE_IN_PARAMS
from autocharge.scene import build_model
from autocharge.training.levels import Obstacle


@pytest.mark.parametrize("level", range(1, 11))
def test_seeded_progressive_start_envelope(level):
    config = Config(level=level, seed=42)
    pose = start_pose(config)
    assert pose == start_pose(config)
    assert pose != start_pose(dataclasses.replace(config, seed=43))
    nominal = 2.5 + 0.04 * (level - 1)
    assert abs(-pose[0] - nominal) <= 0.5
    assert abs(pose[1] - 0.5) <= 0.3
    assert abs(pose[2] - 15) <= 5
    if level > 1:
        assert -pose[0] > -start_pose(Config(level=level - 1, seed=42))[0]


@pytest.fixture(scope="module")
def panel_scene():
    return build_scenario(Config())


def test_fixed_physical_cable_and_infeasible_start(panel_scene):
    model, data, info, _, _ = panel_scene
    assert info.cable_length == pytest.approx(CABLE_LENGTH, abs=1e-8)
    assert np.sum(model.flexedge_length0) == pytest.approx(CABLE_LENGTH, abs=1e-5)
    with pytest.raises(ValueError, match="infeasible"):
        build_scenario(Config(level=10, seed=0))


def test_same_seed_initial_state_and_prefix_repeatability(tmp_path):
    session = Session(tmp_path)
    session.configure(dict(level=1, seed=42))
    initial = session.data.qpos.copy()
    session.status = "running"
    session.tick(50)
    first = session.data.qpos.copy()
    session.action("restart", {})
    np.testing.assert_array_equal(initial, session.data.qpos)
    session.status = "running"
    session.tick(50)
    np.testing.assert_array_equal(first, session.data.qpos)
    session.close()


def test_manual_edits_are_new_attempts_and_invalid_reset_is_transactional(tmp_path):
    session = Session(tmp_path)
    values = dict(
        level=5,
        seed=42,
        mode="manual",
        manual=[dict(kind="box", x=-1.5, y=2.0, size=0.2, angle=0)],
    )
    session.configure(values)
    attempt = session.record.id
    values["manual"][0]["y"] = 2.3
    session.configure(values)
    assert session.record.id != attempt and session.records[attempt].frames
    assert session.obstacles[0].y == 2.3
    retained = session.model
    values["manual"][0]["x"] = float("nan")
    with pytest.raises(ValueError):
        session.configure(values)
    assert session.model is retained
    saved = session.action("save_layout", {"name": "test_layout"})
    assert saved["layout"]["obstacles"][0]["y"] == 2.3
    session.close()


def test_ray_observations_have_range_occlusion_and_no_semantic_map():
    xml = """<mujoco><worldbody>
      <body name="base_link"><joint name="base_x" type="slide" axis="1 0 0"/>
      <joint name="base_y" type="slide" axis="0 1 0"/><joint name="base_yaw" axis="0 0 1"/>
      <geom size=".1" mass="1"/></body>
      <body pos="1.1 0 .2"><geom name="near" type="box" size=".1 .5 1"/></body>
      <body pos="1.8 0 .2"><geom name="occluded" type="box" size=".1 .05 .3"/></body>
      <body pos="4 0 .2"><geom name="far" size=".1"/></body>
      <body pos="3 0 .8"><site name="socket_face"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor = Sensors(model)
    points = sensor.scan(data)
    assert points
    assert all(p.x < 1.5 for p in points)  # rear and far objects never revealed
    assert sensor.port(data) is None  # near box occludes ideal marker
    assert all(
        p.kind == "stone" and p.size == 0.07 for p in points
    )  # generic hit cells, not object extents/types


def test_real_sensor_observation_changes_route():
    # The supervisor never receives this obstacle list; it must discover geometry with rays.
    obstacle = Obstacle("cone", -1.6, 0.6, 0.12)
    model, _ = build_model(dataclasses.replace(DRIVE_IN_PARAMS, obstacles=(obstacle,)))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ctrl = AdaptiveMission(model, data, 5, 42, False)
    for _ in range(300):
        ctrl.step(data)
        mujoco.mj_step(model, data)
        if ctrl.replans:
            break
    assert ctrl.sensors.hit_count > 0
    assert ctrl.replans > 0, ctrl.events
    assert any("observed cells" in e for _, e in ctrl.events)


def test_moving_vehicle_uses_joints_and_motion_is_continuous():
    model, data, info, _, _ = build_scenario(Config(level=9, seed=42))
    ctrl = AdaptiveMission(model, data, 9, 42)
    assert model.body_mocapid[model.body("vehicle").id] < 0
    assert ctrl.request_vehicle_motion(data)
    previous = data.qpos[ctrl.vehicle.qadr].copy()
    seen_motion = False
    for _ in range(600):
        ctrl.step(data)
        mujoco.mj_step(model, data)
        current = data.qpos[ctrl.vehicle.qadr].copy()
        assert np.max(np.abs(current - previous)) < 0.002
        seen_motion |= np.linalg.norm(current) > 1e-4
        if ctrl.sensors.port_speed > 0.001:
            assert ctrl.mission.insertion is None
        previous = current
    assert seen_motion and ctrl.waiting
    assert any("port motion detected" in e for _, e in ctrl.events)


def test_motion_request_during_insertion_retracts_and_inhibits_car():
    model, data, _, _, _ = build_scenario(Config(level=9, seed=42))
    ctrl = AdaptiveMission(model, data, 9, 42)
    ctrl.mission.insertion = InsertionController(
        model, data, socket_estimate=ctrl.sensors.last_port
    )
    ctrl.mission.insertion.phase = Phase.INSERT
    assert not ctrl.request_vehicle_motion(data)
    assert not ctrl.vehicle.moving and not ctrl.vehicle.started
    assert ctrl.recovering and ctrl.mission.insertion.phase == Phase.RETRACT


def test_visual_playback_and_complete_in_process_checkpoint(tmp_path):
    session = Session(tmp_path)
    session.configure(dict(level=1, seed=42))
    session.status = "running"
    session.tick(30)
    index = session.record.capture(session, checkpoint=True)
    expected_q = session.data.qpos.copy()
    visual = session.record.playback(index)
    np.testing.assert_array_equal(visual.qpos, expected_q)
    session.tick(40)
    future = session.data.qpos.copy()
    future_ctrl = session.data.ctrl.copy()
    old = session.record
    path = old.save(tmp_path / "runs")
    loaded = Recording.load(path)
    np.testing.assert_array_equal(loaded.playback(index).qpos, expected_q)
    session.action("restore", {"index": index})
    assert session.record.id != old.id
    session.status = "running"
    session.tick(40)
    np.testing.assert_array_equal(session.data.qpos, future)
    np.testing.assert_array_equal(session.data.ctrl, future_ctrl)
    session.close()


@pytest.mark.parametrize("level", [1, 9])
def test_operator_full_mission_and_moving_port_gate(tmp_path, level):
    session = Session(tmp_path)
    session.configure(dict(level=level, seed=42))
    session.status = "running"
    while session.status == "running":
        session.tick(25)
        ins = session.controller.mission.insertion
        if ins is not None and ins.phase == Phase.INSERT:
            assert not session.controller.vehicle.moving
            assert not session.controller.port_moving
    assert session.controller.phase == "seated", (
        session.controller.mission.events + session.controller.events
    )
    assert not any(session.telemetry.violations.values())
    # A trailing cable may touch course geometry; its force caps still apply.
    # Robot collisions, self contacts and unsafe environment contacts remain forbidden.
    assert not any(
        session.telemetry.contacts.episodes[c]
        for c in (
            "obstacle_robot",
            "robot_self",
            "unsafe_robot_environment",
        )
    )
    if level == 9:
        assert session.controller.replans >= 1
        assert session.controller.vehicle.started
        events = session.controller.events + session.controller.mission.events
        first = lambda prefix: next(t for t, event in events if event.startswith(prefix))
        assert first("avoidance replan:") < first("vehicle motion started:")
        assert first("vehicle motion started:") < first("stationary port confirmed;")
        assert first("stationary port confirmed;") < first("insert:")
    session.close()


def test_detected_external_port_motion_during_insertion_uses_existing_retraction():
    # Valid home alignment, before pin engagement: inject an external actor command
    # that bypasses the operator interlock to exercise the independent observation guard.
    model, _ = build_model(
        dataclasses.replace(
            DRIVE_IN_PARAMS,
            base_start=(0.0, 0.0, 0.0),
            movable_vehicle=True,
            operator_cameras=True,
            cable_length=CABLE_LENGTH,
        )
    )
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ctrl = AdaptiveMission(model, data, 9, 42, False)
    ctrl.mission.insertion = InsertionController(
        model, data, socket_estimate=ctrl.sensors.last_port
    )
    ctrl.mission.insertion.phase = Phase.INSERT
    ctrl.vehicle.request(data)
    for _ in range(200):
        ctrl.step(data)
        mujoco.mj_step(model, data)
        if ctrl.recovering:
            break
    assert ctrl.recovering, ctrl.events
    assert not ctrl.vehicle.moving
    assert ctrl.mission.insertion.phase == Phase.RETRACT
    assert any("port motion detected" in text for _, text in ctrl.events)


def test_restore_original_checkpoint_repeatedly_and_across_scene_models(tmp_path):
    session = Session(tmp_path)
    session.configure(dict(level=1, seed=42))
    source = session.record
    session.status = "running"
    session.tick(30)
    index = source.capture(session, checkpoint=True)
    session.tick(40)
    expected = session.data.qpos.copy()
    for _ in range(2):
        session.action("restore", {"id": source.id, "index": index})
        session.action("resume", {})
        session.tick(40)
        np.testing.assert_array_equal(session.data.qpos, expected)
    session.configure(dict(level=9, seed=42))
    assert session.model is not source.model
    session.action("restore", {"id": source.id, "index": index})
    assert session.model is source.model and session.config.level == 1
    assert session.info.params.movable_vehicle is False
    session.action("resume", {})
    session.tick(40)
    np.testing.assert_array_equal(session.data.qpos, expected)
    source.save(tmp_path / "runs")
    disk = Recording.load(tmp_path / "runs" / (source.id + ".npz"))
    session.records[source.id] = disk
    with pytest.raises(ValueError, match="visual playback only"):
        session.action("restore", {"id": source.id, "index": index})
    session.close()


def test_ground_pick_projection_and_nearest_edge_clearance(panel_scene):
    from autocharge.panel.placement import (
        MIN_CLEARANCE,
        ground_point,
        project,
        placement_check,
        polygon_distance,
    )
    from autocharge.training.telemetry import ContactTelemetry

    assert MIN_CLEARANCE == 0.15
    forward = np.array([0.0, -3.0, -2.0])
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    view = dict(
        pos=[0.0, 3.0, 2.0],
        forward=forward.tolist(),
        up=up.tolist(),
        near=0.1,
        width=0.2,
        center=0.0,
        top=0.06,
        bottom=-0.06,
    )
    uv = project(view, [[-1.0, 0.0, 0.0]])[0]
    np.testing.assert_allclose(ground_point(view, *uv), [-1.0, 0.0], atol=1e-12)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert polygon_distance(square, square + [1.2, 0]) == pytest.approx(0.2)
    assert polygon_distance(square, square + [0.5, 0.5]) == 0
    model, data, _, _, _ = panel_scene
    q = data.qpos.copy()
    bodies = ContactTelemetry(model).robot
    pose = data.xpos[model.body("base_link").id]
    bad = placement_check(model, data, Obstacle("box", pose[0], pose[1], 0.22), bodies)
    assert not bad["valid"] and bad["clearance_m"]["robot"] == 0
    good = placement_check(model, data, Obstacle("box", -1.7, 2.86, 0.22), bodies)
    assert good["valid"]
    np.testing.assert_array_equal(data.qpos, q)


def test_batch_default_uses_panel_sensors_and_case_isolation(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_course_batch.py"
    spec = importlib.util.spec_from_file_location("course_batch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = dict(
        level=1,
        seed=42,
        engine="panel",
        max_time=0.6,
        wall_time=180,
        mode="random",
        manual=None,
        moving_car=True,
        random_start=True,
        alignment_offset_mm=0,
        directory=str(tmp_path / "case"),
    )
    row = module.run_episode(case)
    assert (
        row["pipeline"]
        == "autocharge.panel.Session -> build_scenario + AdaptiveMission"
    )
    assert row["config"]["seed"] == 42 and row["config"]["random_start"]
    np.testing.assert_allclose(row["start_pose"], start_pose(Config(level=1, seed=42)))
    assert row["truncated"] and not row["success"]
    assert (tmp_path / "case" / "scenario.json").exists() and (
        tmp_path / "case" / "result.json"
    ).exists()

@pytest.mark.parametrize('level,seed', [(6, 42), (6, 43), (9, 42), (11, 42), (11, 43)])
def test_generated_challenge_intersects_empty_sensor_route(level, seed):
    import math
    from autocharge.panel.drive import operator_path
    from autocharge.training.navigation import path_clear, segment_clear
    from autocharge.training.course import GOAL

    model, data, info, obstacles, _ = build_scenario(Config(level=level, seed=seed))
    pose = start_pose(Config(level=level, seed=seed))
    initial = operator_path(pose[:2], math.radians(pose[2]), GOAL, 0, [])
    # Route sample must lie inside the actual footprint, not just its safety halo.
    from autocharge.panel.placement import footprint, cross
    first = obstacles[0]
    polygon = footprint(first)
    edges = np.roll(polygon, -1, axis=0) - polygon
    signs = cross(edges[None, :, :], initial[:, None, :] - polygon)
    assert np.any(np.all(signs >= -1e-9, axis=1) | np.all(signs <= 1e-9, axis=1))
    assert not path_clear(initial, [first], clearance=0)
    assert segment_clear((-0.8, 0), GOAL, obstacles, clearance=0.65)
    detour = operator_path(pose[:2], math.radians(pose[2]), GOAL, 0, obstacles)
    assert path_clear(detour, obstacles, clearance=0.50)
    assert info.cable_length == pytest.approx(CABLE_LENGTH)
    assert np.sum(model.flexedge_length0) == pytest.approx(CABLE_LENGTH, abs=1e-5)
    if level == 11:
        assert {o.kind for o in obstacles} == {'box', 'stone', 'slope'}


def test_forward_actor_target_is_along_its_heading(panel_scene):
    from autocharge.panel.adaptive import VehicleMotion
    model, data, _, _, _ = build_scenario(Config(level=9, seed=42))
    actor = VehicleMotion(model, 42, True, forward=True)
    assert actor.request(data)
    delta = actor.destination - actor.origin
    assert 0.10 <= delta[0] <= 0.14
    assert delta[1] == delta[2] == 0
    assert not actor.request(data)
