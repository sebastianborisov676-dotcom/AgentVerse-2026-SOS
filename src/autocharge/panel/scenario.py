"""Seeded scenarios, fixed cable budget and optional actuated vehicle/cameras."""

import dataclasses
import math
import random

import mujoco
import numpy as np

from autocharge.mission import DRIVE_IN_PARAMS
from autocharge.scene import build_model, reset_to_home
from autocharge.training.course import GOAL, make_course, radius
from autocharge.training.levels import Obstacle
from autocharge.training.navigation import course_path
from autocharge.training.telemetry import EpisodeTelemetry
from autocharge.base_controller import DriveLimits

# Measured nominal SAI 64-segment mobile cable. Its length is fixed for all panel resets.
CABLE_LENGTH = 4.545351328240237


@dataclasses.dataclass
class Config:
    level: int = 1
    seed: int = 42
    mode: str = "random"
    manual: list = dataclasses.field(default_factory=list)
    moving_car: bool = True
    random_start: bool = True

    def validate(self):
        if not 1 <= self.level <= 11 or not isinstance(self.seed, int):
            raise ValueError("Level must be 1–11 and seed an integer")
        if self.mode not in ("random", "manual", "hybrid"):
            raise ValueError("Invalid obstacle mode")
        if self.level < 5 and (self.mode != "random" or self.manual):
            raise ValueError("Manual and hybrid placement start at Level 5")
        if len(self.manual) > 40:
            raise ValueError("At most 40 manual obstacles")
        return self


def start_pose(config):
    nominal = 2.24 if config.level == 11 else 2.5 + 0.04 * (config.level - 1)
    if not config.random_start:
        return (-nominal, 0.62, 12.0) if config.level == 11 else (-nominal, 0.5, 15.0)
    rng = random.Random(config.seed)
    variation = 0.05 if config.level == 11 else 0.5
    distance_delta = rng.uniform(-variation, variation)
    if config.level == 11:
        distance_delta = abs(distance_delta)
    return (
        -(nominal + distance_delta),
        (0.62 + rng.uniform(-0.05, 0.05)) if config.level == 11 else 0.5 + rng.uniform(-0.3, 0.3),
        (12 + rng.uniform(-2, 2)) if config.level == 11 else 15 + rng.uniform(-5, 5),
    )


def augment_spec(spec, params):
    if params.operator_cameras:
        spec.body("base_link").add_camera(
            name="robot_rgb", pos=[0.2, 0, 1.05], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75
        )
        socket = spec.body("socket")
        socket.add_site(
            name="port_marker", pos=[0.15, 0, 0.055], size=[0.002, 0, 0], group=4
        )
        socket.add_geom(
            name="port_marker_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.15, 0, 0.055],
            size=[0.04, 0.04, 0.001],
            mass=0,
            contype=0,
            conaffinity=0,
            rgba=[0.95, 0.95, 0.95, 1],
        )
        for index, (dx, dy) in enumerate(((-0.018, -0.018), (0.018, 0.018))):
            socket.add_geom(
                name=f"port_marker_tile_{index}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.15 + dx, dy, 0.0535],
                size=[0.015, 0.015, 0.0005],
                mass=0,
                contype=0,
                conaffinity=0,
                rgba=[0.04, 0.04, 0.04, 1],
            )
    if not params.movable_vehicle:
        return
    vehicle = spec.body("vehicle")
    # The scene declares the vehicle as a mocap body (judge scenarios teleport it);
    # the panel drives it through joints and position servos instead.
    vehicle.mocap = False
    # MuJoCo only allows plane geoms on static bodies: keep the ArUco marker as a thin,
    # massless, non-colliding box once the vehicle has joints.
    marker = spec.geom("aruco_marker")
    if marker is not None:
        marker.type = mujoco.mjtGeom.mjGEOM_BOX
        marker.size = [marker.size[0], marker.size[1], 0.0002]
        marker.mass = 0
    pivot = np.array(spec.geom("car_body").pos)
    pivot[2] = 0
    vehicle.pos = pivot
    for geom in vehicle.geoms:
        geom.pos = np.asarray(geom.pos) - pivot
    spec.body("socket").pos = np.asarray(spec.body("socket").pos) - pivot
    spec.geom("car_body").mass = 1200.0
    for name, kind, axis, kp, kv, force in (
        (
            "vehicle_x",
            mujoco.mjtJoint.mjJNT_SLIDE,
            [1, 0, 0],
            200000.0,
            40000.0,
            4000.0,
        ),
        (
            "vehicle_y",
            mujoco.mjtJoint.mjJNT_SLIDE,
            [0, 1, 0],
            200000.0,
            40000.0,
            4000.0,
        ),
        (
            "vehicle_yaw",
            mujoco.mjtJoint.mjJNT_HINGE,
            [0, 0, 1],
            200000.0,
            40000.0,
            4000.0,
        ),
    ):
        vehicle.add_joint(name=name, type=kind, axis=axis)
        a = spec.add_actuator(
            name=name + "_pos", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[1], a.biasprm[2] = -kp, -kv
        a.forcerange = [-force, force]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


def build_scenario(config):
    config.validate()
    manual = tuple(Obstacle(**o) for o in config.manual)
    pose = start_pose(config)
    approach = (pose[:2], math.radians(pose[2])) if config.level >= 2 and config.mode != "manual" else None
    candidate_check = None
    if approach is not None:
        # Validate route candidates against the actual reset robot/connector geometry,
        # rather than guessing clearance from the robot centre.
        from .placement import placement_check
        empty_params = dataclasses.replace(
            DRIVE_IN_PARAMS, base_start=pose, obstacles=(), cable_length=CABLE_LENGTH,
            movable_vehicle=config.level >= 9, operator_cameras=True,
            environment="classic",
        )
        empty_model, _ = build_model(empty_params)
        empty_data = mujoco.MjData(empty_model)
        reset_to_home(empty_model, empty_data, settle=0)
        robot = EpisodeTelemetry(empty_model).contacts.robot
        def candidate_check(obstacle):
            check = placement_check(empty_model, empty_data, obstacle, robot)
            if not check["valid"]:
                return check
            from .placement import footprint, cross
            polygon = footprint(obstacle)
            edges = np.roll(polygon, -1, axis=0) - polygon
            height = {"slope": 0.18, "barrier": 0.32, "stone": obstacle.size}.get(obstacle.kind, 2 * obstacle.size)
            for point in empty_data.flexvert_xpos:
                if point[2] > height + 0.05:
                    continue
                delta = point[:2] - polygon
                signs = cross(edges, delta)
                inside = np.all(signs >= 0) or np.all(signs <= 0)
                projection = np.clip(np.sum(delta * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
                gap = float(np.min(np.linalg.norm(delta - projection[:, None] * edges, axis=1)))
                if inside or gap < 0.05:
                    check.update(valid=False, reason="Obstacle footprint intersects the routed cable at reset; move it away from the cable")
                    break
            return check
    obstacles = make_course(
        config.level, config.seed, manual, config.mode, protect_start=False,
        approach=approach, candidate_check=candidate_check,
    )
    # Ground-truth checks belong to scenario validation, never the autonomous policy.
    from .drive import operator_path
    operator_path(pose[:2], math.radians(pose[2]), GOAL, 0.0, obstacles)
    params = dataclasses.replace(
        DRIVE_IN_PARAMS,
        base_start=pose,
        obstacles=obstacles,
        cable_length=CABLE_LENGTH,
        movable_vehicle=config.level >= 9,
        operator_cameras=True,
        environment="classic",
    )
    model, info = build_model(params)
    data = mujoco.MjData(model)
    reset_to_home(model, data, settle=0)
    startup = EpisodeTelemetry(model)
    from .placement import placement_check

    for i, obstacle in enumerate(obstacles):
        check = placement_check(model, data, obstacle, startup.contacts.robot)
        if not check["valid"]:
            raise ValueError(f"Obstacle {i + 1}: {check['reason']}")
    for c in data.contact:
        a, b = map(int, c.geom)
        if c.dist < 0 and startup.contacts.classify(a, b) in (
            "obstacle_cable",
            "obstacle_robot",
            "unsafe_robot_environment",
            "robot_self",
        ):
            raise ValueError(
                "Invalid initial geometry: robot or cable intersects scene"
            )
    for _ in range(250):
        mujoco.mj_step(model, data)
        startup.sample(data, DriveLimits())
    mujoco.mj_forward(model, data)
    loads = startup.sample(data, DriveLimits())
    if any(
        startup.contacts.episodes[c]
        for c in (
            "obstacle_robot",
            "obstacle_cable",
            "robot_self",
            "unsafe_robot_environment",
        )
    ):
        raise ValueError(
            "Robot or cable contacted an obstacle during settling; edit layout or choose another seed"
        )
    if loads.max_tension > 60 or loads.strain_relief_load > 45:
        raise ValueError(
            "Cable constraints infeasible after settling; choose another start/seed"
        )
    if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
        raise ValueError("Simulation failed startup validation")
    return model, data, info, obstacles, startup.report()
