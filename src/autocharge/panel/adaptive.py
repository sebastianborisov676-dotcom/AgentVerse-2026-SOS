"""Sensor-only obstacle adaptation and target-motion safety around SAI's mission."""

import dataclasses
import math
import random

import mujoco
import numpy as np

from autocharge.base_controller import DriveLimits, DrivePhase
from autocharge.controller import Phase
from autocharge.metrics import (
    cable_metrics,
    connector_contact_force,
    plug_alignment_error,
)
from autocharge.mission import ChargingMission
from autocharge.scene import orientation_error
from autocharge.training.navigation import NoPathError, path_clear
from .sensors import Sensors
from .drive import OperatorDrive, operator_path


class VehicleMotion:
    def __init__(self, model, seed, enabled, forward=False):
        self.enabled = enabled
        self.forward = forward
        self.available = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "vehicle_x") >= 0
        )
        self.started = False
        self.moving = False
        self.start_time = 0.0
        self.target = np.zeros(3)
        rng = random.Random(seed + 713)
        self.destination = np.array(
            [
                rng.uniform(0.1, 0.2),
                rng.uniform(-0.02, 0.02),
                -math.radians(rng.uniform(2, 5)),
            ]
        )
        if forward:
            # Existing wheel axle is local Y, so the vehicle rolls along local +X.
            self.destination = np.array([rng.uniform(0.10, 0.14), 0.0, 0.0])
        if self.available:
            self.qadr = [
                model.jnt_qposadr[model.joint(n).id]
                for n in ("vehicle_x", "vehicle_y", "vehicle_yaw")
            ]
            self.act = [
                model.actuator(n + "_pos").id
                for n in ("vehicle_x", "vehicle_y", "vehicle_yaw")
            ]

    def request(self, data):
        if not self.available or self.started:
            return False
        self.started = self.moving = True
        self.start_time = float(data.time)
        self.origin = data.qpos[self.qadr].copy()
        if self.forward:
            distance = float(self.destination[0])
            self.destination = self.origin + distance * np.array([math.cos(self.origin[2]), math.sin(self.origin[2]), 0.0])
        return True

    def freeze(self, data):
        if self.available:
            self.target = data.qpos[self.qadr].copy()
            self.destination = self.target.copy()
            self.moving = False
            data.ctrl[self.act] = self.target

    def step(self, data):
        if not self.available:
            return
        if self.moving:
            u = float(np.clip((data.time - self.start_time) / 4.0, 0, 1))
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.target = self.origin + (self.destination - self.origin) * blend
            if u >= 1:
                self.moving = False
        data.ctrl[self.act] = self.target


class AdaptiveMission:
    def __init__(self, model, data, level, seed, moving_car=True):
        self.model = model
        self.sensors = Sensors(model)
        estimate = self.sensors.port(data)
        if estimate is None:
            raise ValueError("Charging-port fiducial is not visible from this start")
        # No layout, object identities, obstacle poses or bounds are given to this controller.
        self.mission = ChargingMission(
            model,
            data,
            socket_estimate=estimate,
            drive_limits=dataclasses.replace(
                DriveLimits(),
                max_speed=0.14 if level == 11 else 0.22,
                max_yaw_rate=0.3 if level == 11 else DriveLimits().max_yaw_rate,
                max_yaw_accel=0.6 if level == 11 else DriveLimits().max_yaw_accel,
                lookahead=0.25,
                dock_pos_tol=0.005,
                dock_yaw_tol=math.radians(0.08),
                timeout=65.0,
            ),
        )
        self.mission.drive = OperatorDrive(model, data, limits=self.mission.drive.lim)
        self.initial_port = (estimate[0].copy(), estimate[1].copy())
        self.reference_port = (estimate[0].copy(), estimate[1].copy())
        self.vehicle = VehicleMotion(model, seed, level >= 9 and moving_car, forward=level in (9, 11))
        self.level = level
        self.avoidance_replans = 0
        self.obstacle_detected = False
        self.events = []
        self.replans = 0
        self.route_version = 0
        self.last_sense = -1.0
        self.waiting = False
        self.recovering = False
        self.failed = None
        self.stable_since = None
        self.scan_points = []
        self.last_target_seen = float(data.time)
        self.port_moving = False

    @property
    def phase(self):
        if self.failed:
            return "stopped"
        if self.recovering:
            return "safe_retraction"
        if self.waiting:
            return "target_moving"
        return self.mission.phase

    @property
    def done(self):
        if self.failed:
            return True
        return not (self.waiting or self.recovering) and self.mission.done

    def log(self, data, text):
        self.events.append((round(float(data.time), 3), text))

    def stop(self, data, reason):
        self.failed = reason
        self.mission.drive._set(data, DrivePhase.STOPPED, reason)
        self.mission.drive.step(data)
        if self.mission.insertion is not None:
            self.mission.insertion.emergency_stop(data)
        self.vehicle.freeze(data)
        self.log(data, reason)

    def hold_arm(self, data):
        m = self.mission
        err = m.q_carry - data.qpos[m.qadr]
        m.q_cmd = np.clip(
            m.q_cmd + m.hold_gain * err * self.model.opt.timestep, m.q_lo, m.q_hi
        )
        data.ctrl[m.arm_act] = m.q_cmd

    def request_vehicle_motion(self, data):
        ins = self.mission.insertion
        estimate = self.sensors.last_port
        plug = data.site_xpos[self.model.site("plug_face").id]
        axial = (
            float(np.dot(plug - estimate[0], estimate[1][:, 2])) if estimate else 0.0
        )
        if (
            ins is not None
            or connector_contact_force(self.model, data).count
            or axial > -0.08
        ):
            self.log(
                data, "vehicle movement inhibited: connector approach/contact interlock"
            )
            if ins is not None and ins.phase not in (Phase.ABORTED, Phase.ESTOP):
                ins._start_retract(
                    data,
                    "target-motion request: safe recovery, vehicle held stationary",
                )
                self.recovering = True
                self.vehicle.freeze(data)
            return False
        accepted = self.vehicle.request(data)
        if accepted:
            self.log(data, "vehicle motion started: scripted actor, insertion interlocked")
        return accepted

    def replan(self, data, dock=None, yaw=None):
        old = self.mission.drive
        dock = old.dock if dock is None else dock
        yaw = old.dock_yaw if yaw is None else yaw
        x, y, heading = old.pose(data)
        path = operator_path((x, y), heading, dock, yaw, self.scan_points, side=old.bypass_side)
        old.path = path
        old.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        old.path_len = float(old.s[-1])
        old.idx = 0
        old.dock = np.array(dock)
        old.dock_yaw = float(yaw)
        old.obstacles = self.scan_points
        old.hold_pose = None
        old._ok_since = None
        old._set(data, DrivePhase.DRIVE, "sensor-based route update")
        self.replans += 1
        self.route_version += 1
        self.log(
            data,
            f"route {self.route_version}: replanned using {len(self.scan_points)} observed cells",
        )

    def step(self, data):
        if self.failed:
            self.mission.drive.step(data)
            return
        drive = self.mission.drive
        ins = self.mission.insertion
        loads = cable_metrics(self.model, data)
        if (
            loads.max_tension > drive.lim.cable_tension_max
            or loads.strain_relief_load > drive.lim.strain_relief_max
        ):
            self.stop(data, "cable load limit: motion inhibited")
            return
        if (
            self.recovering
            and connector_contact_force(self.model, data).total_normal > 70.0
        ):
            self.stop(data, "hard contact limit during recovery: holding")
            return
        # Challenge script controls the actor; policy reacts only to measured target motion.
        if (
            self.vehicle.enabled
            and not self.vehicle.started
            and drive.errors(data)[0] < 0.45
            and ins is None
            and (self.level != 11 or self.avoidance_replans > 0)
        ):
            self.request_vehicle_motion(data)
        was_moving = self.vehicle.moving
        self.vehicle.step(data)
        if was_moving and not self.vehicle.moving:
            self.log(data, "vehicle command completed; awaiting measured stationary port")
        if data.time - self.last_sense >= (0.002 if ins is not None else 0.05) - 1e-9:
            self.last_sense = float(data.time)
            estimate = self.sensors.port(data)
            if estimate is not None:
                self.last_target_seen = float(data.time)
                delta = np.linalg.norm(estimate[0] - self.reference_port[0])
                angle = np.linalg.norm(
                    orientation_error(self.reference_port[1], estimate[1])
                )
                moving = (
                    self.sensors.port_speed > 0.001
                    or self.sensors.port_angular_speed > 0.001
                )
                self.port_moving = moving
                if (
                    (delta > 0.003 or angle > 0.003 or moving)
                    and not self.waiting
                    and not self.recovering
                ):
                    self.waiting = True
                    self.stable_since = None
                    drive._set(
                        data, DrivePhase.STOPPED, "observed port motion: braking"
                    )
                    self.log(
                        data,
                        "port motion detected by visible fiducial; braking; insertion inhibited",
                    )
                    if ins is not None:
                        self.vehicle.freeze(data)
                        ins._start_retract(
                            data, "observed target motion: stop advancing and retract"
                        )
                        self.recovering = True
                if not moving:
                    self.stable_since = (
                        self.stable_since
                        if self.stable_since is not None
                        else float(data.time)
                    )
                else:
                    self.stable_since = None
                if (
                    self.waiting
                    and not self.recovering
                    and self.stable_since is not None
                    and data.time - self.stable_since > 0.6
                ):
                    pos, rot = estimate
                    transform = rot @ self.initial_port[1].T
                    goal = pos[:2] + transform[:2, :2] @ (-self.initial_port[0][:2])
                    yaw = math.atan2(transform[1, 0], transform[0, 0])
                    self.scan_points = self.sensors.scan(data)
                    try:
                        self.replan(data, goal, yaw)
                    except NoPathError as exc:
                        self.stop(data, f"target replanning failed: {exc}")
                        return
                    self.reference_port = (pos.copy(), rot.copy())
                    self.mission._socket_estimate = (pos.copy(), rot.copy())
                    self.waiting = False
                    self.log(
                        data,
                        "stationary port confirmed; updated docking pose and alignment estimate",
                    )
            elif ins is not None:
                self.stop(
                    data,
                    "port observation lost during arm motion: holding, restart required",
                )
                return
            elif data.time - self.last_target_seen > 0.5 and (drive.errors(data)[0] < 0.8 or self.mission.phase != "drive"):
                self.stop(
                    data, "port observation lost: stop rather than insert blindly"
                )
                return
        if self.recovering:
            drive.step(data)
            ins = self.mission.insertion
            if ins is not None:
                ins.step(data)
                if ins.phase != Phase.RETRACT:
                    self.stop(data, "safe recovery complete; operator restart required")
            return
        if self.waiting:
            drive.step(data)
            self.hold_arm(data)
            return
        if data.time - self.sensors.last_scan >= 0.2 - 1e-9 and ins is None:
            self.scan_points = self.sensors.scan(data)
            drive.obstacles = self.scan_points
            if not path_clear(drive.path[drive.idx :], self.scan_points, 0.50):
                if not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.log(data, "obstacle detected: observed cells intersect remaining approach")
                try:
                    self.replan(data)
                    self.avoidance_replans += 1
                    self.log(data, "avoidance replan: sensor-observed obstruction")
                except NoPathError as exc:
                    self.stop(data, f"no observed free route: {exc}")
                    return
        # The commanded actuator-busy interlock supplements measured motion.
        if self.vehicle.moving and drive.phase == DrivePhase.DOCKED:
            drive._set(
                data, DrivePhase.STOPPED, "vehicle actuator busy: insertion inhibited"
            )
            self.waiting = True
            return
        # Stale/moving measurements must never start or continue insertion.
        if ins is not None and self.port_moving:
            self.waiting = self.recovering = True
            self.vehicle.freeze(data)
            ins._start_retract(data, "motion interlock: retract")
            return
        self.mission.step(data)
