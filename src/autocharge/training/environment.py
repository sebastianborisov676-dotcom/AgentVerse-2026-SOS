"""Small RL-ready API without a Gym dependency; baseline or bounded velocity actions.

Observations are privileged simulator state. Reward is an initial shaping heuristic.
"""

import dataclasses
import math

import mujoco
import numpy as np

from autocharge import plug_alignment_error, reset_to_home
from autocharge.base_controller import DriveLimits, DrivePhase
from autocharge.metrics import cable_metrics
from autocharge.mission import DRIVE_IN_PARAMS, ChargingMission
from autocharge.scene import build_model

from .course import make_course
from .navigation import NoPathError
from .telemetry import EpisodeTelemetry


class CourseEnv:
    def __init__(self, level=1, max_time=90.0, manual=(), mode="random", frame_skip=10):
        if max_time <= 0 or frame_skip < 1:
            raise ValueError("Positive horizon and frame_skip required")
        self.level, self.max_time, self.manual, self.mode, self.frame_skip = (
            level,
            max_time,
            manual,
            mode,
            frame_skip,
        )

    def reset(self, seed=0, alignment_offset_mm=0.0):
        self.obstacles = make_course(self.level, seed, self.manual, self.mode)
        self.model, self.info = build_model(
            dataclasses.replace(DRIVE_IN_PARAMS, obstacles=self.obstacles)
        )
        self.data = mujoco.MjData(self.model)
        # Keep the original 0.5 s settling interval; record it separately from operation.
        reset_to_home(self.model, self.data, settle=0)
        self.settling = EpisodeTelemetry(self.model)
        self.initial_obstacle_penetrations = []
        for contact in self.data.contact:
            a, b = map(int, contact.geom)
            category = self.settling.contacts.classify(a, b)
            if category.startswith("obstacle_") and contact.dist < 0:
                self.initial_obstacle_penetrations.append(
                    {"category": category, "depth_m": -float(contact.dist)}
                )
        self.settling.sample(self.data, DriveLimits())
        for _ in range(round(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.settling.sample(self.data, DriveLimits())
        mujoco.mj_forward(self.model, self.data)
        self.telemetry = EpisodeTelemetry(self.model)
        self.telemetry.sample(self.data, DriveLimits())
        sid = self.model.site("socket_face").id
        rot = self.data.site_xmat[sid].reshape(3, 3).copy()
        pos = self.data.site_xpos[sid].copy() + rot[:, 0] * alignment_offset_mm * 0.001
        self.blocked = None
        try:
            self.mission = ChargingMission(
                self.model,
                self.data,
                obstacles=self.obstacles,
                socket_estimate=(pos, rot),
            )
        except NoPathError as exc:
            self.blocked = str(exc)
            self.mission = ChargingMission(
                self.model, self.data, socket_estimate=(pos, rot)
            )
            self.mission.drive._set(self.data, DrivePhase.STOPPED, self.blocked)
        if self.initial_obstacle_penetrations:
            self.blocked = "initial obstacle penetration (invalid reset geometry)"
            self.mission.drive._set(self.data, DrivePhase.STOPPED, self.blocked)
        self.start_time = self.data.time
        self.previous_distance = self.mission.drive.errors(self.data)[0]
        self.terminated = self.mission.done
        self.truncated = False
        self.truncation_reason = "episode simulated-time limit"
        return self.observe(), {
            "seed": seed,
            "level": self.level,
            "blocked": self.blocked,
        }

    def observe(self):
        al = plug_alignment_error(self.model, self.data)
        cab = cable_metrics(self.model, self.data)
        return {
            "base_pose": np.array(self.mission.drive.pose(self.data)),
            "base_velocity": self.data.qvel[self.mission.drive.dadr].copy(),
            "alignment": np.array([al.axial, al.lateral, al.angle]),
            "cable_loads": np.array([cab.max_tension, cab.strain_relief_load]),
            "phase": self.mission.phase,
            "obstacles": np.array(
                [[o.x, o.y, o.size, o.angle] for o in self.obstacles]
            ).reshape(-1, 4),
        }

    def step(self, action=None):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended; reset before stepping")
        if action is not None:
            action = np.asarray(action, dtype=float)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Action must be finite [forward_speed, yaw_rate]")
        for _ in range(self.frame_skip):
            self.mission.step(self.data)
            drive = self.mission.drive
            if action is not None and drive.phase.value == "drive":
                # Existing cable limits still run; static-map guard also checks policy actions.
                from .navigation import segment_clear

                v, w = np.clip(action, [-0.15, -0.6], [0.35, 0.6])
                x, y, yaw = drive.pose(self.data)
                ahead = (
                    np.array([x, y])
                    + np.array([math.cos(yaw), math.sin(yaw)]) * v * 0.8
                )
                if segment_clear((x, y), ahead, self.obstacles, 0.55):
                    self.data.ctrl[drive.act] = [
                        v * math.cos(yaw),
                        v * math.sin(yaw),
                        w,
                    ]
                else:
                    drive._set(
                        self.data, DrivePhase.STOPPED, "policy obstacle clearance"
                    )
                    self.data.ctrl[drive.act] = 0
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Nonfinite simulation state")
            limits = (
                self.mission.insertion.limits
                if self.mission.insertion is not None
                else drive.lim
            )
            self.telemetry.sample(self.data, limits)
            contacts = self.telemetry.contacts.active
            if any(
                key[0] in ("obstacle_robot", "robot_self", "unsafe_robot_environment")
                for key in contacts
            ):
                drive._set(self.data, DrivePhase.STOPPED, "unsafe robot contact")
                if self.mission.insertion is not None:
                    self.mission.insertion.emergency_stop(self.data)
            self.terminated = self.mission.done
            self.truncated = (
                self.data.time - self.start_time >= self.max_time
                and not self.terminated
            )
            if self.terminated or self.truncated:
                break
        distance = drive.errors(self.data)[0]
        reward = (
            self.previous_distance
            - distance
            + (10 if self.mission.phase == "seated" else 0)
            - (5 if self.mission.phase in ("stopped", "aborted") else 0)
        )
        self.previous_distance = distance
        return (
            self.observe(),
            reward,
            self.terminated,
            self.truncated,
            {"events": self.mission.events, "success": self.mission.phase == "seated"},
        )

    def report(self):
        report = self.telemetry.report()
        report["settling"] = self.settling.report()
        report["initial_obstacle_penetrations"] = self.initial_obstacle_penetrations
        drive = self.mission.drive
        report["docking"] = {
            "reached": any(e.startswith("docked:") for _, e in self.mission.events),
            "distance_m": drive.errors(self.data)[0],
            "yaw_error_rad": drive.errors(self.data)[2],
        }
        report["planning"] = {
            "blocked": self.blocked,
            "path_length_m": drive.path_len,
            "max_y_m": float(np.max(drive.path[:, 1])),
        }
        failure = None
        category = None
        if self.blocked:
            failure = self.blocked
            category = (
                "initial_geometry" if self.initial_obstacle_penetrations else "planning"
            )
        elif self.truncated:
            failure, category = (
                getattr(self, "truncation_reason", "episode simulated-time limit"),
                "timeout",
            )
        elif self.mission.phase != "seated":
            failure = (
                drive.events[-1][1]
                if drive.phase == DrivePhase.STOPPED
                else self.mission.events[-1][1]
            )
            category = (
                "mechanical_load"
                if any(word in failure for word in ("tension", "strain relief"))
                else "collision"
                if "contact" in failure
                else "tracking_or_alignment"
            )
        if any(w.number for w in self.data.warning):
            failure, category = "MuJoCo warning", "simulation"
        report["failure_reason"], report["failure_category"] = failure, category
        return report
