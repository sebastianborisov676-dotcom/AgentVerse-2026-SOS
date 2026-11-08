"""Drive the mobile base from its start pose to the docking pose, then hold it there.

Motion model: unicycle (differential-drive) kinematics. The controller only ever commands a
forward speed along the robot's heading plus a yaw rate - no sideways motion - and turns them
into world-frame velocities for the base's planar velocity servos.

Phases::

    DRIVE (pure-pursuit along a smooth Bezier path) -> DOCK (slow final correction) -> DOCKED
    any -> STOPPED (cable limit exceeded or timeout)
"""

from __future__ import annotations

import dataclasses
import enum
import math

import mujoco
import numpy as np

from autocharge.metrics import cable_metrics
from autocharge.scene import BASE_ACTUATORS, BASE_JOINTS


class DrivePhase(enum.Enum):
    DRIVE = "drive"
    DOCK = "dock"
    DOCKED = "docked"
    STOPPED = "stopped"


@dataclasses.dataclass(frozen=True)
class DriveLimits:
    max_speed: float = 0.35  # m/s
    max_accel: float = 0.35  # m/s^2
    max_yaw_rate: float = 0.6  # rad/s
    max_yaw_accel: float = 1.2  # rad/s^2
    lookahead: float = 0.45  # m, pure-pursuit lookahead
    dock_switch_dist: float = 0.06  # m of path left when fine docking starts
    dock_pos_tol: float = 0.010  # m, required docking accuracy
    dock_yaw_tol: float = math.radians(0.5)
    dock_hold_time: float = 0.4  # s inside tolerance before declaring docked
    timeout: float = 45.0  # s
    cable_tension_max: float = 60.0  # N
    strain_relief_max: float = 45.0  # N
    brake_gain: float = 20.0  # 1/s, position hold after docking (stands in for wheel brakes)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _bezier(p0, p1, p2, p3, n=600) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


class DriveController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dock: tuple[float, float, float] = (0.0, 0.0, 0.0),  # x m, y m, yaw deg
        limits: DriveLimits | None = None,
        obstacles=None,
    ) -> None:
        self.m, self.lim = model, limits or DriveLimits()
        self.qadr = [model.jnt_qposadr[model.joint(n).id] for n in BASE_JOINTS]
        self.dadr = [model.jnt_dofadr[model.joint(n).id] for n in BASE_JOINTS]
        self.act = [model.actuator(n).id for n in BASE_ACTUATORS]
        self.dock = np.array([dock[0], dock[1]], dtype=float)
        self.dock_yaw = math.radians(dock[2])
        x, y, yaw = self.pose(data)
        self.start = (x, y, yaw)
        h0 = np.array([math.cos(yaw), math.sin(yaw)])
        h1 = np.array([math.cos(self.dock_yaw), math.sin(self.dock_yaw)])
        dist = float(np.linalg.norm(self.dock - [x, y]))
        k = 0.4 * dist
        self.path = _bezier(np.array([x, y]), np.array([x, y]) + h0 * k, self.dock - h1 * k, self.dock)
        self.obstacles = obstacles
        if obstacles is not None:
            from autocharge.training.navigation import course_path
            self.path = course_path((x, y), yaw, self.dock, self.dock_yaw, obstacles)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.path_len = float(self.s[-1])
        self.idx = 0
        self.v = self.w = 0.0
        self.phase = DrivePhase.DRIVE
        self.phase_start = data.time
        self._ok_since: float | None = None
        self.hold_pose: tuple[float, float, float] | None = None
        self.events: list[tuple[float, str]] = [(round(float(data.time), 3), f"drive: start, path {self.path_len:.2f} m")]
        self.travelled = 0.0
        self._last_xy = np.array([x, y])

    # ----------------------------------------------------------------------------------------
    def pose(self, data) -> tuple[float, float, float]:
        q = data.qpos[self.qadr]
        return float(q[0]), float(q[1]), float(q[2])

    def errors(self, data) -> tuple[float, float, float]:
        """(distance to dock m, lateral error m in dock frame, heading error rad)."""
        x, y, yaw = self.pose(data)
        d = self.dock - [x, y]
        c, s = math.cos(self.dock_yaw), math.sin(self.dock_yaw)
        lateral = -s * d[0] + c * d[1]
        return float(np.linalg.norm(d)), float(lateral), _wrap(self.dock_yaw - yaw)

    @property
    def done(self) -> bool:
        return self.phase in (DrivePhase.DOCKED, DrivePhase.STOPPED)

    def _set(self, data, phase: DrivePhase, why: str) -> None:
        self.phase, self.phase_start, self._ok_since = phase, data.time, None
        self.events.append((round(float(data.time), 3), f"{phase.value}: {why}"))

    # ----------------------------------------------------------------------------------------
    def step(self, data: mujoco.MjData) -> None:
        lim, dt = self.lim, self.m.opt.timestep
        x, y, yaw = self.pose(data)
        xy = np.array([x, y])
        self.travelled += float(np.linalg.norm(xy - self._last_xy))
        self._last_xy = xy

        if self.phase in (DrivePhase.DRIVE, DrivePhase.DOCK):
            if self.obstacles is not None:
                from autocharge.training.navigation import segment_clear
                # Include braking distance; stop rather than cutting a planned corner.
                ahead = xy + np.array([math.cos(yaw), math.sin(yaw)]) * (abs(self.v)*0.5 + self.v**2/(2*lim.max_accel))
                if not segment_clear(xy, ahead, self.obstacles, clearance=0.43):
                    self._set(data, DrivePhase.STOPPED, "obstacle safety clearance")
            cab = cable_metrics(self.m, data)
            if cab.max_tension > lim.cable_tension_max:
                self._set(data, DrivePhase.STOPPED, f"cable tension {cab.max_tension:.1f} N")
            elif cab.strain_relief_load > lim.strain_relief_max:
                self._set(data, DrivePhase.STOPPED, f"strain relief load {cab.strain_relief_load:.1f} N")
            elif data.time - self.events[0][0] > lim.timeout:
                self._set(data, DrivePhase.STOPPED, "timeout")

        v_des = w_des = 0.0
        if self.phase == DrivePhase.DRIVE:
            lo, hi = self.idx, min(self.idx + 80, len(self.path))
            self.idx = lo + int(np.argmin(np.linalg.norm(self.path[lo:hi] - xy, axis=1)))
            remaining = self.path_len - self.s[self.idx]
            if remaining < lim.dock_switch_dist:
                self._set(data, DrivePhase.DOCK, f"{remaining * 100:.1f} cm from dock, fine docking")
            else:
                target_s = self.s[self.idx] + lim.lookahead
                if target_s <= self.path_len:
                    j = int(np.searchsorted(self.s, target_s))
                    target = self.path[min(j, len(self.path) - 1)]
                else:  # extend the path straight past the dock along the docking heading
                    h1 = np.array([math.cos(self.dock_yaw), math.sin(self.dock_yaw)])
                    target = self.dock + h1 * (target_s - self.path_len)
                alpha = _wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
                v_des = min(lim.max_speed, math.sqrt(2 * lim.max_accel * 0.8 * remaining) + 0.03)
                w_des = 2 * v_des * math.sin(alpha) / lim.lookahead
        elif self.phase == DrivePhase.DOCK:
            c, s = math.cos(yaw), math.sin(yaw)
            d = self.dock - xy
            ex = c * d[0] + s * d[1]  # forward error in the robot frame
            eyaw = _wrap(self.dock_yaw - yaw)
            v_des = float(np.clip(1.5 * ex, -0.04, 0.04))
            w_des = float(np.clip(2.5 * eyaw, -0.15, 0.15))
            dist, _, _ = self.errors(data)
            ok = dist < lim.dock_pos_tol and abs(eyaw) < lim.dock_yaw_tol
            if ok:
                self._ok_since = self._ok_since if self._ok_since is not None else data.time
                if data.time - self._ok_since >= lim.dock_hold_time:
                    self.hold_pose = (x, y, yaw)
                    self._set(
                        data,
                        DrivePhase.DOCKED,
                        f"docked {dist * 1e3:.1f} mm / {math.degrees(abs(eyaw)):.2f} deg from target, brakes on",
                    )
            else:
                self._ok_since = None

        if self.phase in (DrivePhase.DOCKED, DrivePhase.STOPPED):
            if self.hold_pose is None:
                self.hold_pose = (x, y, yaw)
            # Brakes: stiff position hold through the velocity servos.
            hx, hy, hyaw = self.hold_pose
            k = lim.brake_gain
            data.ctrl[self.act] = np.clip([k * (hx - x), k * (hy - y), k * _wrap(hyaw - yaw)], -1, 1)
            self.v = self.w = 0.0
            return

        # Acceleration limits (smooth starts and stops).
        self.v += float(np.clip(v_des - self.v, -lim.max_accel * dt, lim.max_accel * dt))
        w_des = float(np.clip(w_des, -lim.max_yaw_rate, lim.max_yaw_rate))
        self.w += float(np.clip(w_des - self.w, -lim.max_yaw_accel * dt, lim.max_yaw_accel * dt))
        data.ctrl[self.act] = [self.v * math.cos(yaw), self.v * math.sin(yaw), self.w]
