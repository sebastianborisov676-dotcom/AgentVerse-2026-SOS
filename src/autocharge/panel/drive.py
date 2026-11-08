"""Sensor-map approaches with a straight convergence segment before fine docking."""

import math
import numpy as np
from autocharge.base_controller import DriveController, DrivePhase, _wrap
from autocharge.training.navigation import (
    course_path, segment_clear, NoPathError, plan_path, approach_paths, path_clear,
)


def operator_path(start, heading, dock, yaw, points, side=None):
    start, dock = np.asarray(start), np.asarray(dock)
    distance = float(np.linalg.norm(dock - start))
    straight = min(0.9, 0.55 * distance)
    gate = dock - np.array([math.cos(yaw), math.sin(yaw)]) * straight
    if not segment_clear(gate, dock, points, 0.50):
        raise NoPathError("Observed final approach is blocked")
    route = course_path(start, heading, gate, yaw, [], clearance=0.50)
    if not path_clear(route, points, clearance=0.50):
        route = None
        base = approach_paths(start, heading, gate, yaw)[0]
        t = np.linspace(0.0, 1.0, len(base))
        normal = np.array([-math.sin(yaw), math.cos(yaw)])
        # The bypass must persist until the final gate, rather than returning to
        # the original lane while the observed obstacle is still alongside us.
        preferred = (-1 if start[1] > 0.7 else 1) if side is None else side
        for side in (preferred, -preferred):
            for amplitude in (0.35, 0.6, 0.9, 1.2):
                candidate = base + side * amplitude * np.sin(np.pi * t)[:, None]**2 * normal
                if path_clear(candidate, points, clearance=0.50):
                    route = candidate
                    break
            if route is None:
                for offset in (0.6, 0.9, 1.2):
                    staging = start + [0, side * offset]
                    if not segment_clear(start, staging, points, clearance=0.50):
                        continue
                    try:
                        onward = plan_path(tuple(staging), tuple(gate), points, clearance=0.50)
                        route = np.vstack([np.linspace(start, staging, 61), onward])
                        break
                    except NoPathError:
                        pass
            if route is not None:
                break
        if route is None:
            route = plan_path(tuple(start), tuple(gate), points, clearance=0.50)
    return np.vstack(
        [route, np.linspace(gate, dock, max(2, math.ceil(straight / 0.01)))[1:]]
    )


class OperatorDrive(DriveController):
    def __init__(self, model, data, **kwargs):
        super().__init__(model, data, **kwargs)
        x, y, yaw = self.pose(data)
        self.bypass_side = -1 if y > 0.7 else 1
        self.path = operator_path((x, y), yaw, self.dock, self.dock_yaw, [])
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.path_len = float(self.s[-1])

    def step(self, data):
        super().step(data)
        if self.phase == DrivePhase.DRIVE:
            x, y, yaw = self.pose(data)
            j = min(int(np.searchsorted(self.s, self.s[self.idx] + self.lim.lookahead)), len(self.path) - 1)
            target = self.path[j]
            alpha = _wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
            if abs(alpha) > 0.35:
                # Navigation wrapper: brake and turn before tracking a sharp bypass.
                # Keep the existing driver's contact/cable checks and angular limits.
                self.v = 0.0
                desired = float(np.clip(2 * alpha, -self.lim.max_yaw_rate, self.lim.max_yaw_rate))
                previous = getattr(self, "turn_rate", 0.0)
                change = self.lim.max_yaw_accel * self.m.opt.timestep
                self.w = self.turn_rate = previous + float(np.clip(desired - previous, -change, change))
                data.ctrl[self.act] = [0, 0, self.w]
            else:
                self.turn_rate = self.w
        if self.phase != DrivePhase.DOCK:
            self.trim = np.zeros(3)
            return
        # Explicit prototype: the existing planar servos permit low-speed XY
        # docking trim. This is not wheel/traction or differential-drive physics.
        x, y, yaw = self.pose(data)
        desired = np.clip(
            10
            * np.array(
                [self.dock[0] - x, self.dock[1] - y, _wrap(self.dock_yaw - yaw)]
            ),
            [-0.015, -0.015, -0.15],
            [0.015, 0.015, 0.15],
        )
        previous = getattr(self, "trim", np.zeros(3))
        change = np.array([0.1, 0.1, 0.3]) * self.m.opt.timestep
        self.trim = previous + np.clip(desired - previous, -change, change)
        data.ctrl[self.act] = self.trim
