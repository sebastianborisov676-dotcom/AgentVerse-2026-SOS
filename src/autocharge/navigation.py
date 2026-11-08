"""Obstacle-aware base navigation: simulated 2D LiDAR -> occupancy grid -> A* -> waypoint driving.

* ``Lidar``: 180 beams over 360 deg, 0.40 m above the floor, 4 m range, cast from the robot base
  against the scene geometry (the simulated equivalent of a planar laser scanner). It returns only
  ranges - no object identities or positions are given to the robot.
* ``OccupancyGrid``: world-frame grid (5 cm cells) filled from scans using the robot's own
  odometry, inflated by the base radius plus a safety margin.
* ``plan_path``: 8-connected A* from the robot to the goal, simplified to line-of-sight waypoints.
* ``WaypointDrive``: chains ``TurnDriveTurn`` legs through the waypoints and ends at the goal heading.
"""

from __future__ import annotations

import heapq
import math

import mujoco
import numpy as np

from autocharge.base_controller import DriveLimits, DrivePhase, TurnDriveTurn
from autocharge.scene import BASE_JOINTS


class Lidar:
    def __init__(self, model: mujoco.MjModel, beams: int = 180, height: float = 0.40, max_range: float = 4.0):
        self.m = model
        self.base = model.body("base_link").id
        self.height, self.max_range = height, max_range
        self.angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
        self.groups = np.array([1, 1, 0, 0, 0, 0], np.uint8)  # scene geoms; the robot's own are groups 2/3
        self._gid = np.zeros(1, np.int32)

    def scan(self, data) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (hit points in world xy [n x 2], ranges [beams], base yaw). Misses are not returned."""
        q = [data.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] for n in BASE_JOINTS]
        x, y, yaw = float(q[0]), float(q[1]), float(q[2])
        origin = np.array([x, y, self.height])
        ranges = np.full(len(self.angles), np.inf)
        pts = []
        for i, a in enumerate(self.angles):
            vec = np.array([math.cos(yaw + a), math.sin(yaw + a), 0.0])
            r = mujoco.mj_ray(self.m, data, origin, vec, self.groups, 1, self.base, self._gid)
            if 0 <= r <= self.max_range:
                ranges[i] = r
                pts.append(origin[:2] + vec[:2] * r)
        return (np.array(pts) if pts else np.zeros((0, 2))), ranges, yaw


class OccupancyGrid:
    def __init__(self, center: tuple[float, float], size: float = 9.0, res: float = 0.05, inflate: float = 0.42):
        self.res = res
        self.n = int(size / res)
        self.origin = np.array(center) - size / 2
        self.occ = np.zeros((self.n, self.n), bool)
        self.inflate = inflate
        self._r = int(math.ceil(inflate / res))
        yy, xx = np.mgrid[-self._r:self._r + 1, -self._r:self._r + 1]
        self._disk = (xx ** 2 + yy ** 2) <= self._r ** 2
        self.blocked = np.zeros_like(self.occ)

    def cell(self, p) -> tuple[int, int]:
        c = np.floor((np.asarray(p) - self.origin) / self.res).astype(int)
        return int(np.clip(c[0], 0, self.n - 1)), int(np.clip(c[1], 0, self.n - 1))

    def world(self, c) -> np.ndarray:
        return self.origin + (np.asarray(c) + 0.5) * self.res

    def add_points(self, pts: np.ndarray) -> None:
        for p in pts:
            i, j = self.cell(p)
            self.occ[i, j] = True
        self._inflate()

    def _inflate(self) -> None:
        b = np.zeros_like(self.occ)
        r = self._r
        for i, j in zip(*np.nonzero(self.occ), strict=True):
            i0, i1, j0, j1 = max(0, i - r), min(self.n, i + r + 1), max(0, j - r), min(self.n, j + r + 1)
            b[i0:i1, j0:j1] |= self._disk[i0 - i + r:i1 - i + r, j0 - j + r:j1 - j + r]
        self.blocked = b

    def mask(self, radius: float) -> np.ndarray:
        """Occupied cells inflated by ``radius`` (e.g. the bare footprint, for route checks)."""
        r = int(math.ceil(radius / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        b = np.zeros_like(self.occ)
        for i, j in zip(*np.nonzero(self.occ), strict=True):
            i0, i1, j0, j1 = max(0, i - r), min(self.n, i + r + 1), max(0, j - r), min(self.n, j + r + 1)
            b[i0:i1, j0:j1] |= disk[i0 - i + r:i1 - i + r, j0 - j + r:j1 - j + r]
        return b

    def free_line(self, a, b) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(2, int(np.linalg.norm(b - a) / (self.res * 0.5)))
        for t in np.linspace(0, 1, steps):
            i, j = self.cell(a + (b - a) * t)
            if self.blocked[i, j]:
                return False
        return True


def plan_path(grid: OccupancyGrid, start, goal, start_free_radius: float = 0.3) -> list[np.ndarray] | None:
    """A* over the inflated grid; cells near the robot's own position are treated as free (the
    robot is standing there). Returns simplified world waypoints ending at ``goal``, or None."""
    blocked = grid.blocked.copy()
    si, sj = grid.cell(start)
    rr = int(start_free_radius / grid.res)
    blocked[max(0, si - rr):si + rr + 1, max(0, sj - rr):sj + rr + 1] = False
    gi, gj = grid.cell(goal)
    if blocked[gi, gj]:
        return None
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq = [(0.0, (si, sj))]
    cost = {(si, sj): 0.0}
    parent: dict = {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == (gi, gj):
            break
        for di, dj, w in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < grid.n and 0 <= nb[1] < grid.n) or blocked[nb]:
                continue
            nc = cost[cur] + w
            if nc < cost.get(nb, 1e18):
                cost[nb] = nc
                parent[nb] = cur
                heapq.heappush(openq, (nc + math.hypot(gi - nb[0], gj - nb[1]), nb))
    if (gi, gj) not in cost:
        return None
    cells = [(gi, gj)]
    while cells[-1] != (si, sj):
        cells.append(parent[cells[-1]])
    pts = [grid.world(c) for c in reversed(cells)]
    pts[0], pts[-1] = np.asarray(start, float), np.asarray(goal, float)
    # Line-of-sight simplification (checked against the inflated grid, start cells freed).
    saved = grid.blocked
    grid.blocked = blocked
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        j = len(pts) - 1
        while j > k + 1 and not grid.free_line(pts[k], pts[j]):
            j -= 1
        out.append(pts[j])
        k = j
    grid.blocked = saved
    return out


def path_length(path) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:], strict=True)))


class WaypointDrive:
    """Drive through waypoints with turn-drive-turn legs; same interface as the other drivers."""

    def __init__(self, model, data, waypoints: list[np.ndarray], final_yaw_deg: float, limits: DriveLimits | None = None):
        self.m, self.lim = model, limits or DriveLimits(timeout=90.0)
        self.waypoints = [np.asarray(w, float) for w in waypoints[1:]]  # first entry is the start
        self.final_yaw = final_yaw_deg
        self.events: list[tuple[float, str]] = [(round(float(data.time), 3), f"drive: {len(self.waypoints)} leg(s)")]
        self.leg = None
        self._start_leg(data)

    def _start_leg(self, data) -> None:
        wp = self.waypoints[0]
        if len(self.waypoints) > 1:
            nxt = self.waypoints[1]
            yaw = math.degrees(math.atan2(nxt[1] - wp[1], nxt[0] - wp[0]))
        else:
            yaw = self.final_yaw
        self.leg = TurnDriveTurn(self.m, data, dock=(float(wp[0]), float(wp[1]), yaw), limits=self.lim)

    @property
    def dock(self) -> np.ndarray:
        return self.waypoints[-1]

    @property
    def phase(self):
        if self.leg.phase == DrivePhase.DOCKED and len(self.waypoints) > 1:
            return DrivePhase.DRIVE
        return self.leg.phase

    @property
    def hold_pose(self):
        return self.leg.hold_pose

    def retarget(self, data, dock) -> None:
        self.waypoints[-1] = np.array(dock[:2], float)
        self.final_yaw = dock[2]
        if len(self.waypoints) == 1:
            self.leg.retarget(data, dock)

    def step(self, data) -> None:
        self.leg.step(data)
        if self.leg.phase == DrivePhase.DOCKED and len(self.waypoints) > 1:
            self.waypoints.pop(0)
            self._start_leg(data)
        elif self.leg.phase == DrivePhase.STOPPED:
            self.events.append(self.leg.events[-1])
