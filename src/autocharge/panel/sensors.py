"""Finite-range ray sensing and an ideal visible fiducial pose sensor. No semantic map."""

import math

import mujoco
import numpy as np

from autocharge.scene import BASE_JOINTS
from autocharge.training.levels import Obstacle


SCAN_RANGE = 2.2
SCAN_FOV_DEG = 240
SCAN_RAYS = 81
MAP_RESOLUTION = 0.12
HIT_RADIUS = 0.07


class Sensors:
    def __init__(self, model):
        self.model = model
        self.qadr = [model.jnt_qposadr[model.joint(n).id] for n in BASE_JOINTS]
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_rgb"
        )
        self.marker_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "port_marker"
        )
        root = model.body("base_link").id
        self.self_bodies = set()
        for b in range(model.nbody):
            parent = b
            while parent and parent != root:
                parent = int(model.body_parentid[parent])
            if parent == root:
                self.self_bodies.add(b)
        self.cells = {}
        self.last_scan = -1.0
        self.last_port = None
        self.port_time = None
        self.port_speed = 0.0
        self.port_angular_speed = 0.0
        self.hit_count = 0

    def ray(self, origin, direction, max_range):
        travelled = 0.0
        point = np.asarray(origin, dtype=float).copy()
        geom = np.array([-1], dtype=np.int32)
        for _ in range(30):
            distance = mujoco.mj_ray(
                self.model, self._data, point, direction, None, 1, -1, geom
            )
            if distance < 0 or travelled + distance > max_range:
                return None
            g = int(geom[0])
            if int(self.model.geom_bodyid[g]) not in self.self_bodies:
                return travelled + float(distance), g
            advance = float(distance) + 0.002
            point += direction * advance
            travelled += advance
        return None

    def scan(self, data):
        self._data = data  # transient binding; checkpoint copying excludes this below
        x, y, yaw = data.qpos[self.qadr]
        self.cells = {k: t for k, t in self.cells.items() if data.time - t < 3.0}
        count = 0
        for z in (0.04, 0.2, 0.45):
            origin = np.array([x + 0.38 * math.cos(yaw), y + 0.38 * math.sin(yaw), z])
            for angle in np.linspace(
                yaw - math.radians(SCAN_FOV_DEG / 2),
                yaw + math.radians(SCAN_FOV_DEG / 2),
                SCAN_RAYS,
            ):
                direction = np.array([math.cos(angle), math.sin(angle), 0.0])
                hit = self.ray(origin, direction, SCAN_RANGE)
                if hit is not None:
                    point = origin + direction * hit[0]
                    key = tuple(map(int, np.rint(point[:2] / MAP_RESOLUTION)))
                    self.cells[key] = float(data.time)
                    count += 1
        self.hit_count = count
        self.last_scan = float(data.time)
        del self._data
        return self.points()

    def points(self):
        return [
            Obstacle("stone", k[0] * MAP_RESOLUTION, k[1] * MAP_RESOLUTION, HIT_RADIUS)
            for k in sorted(self.cells)
        ]

    def port(self, data):
        """Ideal marker pose is exposed only inside range/FOV and without occlusion."""
        self._data = data
        x, y, yaw = data.qpos[self.qadr]
        origin = np.array([x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw), 1.05])
        if self.marker_id >= 0:
            # Ideal marker measurement and calibrated marker-to-inlet transform.
            marker = data.site_xpos[self.marker_id].copy()
            rot = data.site_xmat[self.marker_id].reshape(3, 3).copy()
            target = marker - rot @ np.array([0.15, 0, 0.055])
        else:
            sid = self.model.site("socket_face").id
            target = data.site_xpos[sid].copy()
            rot = data.site_xmat[sid].reshape(3, 3).copy()
            marker = target + rot[:, 0] * 0.15 - rot[:, 2] * 0.03
        if self.camera_id >= 0:
            origin = data.cam_xpos[self.camera_id].copy()
        delta = marker - origin
        distance = float(np.linalg.norm(delta))
        visible = (
            distance < 4.5 and np.dot(delta[:2], [math.cos(yaw), math.sin(yaw)]) > 0
        )
        if self.camera_id >= 0:
            local = data.cam_xmat[self.camera_id].reshape(3, 3).T @ delta
            forward = -local[2]
            vertical = math.tan(
                math.radians(float(self.model.cam_fovy[self.camera_id])) / 2
            )
            visible = (
                visible
                and forward > 0
                and abs(local[1]) < forward * vertical
                and abs(local[0]) < forward * vertical * (5 / 3)
            )
        if visible and distance > 1e-9:
            hit = self.ray(origin, delta / distance, distance)
            visible = hit is None or hit[0] >= distance - 0.01
        del self._data
        if not visible:
            return None
        if self.last_port is not None and data.time > self.port_time:
            dt = data.time - self.port_time
            self.port_speed = float(np.linalg.norm(target - self.last_port[0]) / dt)
            from autocharge.scene import orientation_error

            self.port_angular_speed = float(
                np.linalg.norm(orientation_error(self.last_port[1], rot)) / dt
            )
        self.last_port = (target, rot)
        self.port_time = float(data.time)
        return self.last_port
