"""Operator-only ground picking and conservative projected geometry clearance.

Never supplied to autonomous navigation. Clearance comes from the actual ray sampling.
"""

import math
import numpy as np
from autocharge.training.levels import Obstacle
from autocharge.training.course import validate, BOUNDS
from .sensors import MAP_RESOLUTION, SCAN_RANGE, SCAN_FOV_DEG, SCAN_RAYS

# Half a map-cell diagonal plus half the widest inter-ray gap, rounded up to cm.
MIN_CLEARANCE = (
    math.ceil(
        (
            MAP_RESOLUTION / math.sqrt(2)
            + SCAN_RANGE * math.sin(math.radians(SCAN_FOV_DEG / (SCAN_RAYS - 1)) / 2)
        )
        * 100
    )
    / 100
)


def footprint(o):
    if o.kind in ("stone", "cone"):
        theta = np.linspace(0, 2 * math.pi, 17)[:-1]
        points = np.column_stack([np.cos(theta), np.sin(theta)]) * (
            o.size / math.cos(math.pi / 16)
        )
    else:
        y = 0.055 if o.kind == "barrier" else o.size
        points = np.array([[-o.size, -y], [o.size, -y], [o.size, y], [-o.size, y]])
    angle = math.radians(o.angle)
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, s], [-s, c]]) + np.array([o.x, o.y])


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_distance(a, b):
    ar = np.roll(a, -1, axis=0) - a
    br = np.roll(b, -1, axis=0) - b
    delta = b[None, :, :] - a[:, None, :]
    den = cross(ar[:, None, :], br[None, :, :])
    valid = np.abs(den) > 1e-12
    t = np.divide(
        cross(delta, br[None, :, :]), den, out=np.zeros_like(den), where=valid
    )
    u = np.divide(
        cross(delta, ar[:, None, :]), den, out=np.zeros_like(den), where=valid
    )
    if np.any(valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
        return 0.0
    for point, poly, edges in ((a[0], b, br), (b[0], a, ar)):
        signs = cross(edges, point - poly)
        if np.all(signs >= -1e-12) or np.all(signs <= 1e-12):
            return 0.0

    def distances(points, poly, edges):
        d = points[:, None, :] - poly[None, :, :]
        lens = np.sum(edges * edges, axis=1)
        f = np.clip(
            np.sum(d * edges[None, :, :], axis=2) / np.maximum(lens, 1e-20), 0, 1
        )
        return float(
            np.min(np.linalg.norm(d - f[:, :, None] * edges[None, :, :], axis=2))
        )

    return min(distances(a, b, br), distances(b, a, ar))


def clearance(model, data, obstacle, robot_bodies):
    polygon = footprint(obstacle)
    conn = model.body("connector").id
    result = {"robot": float("inf"), "connector": float("inf")}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b not in robot_bodies or not (
            model.geom_contype[g] or model.geom_conaffinity[g]
        ):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        half = np.abs(rot) @ model.geom_aabb[g, 3:]
        rectangle = np.array(
            [
                [center[0] - half[0], center[1] - half[1]],
                [center[0] + half[0], center[1] - half[1]],
                [center[0] + half[0], center[1] + half[1]],
                [center[0] - half[0], center[1] + half[1]],
            ]
        )
        category = "connector" if b == conn else "robot"
        result[category] = min(result[category], polygon_distance(polygon, rectangle))
    return result


def placement_check(model, data, obstacle, robot_bodies, others=()):
    reason = None
    try:
        validate(tuple(others) + (obstacle,), protect_start=False)
    except ValueError as exc:
        reason = str(exc)
    gap = clearance(model, data, obstacle, robot_bodies)
    if reason is None:
        for name, value in gap.items():
            if value < MIN_CLEARANCE:
                reason = f"{name.capitalize()} nearest projected edge clearance {value:.3f} m is below {MIN_CLEARANCE:.3f} m"
                break
    return dict(
        valid=reason is None,
        reason=reason,
        clearance_m=gap,
        required_clearance_m=MIN_CLEARANCE,
        footprint=footprint(obstacle).tolist(),
    )


def ground_point(view, u, v):
    if not 0 <= u <= 1 or not 0 <= v <= 1:
        raise ValueError("Click inside the rendered image")
    origin = np.array(view["pos"])
    forward = np.array(view["forward"])
    up = np.array(view["up"])
    right = np.cross(forward, up)
    x = view["center"] + (u - 0.5) * view["width"]
    y = view["top"] + v * (view["bottom"] - view["top"])
    ray = forward * view["near"] + right * x + up * y
    if ray[2] >= -1e-8:
        raise ValueError("This pixel points at the sky; click the ground")
    point = origin - ray * (origin[2] / ray[2])
    return point[:2]


def project(view, points):
    origin = np.array(view["pos"])
    f = np.array(view["forward"])
    up = np.array(view["up"])
    right = np.cross(f, up)
    relative = np.asarray(points) - origin
    depth = relative @ f
    horizontal = (relative @ right) * view["near"] / np.maximum(depth, 1e-9)
    vertical = (relative @ up) * view["near"] / np.maximum(depth, 1e-9)
    return np.column_stack(
        [
            0.5 + (horizontal - view["center"]) / view["width"],
            (view["top"] - vertical) / (view["top"] - view["bottom"]),
        ]
    ).tolist()
