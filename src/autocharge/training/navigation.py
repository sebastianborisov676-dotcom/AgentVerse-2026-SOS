"""Deterministic grid A* with swept-disc segment checks (privileged map baseline)."""

import heapq
import itertools
import math

import numpy as np

from .course import BOUNDS, radius


class NoPathError(ValueError):
    pass


def segment_clear(a, b, obstacles, clearance=0.55):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    dd = float(d @ d)
    for o in obstacles:
        p = np.array([o.x, o.y])
        t = float(np.clip((p - a) @ d / dd, 0, 1)) if dd else 0
        if np.linalg.norm(p - (a + t * d)) <= radius(o) + clearance:
            return False
    # Existing charger pedestal is also a fixed hazard.
    p = np.array([-0.35, -0.70])
    t = float(np.clip((p - a) @ d / dd, 0, 1)) if dd else 0
    return np.linalg.norm(p - (a + t * d)) > 0.13 + min(clearance, 0.55)


def plan_path(start, goal, obstacles, resolution=0.1, clearance=0.55):
    if resolution <= 0 or clearance < 0:
        raise ValueError("Invalid planner resolution/clearance")
    start, goal = tuple(start), tuple(goal)
    xmin, xmax, ymin, ymax = BOUNDS

    def point(k):
        return (round(xmin + k[0] * resolution, 8), round(ymin + k[1] * resolution, 8))

    def key(p):
        return (round((p[0] - xmin) / resolution), round((p[1] - ymin) / resolution))

    source, target = key(start), key(goal)

    def clear(a, b):
        return segment_clear(a, b, obstacles, clearance)

    if not clear(start, point(source)) or not clear(point(target), goal):
        raise NoPathError("Start or goal blocked")
    queue = [(math.dist(start, goal), 0, source)]
    costs = {source: 0}
    parents = {}
    while queue:
        _, cost, k = heapq.heappop(queue)
        if cost > costs[k]:
            continue
        if k == target:
            route = [goal, point(k)]
            while k in parents:
                k = parents[k]
                route.append(point(k))
            route.append(start)
            route.reverse()
            # Greedy shortcutting uses the same swept collision checks.
            smooth = [route[0]]
            i = 0
            while i < len(route) - 1:
                j = len(route) - 1
                while j > i + 1 and not clear(route[i], route[j]):
                    j -= 1
                smooth.append(route[j])
                i = j
            dense = [smooth[0]]
            for a, b in itertools.pairwise(smooth):
                n = max(1, math.ceil(math.dist(a, b) / 0.02))
                dense.extend(np.linspace(a, b, n + 1)[1:])
            return np.asarray(dense)
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            nxt = (k[0] + dx, k[1] + dy)
            p = point(nxt)
            if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax) or not clear(
                point(k), p
            ):
                continue
            new = cost + math.hypot(dx, dy) * resolution
            if new < costs.get(nxt, float("inf")):
                costs[nxt] = new
                parents[nxt] = k
                heapq.heappush(queue, (new + math.dist(p, goal), new, nxt))
    raise NoPathError("No collision-free route in course bounds")


def approach_paths(start, yaw=math.pi / 12, goal=(0.0, 0.0), goal_yaw=0.0):
    """Smooth bounded bypass family; bump has zero tangent at both endpoints."""
    start, goal = np.asarray(start), np.asarray(goal)
    h0 = np.array([math.cos(yaw), math.sin(yaw)])
    h1 = np.array([math.cos(goal_yaw), math.sin(goal_yaw)])
    k = 0.4 * np.linalg.norm(goal - start)
    t = np.linspace(0, 1, 600)[:, None]
    base = (
        (1 - t) ** 3 * start
        + 3 * (1 - t) ** 2 * t * (start + k * h0)
        + 3 * (1 - t) * t**2 * (goal - k * h1)
        + t**3 * goal
    )
    normal = np.array([-h1[1], h1[0]])
    return [
        base + amplitude * np.sin(math.pi * np.clip(t / 0.55, 0, 1)) ** 4 * normal
        for amplitude in (0.0, 0.35, 0.7)
    ]


def path_clear(path, obstacles, clearance=0.60):
    """Vectorized swept-segment clearance, including fixed charger footprint."""
    path = np.asarray(path)
    a, d = path[:-1], np.diff(path, axis=0)
    dd = np.sum(d * d, axis=1)
    for x, y, r, margin in [(o.x, o.y, radius(o), clearance) for o in obstacles] + [
        (-0.35, -0.7, 0.13, min(clearance, 0.55))
    ]:
        p = np.array([x, y])
        t = np.clip(
            np.divide(
                np.sum((p - a) * d, axis=1), dd, out=np.zeros_like(dd), where=dd > 0
            ),
            0,
            1,
        )
        if np.any(np.linalg.norm(p - a - t[:, None] * d, axis=1) <= r + margin):
            return False
    return True


def course_path(start, yaw, dock, dock_yaw, obstacles, clearance=0.60):
    for path in approach_paths(start, yaw, dock, dock_yaw):
        if path_clear(path, obstacles, clearance):
            return path
    h1 = np.array([math.cos(dock_yaw), math.sin(dock_yaw)])
    approach = np.asarray(dock) - h1 * 0.8
    if not segment_clear(approach, dock, obstacles, clearance=clearance):
        raise NoPathError("Final docking corridor blocked")
    route = plan_path(start, tuple(approach), obstacles, clearance=clearance)
    return np.vstack([route, np.linspace(approach, dock, 81)[1:]])
