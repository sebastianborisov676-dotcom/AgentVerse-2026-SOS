"""Validated course layouts and contact geometry; angles are degrees, sizes metres."""

import json
import math
import random
from pathlib import Path

import mujoco
import numpy as np

from .levels import LEVELS, SIZES, Obstacle

BOUNDS = (-3.8, 0.0, -4.5, 4.5)
START = (-3.0, 0.5)
GOAL = (0.0, 0.0)
# Conservative XY hull of the nominal cable's measured swept area (including
# the initial routed cable), padded by geometry checks below. Metres.
# Derived from outputs/course_reference_envelopes.json, not a rope planner.
CABLE_SWEEP = np.array(
    [
        [-2.8, 0.5],
        [-2.0, 0.4],
        [-1.4, 0.2],
        [-0.7, -0.1],
        [0.7, -0.1],
        [0.7, -0.5],
        [-0.4, -1.65],
        [-1.1, -1.65],
        [-1.5, -1.4],
        [-2.0, -1.15],
        [-2.4, -0.7],
        [-2.8, 0.0],
    ]
)


def cable_lane_clear(o, margin=0.15):
    """Disc-vs-polygon check: keep placement away from the nominal rope workspace."""
    point = np.array([o.x, o.y])
    a, b = CABLE_SWEEP, np.roll(CABLE_SWEEP, -1, axis=0)
    delta = b - a
    t = np.clip(
        np.sum((point - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1
    )
    if (
        np.min(np.linalg.norm(point - a - t[:, None] * delta, axis=1))
        <= radius(o) + margin
    ):
        return False
    # Ray casting also excludes obstacle centres inside the polygon.
    crossing = ((a[:, 1] > o.y) != (b[:, 1] > o.y)) & (
        o.x
        < (b[:, 0] - a[:, 0]) * (o.y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-30) + a[:, 0]
    )
    return not bool(np.count_nonzero(crossing) % 2)


def radius(o):
    # Circumscribed footprint, shared by generator, planner and runtime guard.
    return o.size * (math.sqrt(2) if o.kind in ("box", "slope") else 1.0)


def validate(obstacles, protect_start=True):
    for index, o in enumerate(obstacles):
        label = f"Obstacle {index + 1} ({o.kind} at {o.x:.2f}, {o.y:.2f}): "
        if o.kind not in SIZES or not all(
            math.isfinite(v) for v in (o.x, o.y, o.size, o.angle)
        ):
            raise ValueError(label + "Invalid obstacle kind or nonfinite geometry")
        if not 0 < o.size <= 0.6:
            raise ValueError(label + "Obstacle size must be in (0, 0.6] metres")
        if not (BOUNDS[0] <= o.x <= BOUNDS[1] and BOUNDS[2] <= o.y <= BOUNDS[3]):
            raise ValueError(label + "Obstacle outside course bounds")
        if any(
            math.dist((o.x, o.y), q) < radius(o) + 0.65
            for q in ((START, GOAL) if protect_start else (GOAL,))
        ):
            raise ValueError(label + "Obstacle intersects protected start/dock area")
    for i, o in enumerate(obstacles):
        for j, b in enumerate(obstacles[:i]):
            distance = math.dist((o.x, o.y), (b.x, b.y))
            required = radius(o) + radius(b) + 0.05
            if distance < required:
                raise ValueError(
                    f"Obstacle {i+1} ({o.kind}) overlaps obstacle {j+1} ({b.kind}): "
                    f"centre distance {distance:.3f} m below conservative footprint "
                    f"separation {required:.3f} m. Move or remove either entry"
                )
    return tuple(obstacles)


def make_course(level, seed=0, manual=(), mode="random", protect_start=True, approach=None, candidate_check=None):
    if level not in LEVELS or mode not in ("random", "manual", "hybrid"):
        raise ValueError("Expected level 1..11 and random/manual/hybrid mode")
    if level < 5 and (manual or mode != "random"):
        raise ValueError("Manual placement starts at Level 5")
    if manual and mode == "random":
        raise ValueError("Use manual or hybrid mode for manual obstacles")
    from .navigation import course_path, path_clear, segment_clear

    obstacles = list(validate(manual, protect_start=protect_start))
    if not segment_clear((-0.8, 0), GOAL, obstacles, clearance=0.65):
        blocked = [
            f"obstacle {i + 1}"
            for i, o in enumerate(obstacles)
            if not segment_clear((-0.8, 0), GOAL, [o], clearance=0.65)
        ]
        raise ValueError(
            "Manual obstacles block the final docking corridor: "
            + ", ".join(blocked)
            + ". Move or remove these entries"
        )
    witness = course_path(START, math.radians(15), GOAL, 0.0, obstacles)
    # Preserve the complete path, rather than accepting a disconnected empty dock.
    if not path_clear(witness, obstacles):
        raise ValueError("Manual obstacles leave no feasible route")
    rng = random.Random(seed)
    count = LEVELS[level]["count"] if mode != "manual" else 0
    kinds = [
        LEVELS[level]["types"][i % len(LEVELS[level]["types"])] for i in range(count)
    ]
    rng.shuffle(kinds)
    if approach is not None and count:
        # Scenario-only ground truth: intersect the EMPTY-MAP operator route,
        # then prove a detour exists. Never pass this route/layout to the policy.
        from autocharge.panel.drive import operator_path
        start, heading = approach
        initial = operator_path(start, heading, GOAL, 0.0, [])
        preferred = next((k for k in ("box", "barrier", "cone", "stone", "slope") if k in kinds), kinds[0])
        if (level in (6, 9) or (level != 11 and math.dist(start, GOAL) < 2.6)) and "stone" in kinds:
            preferred = "stone"
        found = False
        last_reason = "no candidate preserves the start and final docking corridor"
        for _ in range(160):
            fraction = rng.uniform(0.48, 0.52) if level == 11 else (rng.uniform(0.48, 0.52) if math.dist(start, GOAL) < 2.6 else rng.uniform(0.50, 0.55))
            if preferred == "stone" and math.dist(start, GOAL) >= 2.6:
                fraction = rng.uniform(0.54, 0.62)
            point = initial[int(fraction * (len(initial) - 1))]
            candidate = Obstacle(preferred, float(point[0]), float(point[1]), min(SIZES[preferred], 0.20 if preferred == "barrier" else 0.12), 90 if preferred == "barrier" else 0)
            if level == 11:
                candidate.size = 0.05
                candidate.x += 0.025
                if candidate.x - start[0] < 0.70:
                    continue
            # Cross the route near the lower footprint edge, leaving a modest
            # detour on the charger side instead of demanding a full-width wrap.
            if preferred in ("box", "barrier"):
                candidate.y += 0.85 * candidate.size
            try:
                validate(obstacles + [candidate], protect_start=protect_start)
                if math.dist(start, point) < radius(candidate) + 0.55:
                    continue
                if not segment_clear((-0.8, 0), GOAL, [candidate], clearance=0.65):
                    continue
                if candidate_check is not None:
                    check = candidate_check(candidate)
                    if not check["valid"]:
                        last_reason = check["reason"]
                        continue
                detour = operator_path(start, heading, GOAL, 0.0, obstacles + [candidate])
                # The broad nominal cable envelope is deliberately not used here:
                # actual initial contacts, settling and runtime load limits still apply.
                witness = detour
                obstacles.append(candidate)
                kinds.remove(preferred)
                found = True
                break
            except ValueError as exc:
                last_reason = str(exc)
                continue
        if not found:
            raise ValueError("Cannot place route-crossing obstacle: " + last_reason + "; edit manual layout or seed")
    # Place large footprints first to avoid fragmenting the available space.
    kinds.sort(key=lambda kind: SIZES[kind], reverse=True)
    for index, kind in enumerate(kinds):
        for attempt in range(4000):
            o = Obstacle(
                kind,
                round(rng.uniform(-2.6, -0.8), 4),
                round(rng.uniform(-4.1, 4.1), 4),
                SIZES[kind],
                rng.choice((0, 15, 30, 45)),
            )
            if index == 0 and not manual and approach is None:
                # Keep a real obstacle close to the protected driving lane, rather
                # than placing the entire challenge at the remote course edges.
                j = rng.randrange(150, 360)
                tangent = witness[j + 1] - witness[j - 1]
                normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
                point = witness[j] + normal * (radius(o) + rng.uniform(0.67, 0.80))
                o.x, o.y = map(float, np.round(point, 4))
            try:
                validate(obstacles + [o], protect_start=protect_start)
            except ValueError:
                continue
            if not cable_lane_clear(o):
                continue
            if candidate_check is not None and not candidate_check(o)["valid"]:
                continue
            if not segment_clear((-0.8, 0), GOAL, [o], clearance=0.65):
                continue
            if not path_clear(witness, [o], clearance=0.65):
                continue
            obstacles.append(o)
            break
        else:
            raise ValueError(
                "Cannot fit requested course; change seed or manual layout"
            )
    return tuple(obstacles)


def load_manual(path):
    return tuple(Obstacle(**o) for o in json.loads(Path(path).read_text())["obstacles"])


def add_obstacles(spec, obstacles, protect_start=True):
    validate(obstacles, protect_start=protect_start)
    g = mujoco.mjtGeom
    for i, o in enumerate(obstacles):
        # Separate world-child bodies retain contact with base_link, which excludes world.
        body = spec.worldbody.add_body(
            name=f"obstacle_{i}",
            pos=[o.x, o.y, 0],
            quat=[
                math.cos(math.radians(o.angle) / 2),
                0,
                0,
                math.sin(math.radians(o.angle) / 2),
            ],
        )
        kw = {
            "name": f"obstacle_{i}_geom",
            "friction": [0.8, 0.005, 0.0001],
            "contype": 3,
            "conaffinity": 3,
            "rgba": [0.9, 0.5, 0.12, 1],
        }
        if o.kind == "stone":
            body.add_geom(
                type=g.mjGEOM_ELLIPSOID,
                size=[o.size, o.size, o.size / 2],
                pos=[0, 0, o.size / 2],
                **kw,
            )
        elif o.kind == "cone":
            # Convex mesh gives a true cone approximation rather than a cylinder.
            n = 12
            vertices = [
                [
                    o.size * math.cos(k * 2 * math.pi / n),
                    o.size * math.sin(k * 2 * math.pi / n),
                    0,
                ]
                for k in range(n)
            ] + [[0, 0, 2 * o.size]]
            spec.add_mesh(name=f"cone_{i}", uservert=[v for pt in vertices for v in pt])
            body.add_geom(type=g.mjGEOM_MESH, meshname=f"cone_{i}", **kw)
        elif o.kind == "slope":
            vertices = [
                [x, y, z]
                for x, z in ((-o.size, 0), (o.size, 0), (o.size, 0.18))
                for y in (-o.size, o.size)
            ]
            spec.add_mesh(
                name=f"slope_{i}", uservert=[v for pt in vertices for v in pt]
            )
            body.add_geom(type=g.mjGEOM_MESH, meshname=f"slope_{i}", **kw)
        else:
            h = 0.16 if o.kind == "barrier" else o.size
            body.add_geom(
                type=g.mjGEOM_BOX,
                size=[o.size, o.size if o.kind == "box" else 0.055, h],
                pos=[0, 0, h],
                **kw,
            )
