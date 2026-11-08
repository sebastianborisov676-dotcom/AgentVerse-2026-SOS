"""Every-physics-step contact/load telemetry; pair episodes are not impact counts."""

from collections import Counter
from types import SimpleNamespace

import mujoco
import numpy as np

from autocharge.metrics import _eq_rows

CATEGORIES = (
    "wheel_floor",
    "cable_floor",
    "connector_socket",
    "obstacle_robot",
    "obstacle_cable",
    "obstacle_other",
    "cable_robot",
    "robot_self",
    "unsafe_robot_environment",
    "other",
)


class ContactTelemetry:
    def __init__(self, model):
        self.model = model
        self.robot = set()
        root = model.body("base_link").id
        for b in range(model.nbody):
            parent = b
            while parent and parent != root:
                parent = int(model.body_parentid[parent])
            if parent == root:
                self.robot.add(b)
        self.connector = model.body("connector").id
        self.floor = model.geom("floor").id
        self.names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            for g in range(model.ngeom)
        ]
        self.wheels = {
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == root
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        }
        self.socket = model.body("socket").id
        self.samples = Counter(dict.fromkeys(CATEGORIES, 0))
        self.episodes = self.samples.copy()
        self.peaks = dict.fromkeys(CATEGORIES, 0.0)
        self.penetration = dict.fromkeys(CATEGORIES, 0.0)
        self.active = set()
        self.pairs = {}
        self.steps = 0

    def classify(self, a, b):
        # Negative geom id is a flex endpoint; flex contact is identified before indexing.
        if a < 0 or b < 0:
            g = max(a, b)
            if g < 0:
                return "other"
            if g == self.floor:
                return "cable_floor"
            if self.names[g].startswith("obstacle_"):
                return "obstacle_cable"
            if self.model.geom_bodyid[g] in self.robot:
                return "cable_robot"
            return "other"
        ba, bb = (int(self.model.geom_bodyid[g]) for g in (a, b))
        if a == self.floor or b == self.floor:
            other = b if a == self.floor else a
            if other in self.wheels:
                return "wheel_floor"
        if (ba == self.connector and bb == self.socket) or (
            bb == self.connector and ba == self.socket
        ):
            return "connector_socket"
        if any(self.names[g].startswith("obstacle_") for g in (a, b)):
            return (
                "obstacle_robot"
                if ba in self.robot or bb in self.robot
                else "obstacle_other"
            )
        if ba in self.robot and bb in self.robot:
            return "robot_self"
        if ba in self.robot or bb in self.robot:
            return "unsafe_robot_environment"
        return "other"

    def sample(self, data):
        current = set()
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            # Ignore inactive proximity contacts and forces below 1 mN.
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, data, i, force)
            normal = abs(float(force[0]))
            if normal <= 0.001:
                continue
            a, b = map(int, c.geom)
            category = self.classify(a, b)
            labels = sorted(self.names[g] if g >= 0 else "cable" for g in (a, b))
            key = (category, *labels)
            current.add(key)
            self.samples[category] += 1
            self.peaks[category] = max(self.peaks[category], normal)
            self.penetration[category] = max(
                self.penetration[category], max(0.0, -float(c.dist))
            )
        for key in current - self.active:
            self.episodes[key[0]] += 1
            label = "|".join(key)
            self.pairs[label] = self.pairs.get(label, 0) + 1
        self.active = current
        self.steps += 1

    def report(self):
        return {
            "steps": self.steps,
            "pair_episodes": dict(self.episodes),
            "contact_samples": dict(self.samples),
            "peak_normal_N": self.peaks.copy(),
            "max_penetration_m": self.penetration.copy(),
            "pairs": self.pairs.copy(),
        }


class EpisodeTelemetry:
    def __init__(self, model):
        self.contacts = ContactTelemetry(model)
        flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cable")
        self.flex_eq = next(
            e
            for e in range(model.neq)
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_FLEX and model.eq_obj1id[e] == flex
        )
        self.attach_eq = [
            model.equality(n).id
            for n in ("strain_relief_pin", "strain_relief_boot_pin")
        ]
        self.peak_tension = self.peak_strain_relief = 0.0
        self.violations = Counter(tension=0, strain_relief=0)
        self.violation_episodes = Counter(tension=0, strain_relief=0)
        self.active = set()
        self.first_violation = None

    def sample(self, data, limits):
        self.contacts.sample(data)
        model = self.contacts.model
        tensions = np.abs(_eq_rows(model, data, self.flex_eq))
        sr = np.zeros(3)
        for eq in self.attach_eq:
            force = _eq_rows(model, data, eq)
            if force.size == 3:
                sr += force
        cab = SimpleNamespace(
            max_tension=float(np.max(tensions)) if tensions.size else 0.0,
            strain_relief_load=float(np.linalg.norm(sr)),
        )
        self.peak_tension = max(self.peak_tension, cab.max_tension)
        self.peak_strain_relief = max(self.peak_strain_relief, cab.strain_relief_load)
        active = set()
        for name, value, limit in (
            ("tension", cab.max_tension, limits.cable_tension_max),
            ("strain_relief", cab.strain_relief_load, limits.strain_relief_max),
        ):
            if value > limit:
                active.add(name)
                self.violations[name] += 1
                if name not in self.active:
                    self.violation_episodes[name] += 1
                if self.first_violation is None:
                    self.first_violation = {
                        "time": float(data.time),
                        "kind": name,
                        "value_N": value,
                        "limit_N": limit,
                    }
        self.active = active
        return cab

    def report(self):
        return {
            "contacts": self.contacts.report(),
            "cable": {
                "peak_tension_N": self.peak_tension,
                "peak_strain_relief_N": self.peak_strain_relief,
                "violation_samples": dict(self.violations),
                "violation_episodes": dict(self.violation_episodes),
                "first_violation": self.first_violation,
            },
        }
