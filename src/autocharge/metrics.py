"""Measurements used by the controller, tests and reports."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from autocharge.scene import orientation_error


@dataclasses.dataclass
class Alignment:
    lateral: float  # m, plug-face offset perpendicular to the socket axis
    axial: float  # m, plug-face position along the socket axis (0 = fully seated, <0 = outside)
    angle: float  # rad, total orientation error between plug and socket frames


def plug_alignment_error(model: mujoco.MjModel, data: mujoco.MjData) -> Alignment:
    pid, sid = model.site("plug_face").id, model.site("socket_face").id
    rot_s = data.site_xmat[sid].reshape(3, 3)
    d = rot_s.T @ (data.site_xpos[pid] - data.site_xpos[sid])
    ang = orientation_error(data.site_xmat[pid].reshape(3, 3), rot_s)
    return Alignment(float(np.hypot(d[0], d[1])), float(d[2]), float(np.linalg.norm(ang)))


@dataclasses.dataclass
class ContactForce:
    total_normal: float  # N, sum of normal forces between connector and vehicle/socket
    axial_resistance: float  # N, force on the connector opposing insertion (socket axis)
    lateral: float  # N, net force on the connector perpendicular to the socket axis
    count: int


def _vehicle_bodies(model: mujoco.MjModel) -> set[int]:
    root = model.body("vehicle").id
    return {b for b in range(model.nbody) if b == root or model.body_parentid[b] == root}


def connector_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> ContactForce:
    conn = model.body("connector").id
    veh = _vehicle_bodies(model)
    axis = data.site_xmat[model.site("socket_face").id].reshape(3, 3)[:, 2]
    f6 = np.zeros(6)
    net = np.zeros(3)
    total = 0.0
    count = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom[0] < 0 or c.geom[1] < 0:  # flex contacts
            continue
        b1, b2 = model.geom_bodyid[c.geom[0]], model.geom_bodyid[c.geom[1]]
        if b1 == conn and b2 in veh:
            sign = -1.0  # force on geom1 is opposite to the force on geom2
        elif b2 == conn and b1 in veh:
            sign = 1.0
        else:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        frame = c.frame.reshape(3, 3)
        net += sign * (frame.T @ f6[:3])
        total += abs(f6[0])
        count += 1
    axial = float(-(net @ axis))
    lateral = float(np.linalg.norm(net - (net @ axis) * axis))
    return ContactForce(total, axial, lateral, count)


@dataclasses.dataclass
class CableState:
    max_tension: float  # N, largest axial force over all cable edges (edge equality)
    end_tension: float  # N, axial force in the last edge (at the strain relief)
    strain_relief_load: float  # N, net force the cable applies to the connector boot
    max_edge_strain: float  # relative edge elongation (inextensibility check)
    min_clearance: float  # m, lowest cable surface point relative to the floor (<0 = penetration)
    length: float  # m, current total cable length
    span: float  # m, straight-line distance from anchor to strain relief
    max_speed: float  # m/s, fastest cable vertex


def _eq_rows(model, data, eq_id) -> np.ndarray:
    rows = np.nonzero(
        (data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        & (data.efc_id[: data.nefc] == eq_id)
    )[0]
    return data.efc_force[rows]


def cable_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> CableState:
    flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cable")
    vadr, vnum = model.flex_vertadr[flex], model.flex_vertnum[flex]
    eadr, enum = model.flex_edgeadr[flex], model.flex_edgenum[flex]

    flex_eq = [
        e
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_FLEX and model.eq_obj1id[e] == flex
    ]
    tensions = np.abs(_eq_rows(model, data, flex_eq[0])) if flex_eq else np.zeros(1)
    if tensions.size == 0:
        tensions = np.zeros(1)

    sr = np.zeros(3)
    for name in ("strain_relief_pin", "strain_relief_boot_pin"):
        f = _eq_rows(model, data, model.equality(name).id)
        if f.size == 3:
            sr += f

    lengths = data.flexedge_length[eadr : eadr + enum]
    rest = model.flexedge_length0[eadr : eadr + enum]
    strain = float(np.max(np.abs(lengths / rest - 1.0)))
    vpos = data.flexvert_xpos[vadr : vadr + vnum]
    clearance = float(np.min(vpos[:, 2]) - model.flex_radius[flex])
    anchor = data.site_xpos[model.site("cable_anchor").id]
    srp = data.site_xpos[model.site("strain_relief").id]

    speeds = []
    for i in range(2, vnum):
        b = model.body(f"cable_{i}").id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        speeds.append(np.linalg.norm(vel[3:]))

    return CableState(
        max_tension=float(np.max(tensions)),
        end_tension=float(tensions[-1]),
        strain_relief_load=float(np.linalg.norm(sr)),
        max_edge_strain=strain,
        min_clearance=clearance,
        length=float(np.sum(lengths)),
        span=float(np.linalg.norm(srp - anchor)),
        max_speed=float(max(speeds) if speeds else 0.0),
    )
