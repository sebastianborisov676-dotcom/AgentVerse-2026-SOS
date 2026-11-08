"""Independent evaluator: judges docking from the simulator's ground truth.

This module is the "answer key". It is used by tests and benchmarks only; nothing in the
autonomous path (perception, autonomy, controller decisions) imports it. It reads the socket
*body* pose (the physical inlet geometry), not the ``socket_face`` helper site, so the
ground-truth poisoning test can corrupt that site without affecting the judgement.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np

from autocharge.scene import PIN_LENGTH, orientation_error

AC_CLEARANCE = 0.0020  # matches SceneParams.ac_clearance default


@dataclasses.dataclass
class DockingVerdict:
    docked: bool
    depth_mm: float  # pin engagement (0 = pin tips at the socket face)
    lateral_mm: float  # plug axis offset from the socket axis
    angle_deg: float
    reason: str


def socket_pose_truth(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    b = model.body("socket").id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def check_docking(model: mujoco.MjModel, data: mujoco.MjData, min_depth: float = 0.030) -> DockingVerdict:
    """Mechanical docking = pins at least 30 mm deep, within the cavity clearance, axes aligned."""
    p_s, r_s = socket_pose_truth(model, data)
    c = model.body("connector").id
    r_c = data.xmat[c].reshape(3, 3)
    face = data.xpos[c] + r_c @ np.array([0.0, 0.0, 0.05])  # plug face in the connector frame
    d = r_s.T @ (face - p_s)
    depth = d[2] + PIN_LENGTH
    lateral = float(math.hypot(d[0], d[1]))
    ang = math.degrees(float(np.linalg.norm(orientation_error(r_c, r_s))))
    if depth < min_depth:
        reason = f"not inserted (depth {depth * 1e3:.1f} mm)"
    elif lateral > AC_CLEARANCE + 0.0005:
        reason = f"misaligned ({lateral * 1e3:.2f} mm off axis)"
    elif ang > 3.0:
        reason = f"tilted ({ang:.1f} deg)"
    else:
        reason = "docked"
    return DockingVerdict(reason == "docked", depth * 1e3, lateral * 1e3, ang, reason)
