"""Scene builder: Google Robot + CCS2-style connector + flexible cable + charger + vehicle socket.

The Google Robot MJCF from MuJoCo Menagerie is *included unmodified*. Everything AutoCharger
adds is generated here:

* ``charger`` - stationary charging pedestal with a cable outlet (cable anchor).
* ``cable`` - 1D flexcomp rod (MuJoCo 4 discrete elastic rod: bending + torsion elasticity,
  inextensible edges via edge equality, frictional contact). Its first edge is clamped in the
  charger outlet.
* ``connector`` - simplified CCS2-style plug, added as a child body of ``link_gripper`` at
  compile time (rigid mount, no joint). Its rear handle carries a strain-relief boot; the last
  cable edge is held to the boot by two site-to-site connect constraints.
* ``vehicle`` - static car body with a simplified CCS2-style inlet (AC upper + two DC lower
  cavities built from ring segments, with radial clearance).

Frames: the plug face site ``plug_face`` and the socket site ``socket_face`` share the same
convention: +z is the insertion direction and +x is connector "up". When the two frames
coincide the connector is fully seated.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import pathlib

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOT_DIR = PROJECT_ROOT / "models" / "mujoco_menagerie" / "google_robot"

ARM_JOINTS = (
    "joint_torso",
    "joint_shoulder",
    "joint_bicep",
    "joint_elbow",
    "joint_forearm",
    "joint_wrist",
    "joint_gripper",
)
FINGER_JOINTS = ("joint_finger_right", "joint_finger_left")
# Optional planar mobile base (added at load time; the Menagerie model has no wheel joints).
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
BASE_ACTUATORS = ("base_x_vel", "base_y_vel", "base_yaw_vel")

# Collision bitmasks. Robot meshes, floor and cable use the MuJoCo default (1/1).
# The connector uses bit 2 only, so it never collides with the gripper it is mounted in or with
# the cable it is attached to. The vehicle uses bits 1|2 so it collides with everything.
CONNECTOR_BITS = 2
VEHICLE_BITS = 3

# Connector geometry (connector frame: +z insertion direction, +x up), metres.
FACE_Z = 0.05  # plug face plane
PIN_LENGTH = 0.036  # how far the pins protrude from the face
AC_CENTER = (0.012, 0.0)
AC_RADIUS = 0.024
DC_CENTERS = ((-0.032, 0.016), (-0.032, -0.016))
DC_RADIUS = 0.008
STRAIN_RELIEF_POS = np.array([-0.118, 0.0, -0.075])
STRAIN_RELIEF_EXIT = np.array([-1.0, 0.0, 0.0])  # cable leaves the boot along connector -x

# Socket geometry (socket frame, same convention as connector).
CAVITY_DEPTH = 0.045

# Fiducial marker on the car body beside the inlet (like a printed target on the charging flap
# surround). The robot's camera detects it; the fixed marker-to-socket transform below is the
# inlet's known geometry, not the car's pose. ArUco DICT_4X4_50, id 7.
MARKER_ID = 7
MARKER_SIZE = 0.070  # m, black square edge
MARKER_PIX = 200  # texture pixels across the black square
MARKER_PAD = 40  # white quiet-zone pixels on each side
# Marker frame (OpenCV convention: x right, y up, z out towards the viewer) in the socket frame.
MARKER_POS_S = np.array([0.140, 0.0, CAVITY_DEPTH + 0.0145])
MARKER_ROT_S = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).T  # columns x_m, y_m, z_m
RING_WALL = 0.004
RING_SEGMENTS = 16


@dataclasses.dataclass(frozen=True)
class SceneParams:
    """Tunable scene parameters. Defaults are the validated configuration."""

    # Socket face centre in world coordinates; socket axis points along world +x (into the car).
    socket_pos: tuple[float, float, float] = (0.70, -0.32, 0.82)
    # Charger pedestal ground position and cable outlet height.
    charger_xy: tuple[float, float] = (-0.35, -0.70)
    outlet_height: float = 1.05
    # Cable.
    cable_segments: int = 40
    cable_radius: float = 0.012
    cable_mass_per_m: float = 0.9  # kg/m (real CCS2 DC cables are ~1-2 kg/m)
    cable_young: float = 6e7  # Pa, effective bending modulus of the composite cable
    cable_damping: float = 0.02  # Rayleigh damping (s)
    cable_friction: float = 0.8
    # Socket radial clearance per cavity.
    ac_clearance: float = 0.0020
    dc_clearance: float = 0.0015
    # Distance of the plug face in front of the socket face at the home pose.
    home_standoff: float = 0.22
    # Connector mass (kg). Real CCS2 handles are ~0.8-1.2 kg.
    connector_mass: float = 0.9
    timestep: float = 0.002
    # Mobile base: when True the robot base gets planar x / y / yaw joints driven by velocity
    # servos, and starts at ``base_start`` (x m, y m, yaw deg). The docking pose is the origin,
    # where the arm's home pose was solved.
    mobile_base: bool = False
    base_start: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclasses.dataclass
class SceneInfo:
    """Facts computed while building the scene (useful for tests and reports)."""

    params: SceneParams
    home_qpos_arm: np.ndarray
    ik_error_pos: float
    ik_error_rot: float
    cable_length: float
    cable_segment: float
    anchor_to_connector: float  # straight-line outlet -> strain relief at home pose
    socket_rot: np.ndarray


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------


def socket_rotation() -> np.ndarray:
    """World rotation of the socket frame: +z along world +x, +x along world +z."""
    x = np.array([0.0, 0.0, 1.0])
    z = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat).reshape(9))
    return q


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.ravel(v))


def _ring_geoms(cx: float, cy: float, inner_r: float, prefix: str) -> list[str]:
    """Ring of boxes forming a cylindrical cavity of radius ``inner_r`` (socket frame)."""
    geoms = []
    rc = inner_r + RING_WALL / 2
    half_tan = inner_r * math.tan(math.pi / RING_SEGMENTS) + RING_WALL / 2
    for k in range(RING_SEGMENTS):
        th = 2 * math.pi * k / RING_SEGMENTS
        pos = (cx + rc * math.cos(th), cy + rc * math.sin(th), CAVITY_DEPTH / 2)
        quat = (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))
        geoms.append(
            f'<geom name="{prefix}_wall{k}" class="socket" type="box" pos="{_fmt(pos)}" '
            f'quat="{_fmt(quat)}" size="{RING_WALL / 2:.6g} {half_tan:.6g} {CAVITY_DEPTH / 2:.6g}"/>'
        )
    geoms.append(
        f'<geom name="{prefix}_floor" class="socket" type="cylinder" '
        f'pos="{cx:.6g} {cy:.6g} {CAVITY_DEPTH + 0.003:.6g}" size="{inner_r + RING_WALL:.6g} 0.003"/>'
    )
    return geoms


def _socket_xml(p: SceneParams) -> str:
    rot = socket_rotation()
    quat = _mat2quat(rot)
    ac_r = AC_RADIUS + p.ac_clearance
    dc_r = DC_RADIUS + p.dc_clearance
    geoms = _ring_geoms(*AC_CENTER, ac_r, "ac")
    for i, c in enumerate(DC_CENTERS):
        geoms += _ring_geoms(*c, dc_r, f"dc{i}")
    sx, sy, _ = p.socket_pos
    car_face_x = sx + CAVITY_DEPTH + 0.015
    return f"""
    <body name="vehicle" pos="0 0 0" mocap="true">
      <geom name="car_body" class="vehicle" type="box" pos="{car_face_x + 0.9:.6g} {sy + 0.2:.6g} 0.75"
            size="0.9 1.0 0.43" rgba="0.62 0.05 0.06 1"/>
      <geom name="car_wheel" type="cylinder" pos="{car_face_x + 0.45:.6g} {sy - 0.55:.6g} 0.32"
            size="0.32 0.11" quat="0.7071 0.7071 0 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <body name="socket" pos="{_fmt(p.socket_pos)}" quat="{_fmt(quat)}">
        <site name="socket_face" pos="0 0 0" size="0.004" rgba="0 1 0 1" group="4"/>
        <geom name="marker_sticker" type="box" pos="0.12 0.018 0.0585" size="0.02 0.02 0.0005"
              rgba="0.95 0.85 0.1 0" contype="0" conaffinity="0"/>
        <geom name="charge_flap" type="box" pos="-0.008 0 -0.004" size="0.07 0.06 0.002"
              rgba="0.62 0.05 0.06 0" contype="4" conaffinity="4"/>
        <geom name="inlet_backplate" class="socket" type="box" pos="-0.008 0 {CAVITY_DEPTH + 0.010:.6g}"
              size="0.075 0.065 0.004" rgba="0.12 0.12 0.12 1"/>
        <geom name="inlet_recess_bottom" class="socket" type="box" pos="-0.099 0 {(CAVITY_DEPTH + 0.0145) / 2:.6g}"
              size="0.004 0.07 {(CAVITY_DEPTH + 0.0145) / 2:.6g}" rgba="0.2 0.2 0.2 1"/>
        {"".join(geoms)}
      </body>
    </body>"""


def _charger_xml(p: SceneParams) -> str:
    cx, cy = p.charger_xy
    h = p.outlet_height
    return f"""
    <body name="charger" pos="{cx:.6g} {cy:.6g} 0">
      <geom name="charger_pedestal" type="box" pos="0 0 {h / 2 + 0.1:.6g}" size="0.10 0.08 {h / 2 + 0.1:.6g}"
            rgba="0.85 0.86 0.88 1"/>
      <geom name="charger_screen" type="box" pos="0.101 0 {h + 0.05:.6g}" size="0.002 0.05 0.07"
            rgba="0.05 0.35 0.75 1" contype="0" conaffinity="0"/>
      <geom name="charger_holster" type="cylinder" pos="0 -0.095 {h:.6g}" quat="0.7071 0.7071 0 0"
            size="0.022 0.015" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <body name="cable_outlet" pos="0 -0.11 {h:.6g}">
        <site name="cable_anchor" size="0.006" rgba="1 0.5 0 1"/>
        CABLE_PLACEHOLDER
      </body>
    </body>"""


def _cable_xml(p: SceneParams, seg: float) -> str:
    """Flexcomp cable with a straight rest shape along -y from the outlet."""
    n = p.cable_segments
    points = np.zeros((n + 1, 3))
    points[:, 1] = -seg * np.arange(n + 1)
    mass = p.cable_mass_per_m * seg * n
    return f"""
        <flexcomp name="cable" type="direct" dim="1" point="{_fmt(points)}" radius="{p.cable_radius}"
                  mass="{mass:.6g}" rgba="0.05 0.05 0.05 1">
          <edge equality="true" solref="0.004 1"/>
          <elasticity young="{p.cable_young:.6g}" damping="{p.cable_damping:.6g}"/>
          <contact selfcollide="none" condim="3" friction="{p.cable_friction} 0.005 0.0001"
                   solref="0.004 1"/>
          <pin id="0 1"/>
        </flexcomp>"""


def _scene_xml(p: SceneParams, seg: float | None) -> str:
    robot = (ROBOT_DIR / "robot.xml").as_posix()
    assets = (ROBOT_DIR / "assets").as_posix()
    charger = _charger_xml(p).replace(
        "CABLE_PLACEHOLDER", _cable_xml(p, seg) if seg is not None else ""
    )
    return f"""<mujoco model="autocharge_google_robot">
  <include file="{robot}"/>
  <compiler assetdir="{assets}"/>
  <option timestep="{p.timestep}" integrator="discrete" solver="Newton" tolerance="1e-8" iterations="100"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="210" elevation="-20" offwidth="1280" offheight="960"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.35 0.35 0.37"
             rgb2="0.30 0.30 0.32" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
  </asset>
  <default>
    <default class="vehicle">
      <geom contype="{VEHICLE_BITS}" conaffinity="{VEHICLE_BITS}" friction="0.6 0.005 0.0001"/>
      <default class="socket">
        <geom rgba="0.18 0.18 0.18 1" friction="0.3 0.005 0.0001" solref="0.008 1"/>
      </default>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 3"/>
    <light pos="1 -1 1.5" dir="-1 1 -1" diffuse="0.5 0.5 0.5" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.9 0.005 0.0001"/>
    <camera name="overview" pos="0.2 -2.4 1.6" xyaxes="1 0 0 0 0.55 0.83"/>
    <!-- Environment props, hidden by default (alpha 0, collision bit 4 = inert) and switched on per challenge. -->
    <geom name="bay_line_l" type="box" pos="0.9 0.75 0.001" size="1.6 0.04 0.001" rgba="0.95 0.95 0.9 0" contype="0" conaffinity="0"/>
    <geom name="bay_line_r" type="box" pos="0.9 -1.35 0.001" size="1.6 0.04 0.001" rgba="0.95 0.95 0.9 0" contype="0" conaffinity="0"/>
    <geom name="bay_line_back" type="box" pos="-0.7 -0.3 0.001" size="0.04 1.05 0.001" rgba="0.95 0.95 0.9 0" contype="0" conaffinity="0"/>
    <geom name="lane_arrow" type="box" pos="0.2 -0.3 0.001" size="0.25 0.02 0.001" rgba="0.95 0.8 0.1 0" contype="0" conaffinity="0"/>
    <geom name="pillar_1" type="cylinder" pos="-0.6 1.3 1.2" size="0.18 1.2" rgba="0.62 0.62 0.6 0" contype="4" conaffinity="4"/>
    <geom name="pillar_2" type="cylinder" pos="1.6 -2.1 1.2" size="0.18 1.2" rgba="0.62 0.62 0.6 0" contype="4" conaffinity="4"/>
    <geom name="parked_car" type="box" pos="2.4 2.6 0.7" size="0.9 0.95 0.45" rgba="0.15 0.3 0.7 0" contype="4" conaffinity="4"/>
    <geom name="parked_car_wheel" type="cylinder" pos="2.0 1.62 0.32" size="0.32 0.11" quat="0.7071 0.7071 0 0" rgba="0.08 0.08 0.08 0" contype="0" conaffinity="0"/>
    <geom name="obstacle" type="box" pos="0.57 -0.30 0.80" size="0.008 0.06 0.06"
          rgba="0.95 0.6 0.1 0" contype="4" conaffinity="4"/>
    <camera name="socket_view" pos="0.25 -0.32 0.95" xyaxes="0 -1 0 0.25 0 0.97"/>
    {_charger_xml_placeholder()}
    {_socket_xml(p)}
  </worldbody>
</mujoco>""".replace("CHARGER_PLACEHOLDER", charger)


def _charger_xml_placeholder() -> str:
    return "CHARGER_PLACEHOLDER"


def _add_marker(spec: mujoco.MjSpec) -> None:
    """Textured ArUco marker on the car body, a child of the socket so it moves with the car."""
    import cv2  # local import: only needed to build the texture

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, MARKER_ID, MARKER_PIX)
    img = cv2.copyMakeBorder(img, MARKER_PAD, MARKER_PAD, MARKER_PAD, MARKER_PAD, cv2.BORDER_CONSTANT, value=255)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    n = rgb.shape[0]
    tex = spec.add_texture(name="aruco_marker", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=n, nchannel=3)
    tex.data = rgb.tobytes()
    mat = spec.add_material(name="aruco_marker")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "aruco_marker"
    half = MARKER_SIZE * n / MARKER_PIX / 2
    g = spec.body("socket").add_geom(
        name="aruco_marker", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[half, half, 0.01],
        pos=list(MARKER_POS_S), quat=list(_mat2quat(MARKER_ROT_S)), material="aruco_marker",
    )
    g.contype = g.conaffinity = 0


def _add_connector(spec: mujoco.MjSpec, p: SceneParams, boot_offset: float) -> None:
    """Rigidly mount the CCS2-style connector on the robot's gripper (no joint)."""
    gripper = spec.body("link_gripper")
    # The gripper site of the robot model is at (0, 0, 0.13) in link_gripper.
    conn = gripper.add_body(name="connector", pos=[0, 0, 0.13], gravcomp=1.0)
    conn.mass = p.connector_mass
    conn.ipos = [-0.03, 0, -0.02]
    conn.inertia = [0.004, 0.004, 0.002]
    conn.explicitinertial = True

    g = mujoco.mjtGeom

    def geom(name, gtype, size, pos, rgba, quat=None, fromto=None, collide=True):
        kw = {"name": name, "type": gtype, "size": size, "rgba": rgba, "mass": 0}
        if fromto is not None:
            kw["fromto"] = fromto
        else:
            kw["pos"] = pos
        if quat is not None:
            kw["quat"] = quat
        gm = conn.add_geom(**kw)
        gm.contype = CONNECTOR_BITS if collide else 0
        gm.conaffinity = CONNECTOR_BITS if collide else 0
        gm.friction = [0.3, 0.005, 0.0001]
        return gm

    black = [0.08, 0.08, 0.09, 1]
    grey = [0.25, 0.25, 0.27, 1]
    # Housing (front shell around the pins).
    geom("conn_housing", g.mjGEOM_BOX, [0.0425, 0.034, 0.05], [-0.0025, 0, 0.0], black)
    # Handle / grip and strain relief boot (rear, hanging below the housing).
    geom("conn_handle", g.mjGEOM_BOX, [0.035, 0.026, 0.03], [-0.07, 0, -0.06], black)
    sr = STRAIN_RELIEF_POS
    geom(
        "conn_boot",
        g.mjGEOM_CAPSULE,
        [0.017, 0, 0],
        None,
        grey,
        fromto=list(sr - STRAIN_RELIEF_EXIT * 0.03) + list(sr),
        collide=False,
    )
    # Pins (the only parts expected to enter the socket): a cylinder plus an ellipsoidal cap that
    # acts as the lead-in chamfer of a real contact pin (sharp cylinder edges catch on the
    # segmented socket rim and produce unphysical solver spikes).
    pins = [("conn_pin_ac", AC_CENTER, AC_RADIUS, 0.004)]
    pins += [(f"conn_pin_dc{i}", c, DC_RADIUS, 0.003) for i, c in enumerate(DC_CENTERS)]
    for name, (cx, cy), radius, chamfer in pins:
        body_half = (PIN_LENGTH - chamfer) / 2
        geom(name, g.mjGEOM_CYLINDER, [radius, body_half, 0], [cx, cy, FACE_Z + body_half], grey)
        geom(
            f"{name}_tip",
            g.mjGEOM_ELLIPSOID,
            [radius, radius, chamfer],
            [cx, cy, FACE_Z + PIN_LENGTH - chamfer],
            grey,
        )
    conn.add_site(name="plug_face", pos=[0, 0, FACE_Z], size=[0.004, 0, 0], rgba=[0, 1, 1, 1], group=4)
    conn.add_site(name="plug_tip", pos=[0, 0, FACE_Z + PIN_LENGTH], size=[0.003, 0, 0], group=4)
    conn.add_site(name="strain_relief", pos=list(sr), size=[0.005, 0, 0], rgba=[1, 0.5, 0, 1])
    conn.add_site(
        name="strain_relief_boot",
        pos=list(sr + STRAIN_RELIEF_EXIT * boot_offset),
        size=[0.004, 0, 0],
        rgba=[1, 0.5, 0, 1],
        group=4,
    )
    # Robot camera on top of the housing, looking along the insertion axis (+z), image-up ~ +x.
    conn.add_camera(
        name="connector_cam", pos=[0.052, 0, 0.03], fovy=70,
        # tilted 20 deg up so the inlet marker stays in view all the way to the seated position
        xyaxes=[0, 1, 0, math.cos(math.radians(20)), 0, -math.sin(math.radians(20))],
    )


# --------------------------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------------------------


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` (3x3) to ``target`` (3x3)."""
    q_cur = _mat2quat(current)
    q_tgt = _mat2quat(target)
    q_cur_inv = np.zeros(4)
    mujoco.mju_negQuat(q_cur_inv, q_cur)
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_tgt, q_cur_inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def arm_dof_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    jids = np.array([model.joint(n).id for n in ARM_JOINTS])
    return model.jnt_qposadr[jids], model.jnt_dofadr[jids]


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_init: np.ndarray,
    iters: int = 400,
) -> tuple[np.ndarray, float, float]:
    """Kinematic damped-least-squares IK over the 7 arm joints. Returns (q, pos_err, rot_err)."""
    qadr, dadr = arm_dof_indices(model)
    jids = [model.joint(n).id for n in ARM_JOINTS]
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    sid = model.site(site).id
    q = np.clip(q_init.copy(), lo + 0.05, hi - 0.05)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pe = re = np.inf
    for _ in range(iters):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        ep = target_pos - data.site_xpos[sid]
        er = orientation_error(data.site_xmat[sid].reshape(3, 3), target_rot)
        pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
        if pe < 1e-5 and re < 1e-4:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        jac = np.vstack([jacp[:, dadr], jacr[:, dadr]])
        err = np.concatenate([ep, er])
        lam = 1e-3
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)
        # Mild pull away from joint limits in the null space.
        mid = 0.5 * (lo + hi)
        null = np.eye(len(q)) - np.linalg.pinv(jac) @ jac
        dq += null @ (0.02 * (mid - q))
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    return q, pe, re


# --------------------------------------------------------------------------------------------
# Cable routing
# --------------------------------------------------------------------------------------------


def _chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        new = [points[0]]
        for a, b in itertools.pairwise(points):
            new.append(0.75 * a + 0.25 * b)
            new.append(0.25 * a + 0.75 * b)
        new.append(points[-1])
        points = np.array(new)
    return points


def _densify(points: np.ndarray, step: float = 0.002) -> np.ndarray:
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.linspace(0, s[-1], int(s[-1] / step) + 2)
    return np.column_stack([np.interp(t, s, points[:, k]) for k in range(3)])


def _march(dense: np.ndarray, seg: float, n_seg: int) -> tuple[np.ndarray, float]:
    """Walk along ``dense`` placing points exactly ``seg`` apart (chord length)."""
    pts = [dense[0]]
    i = 0
    for _ in range(n_seg):
        start = pts[-1]
        while i < len(dense) - 1 and np.linalg.norm(dense[i + 1] - start) < seg:
            i += 1
        if i >= len(dense) - 1:
            pts.append(dense[-1])
            continue
        a, b = dense[i], dense[i + 1]
        # Solve |a + u (b - a) - start| = seg for u in [0, 1].
        ab, as_ = b - a, a - start
        qa, qb, qc = ab @ ab, 2 * ab @ as_, as_ @ as_ - seg * seg
        u = (-qb + math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
        pts.append(a + u * ab)
    pts = np.array(pts)
    return pts, float(np.linalg.norm(pts[-1] - dense[-1]))


def _resample(points: np.ndarray, n_seg: int) -> tuple[np.ndarray, float]:
    """Equal-chord resampling so the cable starts at its rest edge lengths (no initial strain)."""
    dense = _densify(points)
    arc = float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
    lo, hi = arc / n_seg * 0.9, arc / n_seg
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pts, _ = _march(dense, mid, n_seg)
        # Too short a chord leaves the last point before the end of the curve.
        remaining = np.linalg.norm(dense[-1] - pts[-2]) - mid
        if remaining > 0:
            lo = mid
        else:
            hi = mid
    seg = 0.5 * (lo + hi)
    pts, _ = _march(dense, seg, n_seg)
    pts[-1] = dense[-1]
    return pts, seg


def route_cable(
    p: SceneParams, anchor: np.ndarray, sr_pos: np.ndarray, exit_dir: np.ndarray
) -> tuple[np.ndarray, float]:
    """Initial slack route: out of the outlet, down to the floor, along it, up into the boot."""
    r = p.cable_radius + 0.0005  # start resting just above the floor
    out_dir = np.array([0.0, -1.0, 0.0])
    ctrl = np.array(
        [
            anchor,
            anchor + out_dir * 0.26,
            anchor + out_dir * 0.40 + np.array([0, 0, -0.25]),
            np.array([anchor[0] + 0.05, anchor[1] - 0.40, r]),
            np.array([0.25, anchor[1] - 0.30, r]),
            np.array([sr_pos[0] - 0.05, sr_pos[1] - 0.25, r]),
            sr_pos + exit_dir * 0.35,
            sr_pos,
        ]
    )
    smooth = _chaikin(ctrl, 3)
    pts, seg = _resample(smooth, p.cable_segments)
    # Exact end conditions: first edge along the outlet direction, last edge along the boot axis
    # (both ends of the smoothed route are straight for longer than one segment).
    pts[0] = anchor
    pts[1] = anchor + out_dir * seg
    pts[-1] = sr_pos
    pts[-2] = sr_pos + exit_dir * seg
    return pts, seg


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def _make_mobile(spec: mujoco.MjSpec) -> None:
    """Give the robot base planar joints + velocity servos (unicycle-style drive is commanded by
    the drive controller). The robot XML file is untouched; this edits the in-memory spec only."""
    base = spec.body("base_link")
    for name, jtype, axis in (
        ("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
        ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
        ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1]),
    ):
        j = base.add_joint(name=name, type=jtype, axis=axis)
        j.damping = [0.0, 0.0, 0.0]
        j.frictionloss, j.armature = 0.0, 0.0
    for name, joint, kv, frc in (
        ("base_x_vel", "base_x", 3000.0, 800.0),
        ("base_y_vel", "base_y", 3000.0, 800.0),
        ("base_yaw_vel", "base_yaw", 400.0, 300.0),
    ):
        a = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gainprm[0] = kv
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[2] = -kv
        a.ctrlrange = [-1.0, 1.0]
        a.forcerange = [-frc, frc]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # The base rides on the floor kinematically (no wheel-floor contact dynamics).
    spec.add_exclude(bodyname1="world", bodyname2="base_link")


def route_cable_mobile(
    p: SceneParams, anchor: np.ndarray, sr_pos: np.ndarray, exit_dir: np.ndarray
) -> tuple[np.ndarray, float]:
    """Initial route for a distant robot: out of the outlet, along the floor to the robot's
    right-hand side (clear of its drive path), then up into the boot."""
    r = p.cable_radius + 0.0005
    out_dir = np.array([0.0, -1.0, 0.0])
    ctrl = np.array(
        [
            anchor,
            anchor + out_dir * 0.26,
            anchor + out_dir * 0.40 + np.array([0, 0, -0.25]),
            np.array([anchor[0] + 0.05, anchor[1] - 0.40, r]),
            np.array([0.5 * (anchor[0] + sr_pos[0]), anchor[1] - 0.45, r]),
            np.array([sr_pos[0] + 0.10, sr_pos[1] - 0.60, r]),
            sr_pos + exit_dir * 0.35,
            sr_pos,
        ]
    )
    pts, seg = _resample(_chaikin(ctrl, 3), p.cable_segments)
    pts[0] = anchor
    pts[1] = anchor + out_dir * seg
    pts[-1] = sr_pos
    pts[-2] = sr_pos + exit_dir * seg
    return pts, seg


def _set_base(model, qpos, pose) -> None:
    x, y, yaw_deg = pose
    for name, v in zip(BASE_JOINTS, (x, y, math.radians(yaw_deg)), strict=True):
        qpos[model.jnt_qposadr[model.joint(name).id]] = v


def _home_ik(model, data, p: SceneParams):
    rot = socket_rotation()
    target = np.array(p.socket_pos) - rot[:, 2] * p.home_standoff
    best = None
    rng = np.random.default_rng(0)
    seeds = [np.array([0.0, 0.6, 0.0, 1.4, 0.0, 0.8, 0.0])]
    seeds += [rng.uniform(-1.5, 1.5, 7) for _ in range(24)]
    for seed in seeds:
        q, pe, re = solve_ik(model, data, "plug_face", target, rot, seed)
        jids = [model.joint(n).id for n in ARM_JOINTS]
        lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        score = pe * 100 + re * 10 - min(margin, 0.3)
        if best is None or score < best[0]:
            best = (score, q, pe, re)
    return best[1], best[2], best[3]


def build_model(params: SceneParams | None = None) -> tuple[mujoco.MjModel, SceneInfo]:
    """Build and compile the full AutoCharger scene. The robot XML file is never modified."""
    p = params or SceneParams()
    if not (ROBOT_DIR / "robot.xml").is_file():
        raise FileNotFoundError(
            f"Google Robot model not found in {ROBOT_DIR}. Run: bash scripts/setup.sh"
        )

    # Pass 1: no cable, solve the home pose and locate the strain relief.
    spec0 = mujoco.MjSpec.from_string(_scene_xml(p, None))
    _add_connector(spec0, p, boot_offset=0.06)
    _add_marker(spec0)
    if p.mobile_base:
        _make_mobile(spec0)
    m0 = spec0.compile()
    d0 = mujoco.MjData(m0)
    q_home, ik_pe, ik_re = _home_ik(m0, d0, p)  # solved with the base at the docking pose
    qadr, _ = arm_dof_indices(m0)
    d0.qpos[qadr] = q_home
    if p.mobile_base:
        _set_base(m0, d0.qpos, p.base_start)
    mujoco.mj_kinematics(m0, d0)
    anchor = d0.site_xpos[m0.site("cable_anchor").id].copy()
    sr_pos = d0.site_xpos[m0.site("strain_relief").id].copy()
    conn_rot = d0.xmat[m0.body("connector").id].reshape(3, 3)
    exit_world = conn_rot @ STRAIN_RELIEF_EXIT
    router = route_cable_mobile if p.mobile_base else route_cable
    route, seg = router(p, anchor, sr_pos, exit_world)

    # Pass 2: full scene with cable, attachment constraints and home keyframe.
    spec = mujoco.MjSpec.from_string(_scene_xml(p, seg))
    _add_connector(spec, p, boot_offset=seg)
    _add_marker(spec)
    if p.mobile_base:
        _make_mobile(spec)
    n = p.cable_segments
    last, second_last = spec.body(f"cable_{n}"), spec.body(f"cable_{n - 1}")
    last.add_site(name="cable_end", size=[0.004, 0, 0], group=4)
    second_last.add_site(name="cable_end_prev", size=[0.004, 0, 0], group=4)
    for s1, s2, name in (
        ("cable_end", "strain_relief", "strain_relief_pin"),
        ("cable_end_prev", "strain_relief_boot", "strain_relief_boot_pin"),
    ):
        spec.add_equality(
            name=name,
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=s1,
            name2=s2,
            solref=[0.004, 1.0],
        )
    model = spec.compile()

    # Home keyframe: robot at the IK pose, cable vertices displaced from the straight rest line
    # onto the slack route (vertex bodies carry three slide joints each).
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    qadr, _ = arm_dof_indices(model)
    qpos[qadr] = q_home
    for fj in FINGER_JOINTS:
        qpos[model.jnt_qposadr[model.joint(fj).id]] = 0.6
    if p.mobile_base:
        _set_base(model, qpos, p.base_start)
    outlet_id = model.body("cable_outlet").id
    mujoco.mj_kinematics(model, data)
    outlet_pos = data.xpos[outlet_id].copy()
    outlet_rot = data.xmat[outlet_id].reshape(3, 3)
    for i in range(2, n + 1):
        b = model.body(f"cable_{i}").id
        rest_world = outlet_pos + outlet_rot @ model.body_pos[b]
        disp_local = outlet_rot.T @ (route[i] - rest_world)
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                qpos[model.jnt_qposadr[j]] = float(model.jnt_axis[j] @ disp_local)
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) in BASE_ACTUATORS:
            continue  # velocity servos: hold still
        jid = model.actuator_trnid[a, 0]
        ctrl[a] = qpos[model.jnt_qposadr[jid]]
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()

    info = SceneInfo(
        params=p,
        home_qpos_arm=q_home,
        ik_error_pos=ik_pe,
        ik_error_rot=ik_re,
        cable_length=seg * n,
        cable_segment=seg,
        anchor_to_connector=float(np.linalg.norm(sr_pos - anchor)),
        socket_rot=socket_rotation(),
    )
    return model, info


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, settle: float = 0.5) -> None:
    """Load the ``home`` keyframe and let the cable settle under gravity with the arm holding.

    Starting from rest, gravity is applied to the cable's edge constraints in a single step, which
    produces a short (~0.15 s) numerical tension spike of ~20 N. ``settle`` seconds of holding
    removes it so the controller and its safety limits see only physical loads.
    """
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for _ in range(round(settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
