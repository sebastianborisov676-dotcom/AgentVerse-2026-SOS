"""Camera perception: find the charging inlet in the robot camera image and estimate its 6-DoF pose.

Pipeline (runs locally, CPU only):

    MuJoCo camera render (RGB, 960x720)  ->  ArUco marker detection (OpenCV)
    -> sub-pixel corner refinement  ->  solvePnP (IPPE_SQUARE)  ->  reprojection check
    -> camera pose from the robot's own kinematics  ->  socket pose in the world frame

The marker is a fiducial fixed on the car body next to the inlet. The marker-to-socket transform is
the inlet's known geometry (a calibration constant), so the controller learns *where the car is*
only from the image. Ground truth is never read here; ``evaluate`` exists only for scoring.

Frames: W world, C camera (MuJoCo: x right, y up, looking -z), Cv the OpenCV camera
(x right, y down, looking +z), M marker, S socket (+z insertion direction, +x up).
"""

from __future__ import annotations

import dataclasses
import math
import time

import cv2
import mujoco
import numpy as np

from autocharge.scene import MARKER_ID, MARKER_POS_S, MARKER_ROT_S, MARKER_SIZE, orientation_error

CV_IN_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes expressed in the MuJoCo camera frame


@dataclasses.dataclass
class PortEstimate:
    detected: bool
    timestamp: float  # simulation time of the image (s)
    confidence: float = 0.0  # 0..1 heuristic from reprojection error and marker size (not a probability)
    position: np.ndarray | None = None  # socket face centre, world frame (m)
    rotation: np.ndarray | None = None  # socket frame in the world (3x3)
    reproj_px: float = float("nan")
    marker_px: float = 0.0  # marker edge length in the image (px)
    range_m: float = float("nan")  # camera to marker distance
    corners_px: np.ndarray | None = None
    latency_ms: float = 0.0  # render + detection + PnP wall time
    reason: str = ""  # why a detection was rejected


class PortPerception:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera: str = "connector_cam",
        width: int = 960,
        height: int = 720,
        pixel_noise: float = 0.0,
        seed: int = 0,
        max_reproj_px: float = 1.5,
        min_marker_px: float = 12.0,  # coarse detection from ~2.5 m; fine alignment happens up close
    ) -> None:
        self.m = model
        self.cam = model.camera(camera).id
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        fy = (height / 2) / math.tan(math.radians(model.cam_fovy[self.cam]) / 2)
        self.K = np.array([[fy, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        s = MARKER_SIZE / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        self.noise = pixel_noise
        self.rng = np.random.default_rng(seed)
        self.max_reproj, self.min_px = max_reproj_px, min_marker_px
        self.last_image: np.ndarray | None = None

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam)
        img = self.renderer.render()
        if self.noise > 0:
            img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        self.last_image = img
        return img

    def observe(self, data: mujoco.MjData) -> PortEstimate:
        t0 = time.perf_counter()
        img = self.capture(data)
        est = self._estimate(data, img)
        est.latency_ms = (time.perf_counter() - t0) * 1e3
        return est

    def _estimate(self, data, img) -> PortEstimate:
        stamp = float(data.time)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or MARKER_ID not in ids.ravel():
            return PortEstimate(False, stamp, reason="inlet marker not visible")
        raw = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2).astype(np.float32)
        if (raw < 2).any() or (raw[:, 0] > self.w - 3).any() or (raw[:, 1] > self.h - 3).any():
            return PortEstimate(False, stamp, corners_px=raw, reason="marker cut off by the image edge")
        edge = float(np.mean(np.linalg.norm(raw - np.roll(raw, 1, axis=0), axis=1)))
        if edge < self.min_px:
            return PortEstimate(False, stamp, corners_px=raw, marker_px=edge, reason="marker too small / too far")
        ref = raw.reshape(4, 1, 2).copy()
        win = max(2, min(5, int(edge / 25)))
        cv2.cornerSubPix(gray, ref, (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        ref = ref.reshape(4, 2)
        if np.max(np.linalg.norm(ref - raw, axis=1)) > 1.5:  # refinement drifted: keep the raw corners
            ref = raw
        pts = ref.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(self.obj, pts, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return PortEstimate(False, stamp, corners_px=pts, reason="pose solver failed")
        proj, _ = cv2.projectPoints(self.obj, rvec, tvec, self.K, None)
        reproj = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - pts, axis=1)))
        if reproj > self.max_reproj:
            return PortEstimate(False, stamp, corners_px=pts, reproj_px=reproj, reason="inconsistent geometry")
        r_cv_m, _ = cv2.Rodrigues(rvec)
        # Camera pose comes from the robot's own joint angles (forward kinematics).
        r_w_c = data.cam_xmat[self.cam].reshape(3, 3)
        p_w_c = data.cam_xpos[self.cam]
        r_w_m = r_w_c @ CV_IN_MJ @ r_cv_m
        p_w_m = p_w_c + r_w_c @ CV_IN_MJ @ tvec.ravel()
        r_w_s = r_w_m @ MARKER_ROT_S.T
        p_w_s = p_w_m - r_w_s @ MARKER_POS_S
        conf = float(np.clip(1.0 - reproj / self.max_reproj, 0, 1) * np.clip(edge / (3 * self.min_px), 0, 1))
        return PortEstimate(True, stamp, conf, p_w_s, r_w_s, reproj, edge, float(np.linalg.norm(tvec)), pts)

    def project_socket(self, data, position, rotation, length=0.06) -> np.ndarray | None:
        """Project the estimated socket centre and insertion axis into the current image (for overlays)."""
        r_w_c = data.cam_xmat[self.cam].reshape(3, 3)
        p_w_c = data.cam_xpos[self.cam]
        pts = np.array([position, position + rotation[:, 2] * length, position + rotation[:, 0] * 0.03])
        cam = (CV_IN_MJ @ r_w_c.T @ (pts - p_w_c).T).T
        if (cam[:, 2] <= 0.01).any():
            return None
        uv = (self.K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]


def evaluate(model, data, est: PortEstimate) -> tuple[float, float]:
    """Scoring only (never used for control): position error (m) and angle error (rad) vs ground truth."""
    sid = model.site("socket_face").id
    pe = float(np.linalg.norm(est.position - data.site_xpos[sid]))
    ae = float(np.linalg.norm(orientation_error(est.rotation, data.site_xmat[sid].reshape(3, 3))))
    return pe, ae


class PoseFilter:
    """Exponential filter on the socket pose with jump detection (the car moved)."""

    def __init__(self, alpha: float = 0.5, jump_m: float = 0.004, jump_rad: float = math.radians(1.5)) -> None:
        self.alpha, self.jump_m, self.jump_rad = alpha, jump_m, jump_rad
        self.pos: np.ndarray | None = None
        self.rot: np.ndarray | None = None
        self.stamp = -1.0

    def update(self, est: PortEstimate) -> bool:
        """Fuse a valid estimate. Returns True when it jumped (target moved) and the filter reset."""
        if self.pos is None:
            self.pos, self.rot, self.stamp = est.position.copy(), est.rotation.copy(), est.timestamp
            return False
        dp = float(np.linalg.norm(est.position - self.pos))
        da = float(np.linalg.norm(orientation_error(self.rot, est.rotation)))
        self.stamp = est.timestamp
        if dp > self.jump_m or da > self.jump_rad:
            self.pos, self.rot = est.position.copy(), est.rotation.copy()
            return True
        a = self.alpha
        self.pos = (1 - a) * self.pos + a * est.position
        # Blend rotations through the rotation vector between them.
        q0, q1 = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q0, self.rot.ravel())
        mujoco.mju_mat2Quat(q1, est.rotation.ravel())
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, q0)
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, q1, neg)
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, dq, 1.0)
        step = np.zeros(4)
        mujoco.mju_axisAngle2Quat(step, v / (np.linalg.norm(v) + 1e-12), a * np.linalg.norm(v))
        qn = np.zeros(4)
        mujoco.mju_mulQuat(qn, step, q0)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, qn)
        self.rot = mat.reshape(3, 3)
        return False
