"""Autonomous charging agent: Observe -> Estimate -> Decide -> Act -> Verify, every control cycle.

* Observe: render the robot camera (10 Hz) and detect the inlet (``perception.py``).
* Estimate: socket pose from PnP, filtered; a jump means the car moved.
* Decide: SEARCHING -> (reachability check) -> TRACKING, or REPOSITIONING (drive the base closer,
  mobile robots only) -> SETTLING (re-observe up close) -> TRACKING; or FAILED "unreachable".
  A supervisor holds on stale perception, re-plans on target jumps, stops on unexpected touch,
  and stops safely when the car moves while the plug is engaged.
* Act: the insertion controller's Cartesian servo on the arm; the drive controller on the base.

The agent never reads the simulator's socket pose. It only uses camera images, the camera's
calibration, its own joint/base state, its own kinematic model, and contact (touch) sensing.
Ground truth is used only by ``evaluation.py`` and test assertions.
"""

from __future__ import annotations

import dataclasses
import enum
import math

import mujoco
import numpy as np

from autocharge.base_controller import DriveController, DriveLimits, DrivePhase, TurnDriveTurn
from autocharge.controller import InsertionController, InsertionLimits, Phase
from autocharge.navigation import Lidar, OccupancyGrid, WaypointDrive, path_length, plan_path
from autocharge.perception import PortEstimate, PortPerception, PoseFilter
from autocharge.scene import ARM_JOINTS, BASE_JOINTS, arm_dof_indices, solve_ik


class AgentState(enum.Enum):
    SEARCHING = "searching"  # no reliable view of the inlet yet: arm holds still
    REPOSITIONING = "repositioning"  # inlet out of arm reach: driving the base closer
    SETTLING = "settling"  # parked: re-observing the inlet up close before deciding again
    TRACKING = "tracking"  # arm controller running on live estimates
    SUCCESS = "success"
    SAFE_STOP = "safe_stop"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    perception_hz: float = 10.0
    search_timeout: float = 6.0  # s without any detection before giving up (arm never moves)
    stale_hold: float = 0.35  # s without a fresh estimate before the approach pauses
    lost_timeout: float = 4.0  # s lost during approach/align before backing off and failing
    min_confidence: float = 0.2
    home_standoff: float = 0.22  # m, design: the ready pose holds the plug this far from the inlet
    reach_tol_m: float = 0.003  # IK residual allowed for "reachable"
    reach_tol_deg: float = 1.0
    scan_speed: float = 0.6  # rad/s base turn rate while scanning for the inlet
    scan_tilts: tuple = (0.0, 0.35, -0.6)  # wrist offsets (rad): level, camera up (~37 deg), down (~-10 deg)
    redrive_m: float = 0.03  # replan the drive if the target estimate moves more than this
    settle_time: float = 0.8  # s of fresh close-up observations after parking
    max_repositions: int = 5
    touch_force: float = 3.0  # N on the arm/connector from anything unexpected -> stop


class AutonomousAgent:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        perception: PortPerception | None = None,
        limits: InsertionLimits | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.m = model
        self.cfg = config or AgentConfig()
        self.perception = perception or PortPerception(model)
        # After a large car move the first insertion attempts can drift (joint stick-slip in an
        # unfamiliar arm pose); the deflection watchdog backs out, so allow a few more retries.
        # Extra servo authority against cable pull after the base has driven: at least 0.4 rad of
        # lead, and up to 12 N m on every joint (the weak kp=10 wrist needs ~1.2 rad for that).
        self.limits = limits or dataclasses.replace(InsertionLimits(), max_retries=4, joint_lead_max=0.4,
                                                    lead_torque_max=12.0)
        self.filter = PoseFilter()
        self.ctrl: InsertionController | None = None
        self.drive: DriveController | None = None
        self.state = AgentState.SEARCHING
        self.t_start = float(data.time)
        self.last_seen = -1e9
        self.last_obs: PortEstimate | None = None
        self.next_obs = float(data.time)
        self.decisions: list[tuple[float, str]] = [(round(float(data.time), 3), "searching: looking for the inlet")]
        self.obs_log: list[dict] = []
        self._lost_logged = False
        self.repositions = 0
        self._settle_start = 0.0
        self.unreachable = False
        self.touched: str | None = None

        # Robot self-knowledge only: arm joints, own kinematic model, optional base joints.
        self.mobile = all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0 for n in BASE_JOINTS)
        self.qadr, _ = arm_dof_indices(model)
        jids = [model.joint(n).id for n in ARM_JOINTS]
        self.arm_act = np.array([InsertionController._actuator_for(model, j) for j in jids])
        self.q_lo, self.q_hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.q_carry = data.qpos[self.qadr].copy()
        self.q_hold = self.q_carry.copy()  # arm hold target (the search tilts the camera via the wrist)
        self.wrist_i = list(ARM_JOINTS).index("joint_wrist")
        self.q_cmd = data.ctrl[self.arm_act].copy()
        self.scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Docking geometry from the robot's own ready pose: the inlet should sit home_standoff in
        # front of the plug. Expressed in the base frame (a design constant, not a car measurement).
        ps = model.site("plug_face").id
        r_p = data.site_xmat[ps].reshape(3, 3)
        socket0 = data.site_xpos[ps] + r_p[:, 2] * self.cfg.home_standoff
        b = model.body("base_link").id
        r_b, p_b = data.xmat[b].reshape(3, 3), data.xpos[b]
        self.p_b_s0 = r_b.T @ (socket0 - p_b)
        self.r_b_s0 = r_b.T @ r_p
        # Touch sensing: which bodies are "me".
        root = model.body("base_link").id
        self.robot_bodies = set()
        for bb in range(model.nbody):
            a = bb
            while a not in (0, root):
                a = model.body_parentid[a]
            if a == root:
                self.robot_bodies.add(bb)
        self.connector = model.body("connector").id
        veh = model.body("vehicle").id
        self.vehicle_bodies = {bb for bb in range(model.nbody) if bb == veh or model.body_parentid[bb] == veh}
        self.floor = model.geom("floor").id
        # Obstacle sensing for driving: planar LiDAR on the base (mobile robots only).
        self.lidar = Lidar(model) if self.mobile else None
        self.grid: OccupancyGrid | None = None
        self.nav_path: list | None = None
        self.lidar_points = np.zeros((0, 2))
        self._next_rescan = 0.0
        self._f6 = np.zeros(6)

    # ----------------------------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.state in (AgentState.SUCCESS, AgentState.SAFE_STOP, AgentState.FAILED)

    @property
    def phase(self) -> str:
        if self.ctrl is not None and self.state == AgentState.TRACKING:
            return self.ctrl.phase.value
        if self.state == AgentState.REPOSITIONING and self.drive is not None:
            return f"drive-{self.drive.phase.value}"
        return self.state.value

    @property
    def events(self) -> list[tuple[float, str]]:
        ev = self.decisions + (self.ctrl.events if self.ctrl else [])
        return sorted(ev, key=lambda e: e[0])

    def _decide(self, data, text: str) -> None:
        self.decisions.append((round(float(data.time), 3), text))

    # ----------------------------------------------------------------------------------------
    def _base_pose(self, data) -> tuple[float, float, float]:
        q = [data.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] for n in BASE_JOINTS]
        return float(q[0]), float(q[1]), math.degrees(float(q[2]))

    def base_target(self, pos, rot) -> tuple[float, float, float]:
        """Where the base should park so the inlet sits at the ready-pose offset (x, y, yaw deg)."""
        r_wb = rot @ self.r_b_s0.T
        yaw = math.atan2(r_wb[1, 0], r_wb[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p = np.asarray(pos) - rz @ self.p_b_s0
        return float(p[0]), float(p[1]), math.degrees(yaw)

    def reachable(self, data, pos, rot, base=None) -> tuple[bool, float, float]:
        """IK on a private copy of the robot's own state: can the plug reach standoff and seated poses?"""
        sd = self.scratch
        sd.qpos[:] = data.qpos
        if base is not None:
            for n, v in zip(BASE_JOINTS, (base[0], base[1], math.radians(base[2])), strict=True):
                sd.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = v
        worst_p = worst_r = 0.0
        q = data.qpos[self.qadr].copy()
        for offset in (self.limits.standoff, 0.0):
            target = np.asarray(pos) - rot[:, 2] * offset
            q, pe, re = solve_ik(self.m, sd, "plug_face", target, rot, q, iters=300)
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
        ok = worst_p < self.cfg.reach_tol_m and math.degrees(worst_r) < self.cfg.reach_tol_deg
        return ok, worst_p, math.degrees(worst_r)

    def _touch(self, data) -> str | None:
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 < 0 or g2 < 0:
                continue  # cable contacts
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue
            mine, other_g, other_b = (b1, g2, b2) if r1 else (b2, g1, b1)
            if other_g == self.floor:
                continue
            if mine == self.connector and other_b in self.vehicle_bodies:
                continue  # plug meeting the inlet: handled by the insertion force limits
            mujoco.mj_contactForce(self.m, data, i, self._f6)
            if abs(self._f6[0]) > self.cfg.touch_force:
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, other_g) or "object"
                return f"{name} ({abs(self._f6[0]):.0f} N)"
        return None

    def _scan(self, data, t: float) -> bool:
        """Active search: turn the base a full 360 deg with the camera level, then tilted up, then
        tilted down (the camera rides on the connector, so the wrist tilts it). False when done."""
        if t - self.t_start < 0.6:  # give the first frames a chance before moving
            return True
        x, y, yaw = self._base_pose(data)
        acts = [self.m.actuator(n).id for n in ("base_x_vel", "base_y_vel", "base_yaw_vel")]
        if not hasattr(self, "_scan_stages"):
            self._scan_hold = (x, y)
            names = {0.0: "camera level", self.cfg.scan_tilts[1]: "camera tilted UP",
                     self.cfg.scan_tilts[2]: "camera tilted DOWN"}
            self._scan_stages = []
            for i, tilt in enumerate(self.cfg.scan_tilts):
                self._scan_stages += [("tilt", tilt, f"search {i + 1}/{len(self.cfg.scan_tilts)}: {names.get(tilt, '')}, "
                                                      "turning 360 deg"), ("turn", 360.0, "")]
            self._scan_stages.append(("tilt", 0.0, ""))
            self._stage_start = None
        if not self._scan_stages:
            data.ctrl[acts] = 0.0
            self.q_hold = self.q_carry.copy()
            return False
        kind, value, note = self._scan_stages[0]
        if self._stage_start is None:
            self._stage_start = (t, yaw)
            if note:
                self._decide(data, f"searching: inlet not in view - {note}")
            if kind == "tilt":
                self.q_hold = self.q_carry.copy()
                self.q_hold[self.wrist_i] = np.clip(self.q_carry[self.wrist_i] + value, self.q_lo[self.wrist_i],
                                                    self.q_hi[self.wrist_i])
        hx, hy = self._scan_hold
        w = 0.0
        if kind == "tilt":
            settled = abs(data.qpos[self.qadr[self.wrist_i]] - self.q_hold[self.wrist_i]) < 0.03
            if settled or t - self._stage_start[0] > 2.0:
                self._scan_stages.pop(0)
                self._stage_start = None
        else:
            target = self._stage_start[1] + value
            err = math.radians(target - yaw)
            w = float(np.clip(1.5 * err, -self.cfg.scan_speed, self.cfg.scan_speed))
            if abs(err) < math.radians(3.0):
                self._scan_stages.pop(0)
                self._stage_start = None
        data.ctrl[acts] = np.clip([20 * (hx - x), 20 * (hy - y), w], -1, 1)
        return True

    def _brake(self, data) -> None:
        """Stop any scanning motion and hold the base where it is."""
        if not self.mobile or self.drive is not None:
            return
        x, y, yaw = self._base_pose(data)
        self.drive = DriveController(self.m, data, dock=(x, y, yaw))
        self.drive.hold_pose = (x, y, math.radians(yaw))
        self.drive.phase = DrivePhase.DOCKED

    def _hold_arm(self, data) -> None:
        err = self.q_hold - data.qpos[self.qadr]
        self.q_cmd = np.clip(self.q_cmd + 3.0 * err * self.m.opt.timestep, self.q_lo, self.q_hi)
        data.ctrl[self.arm_act] = self.q_cmd

    # ----------------------------------------------------------------------------------------
    def step(self, data: mujoco.MjData) -> None:
        t = float(data.time)
        if t + 1e-9 >= self.next_obs:
            self.next_obs += 1.0 / self.cfg.perception_hz
            self._observe(data)
        if self.drive is not None:
            self.drive.step(data)  # drives, or keeps the brakes on once parked
        if self.done:
            if self.ctrl is not None:
                self.ctrl.step(data)
            elif self.state != AgentState.SUCCESS:
                self._hold_arm(data)
            return

        touch = self._touch(data)
        if touch:
            self.touched = touch
            if self.ctrl is not None:
                self.ctrl.safe_stop(data, f"unexpected contact on the arm: {touch}")
                self.ctrl.step(data)
            self.state = AgentState.SAFE_STOP
            self._decide(data, f"safe stop: unexpected contact on the arm with {touch}")
            return

        if self.state == AgentState.SEARCHING:
            self._hold_arm(data)
            if self.mobile:
                if self._scan(data, t):
                    return
                self.state = AgentState.FAILED
                self._brake(data)
                self._decide(data, "failed: searched 360 deg at three camera tilts, inlet never seen; "
                                   "not driving blind")
            elif t - self.t_start > self.cfg.search_timeout:
                self.state = AgentState.FAILED
                self._decide(data, "failed: inlet never seen clearly; not moving the arm")
            return
        if self.state == AgentState.REPOSITIONING:
            self._hold_arm(data)
            self._check_route(data)
            if self.drive is None:
                return
            if self.drive.phase == DrivePhase.DOCKED:
                self.state = AgentState.SETTLING
                self._settle_start = t
                self.filter = PoseFilter()  # fresh close-up estimates only
                self._decide(data, "parked: re-observing the inlet up close")
            elif self.drive.phase == DrivePhase.STOPPED:
                self.state = AgentState.SAFE_STOP
                self._decide(data, f"safe stop while driving: {self.drive.events[-1][1]}")
            return
        if self.state == AgentState.SETTLING:
            self._hold_arm(data)
            if t - self._settle_start > 2.5 and self.last_seen <= self._settle_start:
                # Parked but the inlet is not in view: look around again (scan), then re-plan.
                self.state = AgentState.SEARCHING
                self.t_start = t - 0.6
                self.drive = None
                if hasattr(self, "_scan_stages"):
                    del self._scan_stages
                self._decide(data, "parked but inlet not in view: scanning again")
                return
            if t - self._settle_start > self.cfg.settle_time and self.filter.pos is not None and self.last_seen > self._settle_start:
                self._plan(data)
            return

        c = self.ctrl
        age = t - self.last_seen
        if c.phase in (Phase.APPROACH, Phase.ALIGN, Phase.RETRACT):
            stale = age > self.cfg.stale_hold
            if stale and not c.hold:
                self._decide(data, f"hold: no fresh view of the inlet for {age:.2f} s, pausing approach")
            elif not stale and c.hold:
                self._decide(data, "resume: inlet visible again")
            c.hold = stale
            if age > self.cfg.lost_timeout and c.phase != Phase.RETRACT:
                c.safe_stop(data, f"inlet lost for {age:.1f} s: stopping before contact")
        else:
            c.hold = False  # final insertion is guided by the last estimate and force sensing
        c.step(data)

        if c.phase == Phase.SEATED:
            self.state = AgentState.SUCCESS
            self._decide(data, "success: connector seated")
        elif c.phase in (Phase.SAFE_STOP, Phase.ESTOP):
            self.state = AgentState.SAFE_STOP
            self._decide(data, f"safe stop: {c.events[-1][1].split(': ', 1)[-1]}")
        elif c.phase == Phase.ABORTED:
            self.state = AgentState.FAILED
            self._decide(data, "failed: insertion retries exhausted, holding at standoff")

    # ----------------------------------------------------------------------------------------
    def _plan(self, data) -> None:
        """Reachability decision from the current estimate: insert, drive closer, or refuse."""
        pos, rot = self.filter.pos, self.filter.rot
        self.q_hold = self.q_carry.copy()
        if self.mobile and hasattr(self, "_scan_stages"):
            del self._scan_stages
            if self.drive is None:
                data.ctrl[[self.m.actuator(n).id for n in ("base_x_vel", "base_y_vel", "base_yaw_vel")]] = 0.0
        ok, pe, re = self.reachable(data, pos, rot)
        if ok and self.mobile and self.repositions < self.cfg.max_repositions:
            # Reachable, but is the base where it should be? Re-park if it is well off the ideal
            # pose (e.g. it parked using a coarse estimate taken from far away).
            tgt, cur = self.base_target(pos, rot), self._base_pose(data)
            off = math.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
            dyaw = abs((tgt[2] - cur[2] + 180) % 360 - 180)
            if off > 0.08 or dyaw > 6.0:
                ok = False
                pe = off
        if ok:
            self._brake(data)
            self.ctrl = InsertionController(self.m, data, limits=self.limits, socket_estimate=(pos, rot))
            self.ctrl.q_ref = self.q_cmd.copy()
            self.ctrl.q_cmd = self.q_cmd.copy()
            self.state = AgentState.TRACKING
            self._decide(data, "inlet within arm reach: planning arm approach")
            return
        if self.mobile and self.repositions < self.cfg.max_repositions:
            tgt = self.base_target(pos, rot)
            cur = self._base_pose(data)
            dist = math.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
            ok_there, pe2, re2 = self.reachable(data, pos, rot, base=tgt)
            if ok_there:
                self.repositions += 1
                self._decide(data, f"inlet out of arm reach (IK miss {pe * 1e3:.0f} mm): need to drive base "
                                   f"{dist:.2f} m to x={tgt[0]:.2f} y={tgt[1]:.2f} yaw={tgt[2]:.1f} deg")
                if self._start_drive(data, tgt):
                    self.state = AgentState.REPOSITIONING
                    return
                why = "no collision-free route for the base (LiDAR occupancy grid)"
            else:
                why = f"not reachable even after driving (IK miss {pe2 * 1e3:.0f} mm)"
        else:
            why = f"outside arm reach (IK miss {pe * 1e3:.0f} mm, {re:.1f} deg)"
        self.unreachable = True
        self._brake(data)
        self.state = AgentState.FAILED
        self._decide(data, f"failed: inlet unreachable, {why}; not moving")

    def _start_drive(self, data, tgt) -> bool:
        """LiDAR scan -> occupancy grid -> A* route to the parking pose. False if no route exists."""
        cur = self._base_pose(data)
        pts, _, _ = self.lidar.scan(data)
        self.lidar_points = pts
        dist = math.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
        self.grid = OccupancyGrid(center=((cur[0] + tgt[0]) / 2, (cur[1] + tgt[1]) / 2), size=max(6.0, 2 * dist + 3.0))
        self.grid.add_points(pts)
        path = plan_path(self.grid, cur[:2], tgt[:2])
        if path is None:
            self._decide(data, f"LiDAR: {len(pts)} returns, route to the parking pose is blocked")
            return False
        self.nav_path = path
        direct = len(path) == 2
        if direct:
            self.drive = TurnDriveTurn(self.m, data, dock=tgt, limits=DriveLimits(timeout=60.0))
            self._decide(data, f"LiDAR: direct route clear ({len(pts)} returns), driving {dist:.2f} m")
        else:
            self.drive = WaypointDrive(self.m, data, path, tgt[2])
            self._decide(data, f"LiDAR: obstacle on the direct route, detour {path_length(path):.2f} m via "
                               f"{len(path) - 2} waypoint(s) (direct {dist:.2f} m)")
        self._next_rescan = float(data.time) + 0.5
        return True

    def _check_route(self, data) -> None:
        """While driving: re-scan, add to the map, re-plan if the remaining route became blocked."""
        if self.grid is None or self.nav_path is None or data.time < self._next_rescan:
            return
        self._next_rescan = float(data.time) + 0.5
        pts, _, _ = self.lidar.scan(data)
        self.lidar_points = pts
        before = self.grid.occ.copy()
        self.grid.add_points(pts)
        if not (self.grid.occ & ~before).any():
            return  # nothing new in the map: keep following the current route
        cur = np.array(self._base_pose(data)[:2])
        ahead = [cur] + [w for w in getattr(self.drive, "waypoints", [self.drive.dock])]
        saved = self.grid.blocked.copy()
        self.grid.blocked = self.grid.mask(0.30)  # bare footprint: tolerate small path-following drift
        si, sj = self.grid.cell(cur)
        rr = int(0.3 / self.grid.res)
        self.grid.blocked[max(0, si - rr):si + rr + 1, max(0, sj - rr):sj + rr + 1] = False
        clear = all(self.grid.free_line(a, b) for a, b in zip(ahead[:-1], ahead[1:], strict=True))
        self.grid.blocked = saved
        if clear:
            return
        tgt_xy = ahead[-1]
        yaw = getattr(self.drive, "final_yaw", None)
        if yaw is None:
            yaw = math.degrees(self.drive.dock_yaw)
        path = plan_path(self.grid, cur, tgt_xy)
        if path is None:
            self.drive = None
            self._brake(data)
            self.state = AgentState.SAFE_STOP
            self._decide(data, "safe stop: LiDAR shows the route is now blocked and there is no way round")
            return
        self.nav_path = path
        self.drive = WaypointDrive(self.m, data, path, yaw) if len(path) > 2 else \
            TurnDriveTurn(self.m, data, dock=(float(tgt_xy[0]), float(tgt_xy[1]), yaw), limits=DriveLimits(timeout=60.0))
        self._decide(data, f"LiDAR: new obstacle ahead, re-planned route ({path_length(path):.2f} m)")

    def _observe(self, data) -> None:
        est = self.perception.observe(data)
        self.last_obs = est
        rec = {"t": round(est.timestamp, 3), "detected": est.detected, "conf": round(est.confidence, 3),
               "reproj": None if math.isnan(est.reproj_px) else round(est.reproj_px, 3),
               "marker_px": round(est.marker_px, 1), "latency_ms": round(est.latency_ms, 1), "reason": est.reason,
               "state": self.phase}
        if est.detected and est.confidence >= self.cfg.min_confidence:
            jumped = self.filter.update(est)
            self.last_seen = est.timestamp
            self._lost_logged = False
            rec["pos"] = [round(float(v), 4) for v in est.position]
            if self.state == AgentState.SEARCHING:
                self._decide(data, f"inlet detected {est.range_m:.2f} m away (confidence {est.confidence:.2f}, "
                                   f"reprojection {est.reproj_px:.2f} px)")
                self._plan(data)
            elif self.state == AgentState.REPOSITIONING and self.drive is not None:
                tgt = self.base_target(self.filter.pos, self.filter.rot)
                d = self.drive
                moved = math.hypot(tgt[0] - d.dock[0], tgt[1] - d.dock[1])
                recent = data.time - getattr(self, "_last_redrive", -1e9) < 1.0
                if moved > self.cfg.redrive_m and d.phase in (DrivePhase.DRIVE, DrivePhase.DOCK) and not recent:
                    self._last_redrive = float(data.time)
                    d.retarget(data, tgt)
                    self._decide(data, f"target estimate moved {moved * 100:.0f} cm: re-planning the drive")
            elif self.state == AgentState.TRACKING and self.ctrl is not None:
                if jumped:
                    self._decide(data, "perception: inlet pose jumped, the car moved")
                    if not self.ctrl.engaged(data):
                        ok, pe, _ = self.reachable(data, self.filter.pos, self.filter.rot)
                        if not ok:
                            self.ctrl.safe_stop(data, f"car moved out of arm reach (IK miss {pe * 1e3:.0f} mm)")
                            return
                self.ctrl.retarget(data, self.filter.pos, self.filter.rot, jumped)
        elif not self._lost_logged and self.state not in (AgentState.SEARCHING,):
            self._lost_logged = True
            self._decide(data, f"perception: no reliable inlet view ({est.reason or 'low confidence'})")
        self.obs_log.append(rec)


def socket_estimate_error(model, data, agent: AutonomousAgent) -> tuple[float, float] | None:
    """Evaluation helper: filtered estimate vs ground truth (mm, deg). Not used for control."""
    if agent.filter.pos is None:
        return None
    from autocharge.scene import orientation_error

    sid = model.site("socket_face").id
    pe = float(np.linalg.norm(agent.filter.pos - data.site_xpos[sid])) * 1e3
    ae = math.degrees(float(np.linalg.norm(orientation_error(agent.filter.rot, data.site_xmat[sid].reshape(3, 3)))))
    return pe, ae
