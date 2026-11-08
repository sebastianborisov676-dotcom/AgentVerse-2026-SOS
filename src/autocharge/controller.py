"""Slow, alignment-checked insertion with contact-force and cable-tension limits.

The Google Robot arm is driven through its *original* position actuators. Each control step
the controller computes the plug-face pose error against a moving Cartesian reference and
integrates a damped-least-squares joint update into the position set-points. The integral
form removes steady-state sag caused by servo compliance and the cable's weight.

State machine::

    APPROACH -> ALIGN -> INSERT -> SEATED
                  ^         |
                  +-RETRACT-+   (limit exceeded or alignment lost; retries)
    any -> ABORTED (retries exhausted / timeout)   any -> ESTOP (operator)
"""

from __future__ import annotations

import dataclasses
import enum
import math

import mujoco
import numpy as np

from autocharge.metrics import (
    Alignment,
    cable_metrics,
    connector_contact_force,
    plug_alignment_error,
)
from autocharge.scene import ARM_JOINTS, PIN_LENGTH, arm_dof_indices, orientation_error


class Phase(enum.Enum):
    APPROACH = "approach"
    ALIGN = "align"
    INSERT = "insert"
    SEATED = "seated"
    RETRACT = "retract"
    ABORTED = "aborted"
    ESTOP = "estop"
    SAFE_STOP = "safe_stop"  # supervisor hold: no further motion or force (e.g. car moved while engaged)


@dataclasses.dataclass(frozen=True)
class InsertionLimits:
    standoff: float = 0.06  # m in front of the socket face where alignment is verified
    approach_speed: float = 0.05  # m/s
    insert_speed: float = 0.005  # m/s (slow insertion)
    retract_speed: float = 0.01  # m/s
    align_lateral_tol: float = 0.0010  # m, required before insertion may start
    align_angle_tol: float = math.radians(1.0)
    align_axial_tol: float = 0.003  # m, distance from the standoff point
    align_hold_time: float = 0.25  # s the tolerances must hold continuously
    align_timeout: float = 6.0  # s per alignment attempt
    insert_lateral_max: float = 0.0025  # m, abort insertion beyond this (outside the socket)
    insert_angle_max: float = math.radians(2.5)
    contact_force_soft: float = 30.0  # N, pause advancing above this
    contact_force_max: float = 70.0  # N, retract above this
    soft_pause_timeout: float = 1.5  # s paused at the soft force limit before retracting
    cable_tension_max: float = 60.0  # N, max axial tension anywhere in the cable
    strain_relief_max: float = 45.0  # N, max load the cable puts on the connector boot
    seat_depth: float = PIN_LENGTH - 0.002  # m of pin engagement that counts as seated
    max_retries: int = 2
    force_filter_tau: float = 0.03  # s, first-order low-pass modelling a wrist F/T sensor
    joint_lead_max: float = 0.15  # rad, anti-windup bound |q_ref - q|
    lead_torque_max: float | None = None  # N m; if set, per-joint bound = max(joint_lead_max, this / kp)
    align_max_speed: float = 0.003  # m/s, plug must be settled before insertion starts
    max_replans: int = 6  # target updates that force a re-approach (car moved), before giving up


@dataclasses.dataclass
class Telemetry:
    time: float
    phase: str
    lateral_mm: float  # true (ground-truth) plug/socket errors
    axial_mm: float
    angle_deg: float
    contact_force: float  # filtered, N
    axial_contact_force: float
    cable_max_tension: float
    strain_relief_load: float


class InsertionController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        limits: InsertionLimits | None = None,
        socket_estimate: tuple[np.ndarray, np.ndarray] | None = None,
        gain_pos: float = 8.0,
        gain_rot: float = 8.0,
        lead_gain: float = 1.5,
        posture_gain: float = 0.05,
        friction_comp: float = 1.0,
        fric_eps: float = 0.002,
    ) -> None:
        self.m, self.limits = model, limits or InsertionLimits()
        self.gain_pos, self.gain_rot = gain_pos, gain_rot
        self.lead_gain, self.posture_gain = lead_gain, posture_gain
        self.qadr, self.dadr = arm_dof_indices(model)
        jids = [model.joint(n).id for n in ARM_JOINTS]
        self.q_lo, self.q_hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.act = np.array([self._actuator_for(model, j) for j in jids])
        kp = model.actuator_gainprm[self.act, 0]
        self.fric_offset = friction_comp * model.dof_frictionloss[self.dadr] / kp
        self.lead = np.full(len(kp), self.limits.joint_lead_max)
        if self.limits.lead_torque_max is not None:
            self.lead = np.maximum(self.lead, self.limits.lead_torque_max / kp)
        self.fric_eps = fric_eps
        self.plug = model.site("plug_face").id
        self.sock = model.site("socket_face").id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.q_ref = data.qpos[self.qadr].copy()
        self.q_cmd = self.q_ref.copy()
        self.q_home = self.q_ref.copy()
        self.phase = Phase.APPROACH
        self.phase_start = data.time
        self.retries = 0
        self.events: list[tuple[float, str]] = []
        self.telemetry: list[Telemetry] = []
        self._aligned_since: float | None = None
        self._paused_since: float | None = None
        self.peak = {
            "contact_force": 0.0,  # filtered (what the safety logic acts on)
            "contact_force_raw": 0.0,  # unfiltered per-step solver value
            "cable_tension": 0.0,
            "strain_relief": 0.0,
        }
        self.force_filt = 0.0
        self.hold = False  # set by the autonomy layer while perception is unreliable
        self.replans = 0
        self._last_jump = -1e9
        self._vel = np.zeros(6)

        mujoco.mj_forward(model, data)
        # Socket pose the controller acts on. Ground truth by default; pass an estimate (e.g. from
        # camera perception) to make every alignment decision use it instead.
        if socket_estimate is None:
            socket_estimate = (data.site_xpos[self.sock], data.site_xmat[self.sock].reshape(3, 3))
        self.socket_pos = np.array(socket_estimate[0], dtype=float).copy()
        self.rot_d = np.array(socket_estimate[1], dtype=float).reshape(3, 3).copy()
        self.axis = self.rot_d[:, 2].copy()
        # Reference axial coordinate (plug face along socket axis); starts at the current pose.
        self.ref_pos = data.site_xpos[self.plug].copy()
        self._log(data, "start: approach")

    @staticmethod
    def _actuator_for(model, jid) -> int:
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == jid:
                return a
        raise ValueError(f"no actuator for joint {jid}")

    # ----------------------------------------------------------------------------------------
    def emergency_stop(self, data: mujoco.MjData) -> None:
        # Freeze the last commanded set-points (they already contain the integral term holding
        # the cable's weight); resetting them to the measured joints would let the arm sag.
        self._set_phase(data, Phase.ESTOP, "operator emergency stop")

    @property
    def done(self) -> bool:
        return self.phase in (Phase.SEATED, Phase.ABORTED, Phase.ESTOP, Phase.SAFE_STOP)

    def _log(self, data, msg: str) -> None:
        self.events.append((round(float(data.time), 3), msg))

    def _set_phase(self, data, phase: Phase, why: str) -> None:
        self.phase = phase
        self.phase_start = data.time
        self._aligned_since = None
        self._paused_since = None
        self._log(data, f"{phase.value}: {why}")

    def engaged(self, data) -> bool:
        """Pins are inside the socket (estimated): from here the robot must not chase the car."""
        # 8 mm of pin depth: past the 4 mm chamfer lead-in, so chamfer deflection still backs out.
        return self._estimated_alignment(data).axial > -PIN_LENGTH + 0.008

    def safe_stop(self, data, why: str) -> None:
        if self.phase != Phase.SAFE_STOP:
            self._set_phase(data, Phase.SAFE_STOP, why)

    def retarget(self, data, position, rotation, jumped: bool) -> None:
        """New socket estimate from perception (visual servoing input)."""
        if self.phase in (Phase.ABORTED, Phase.ESTOP, Phase.SAFE_STOP):
            return
        moved = float(np.linalg.norm(np.asarray(position) - self.socket_pos)) * 1e3
        if self.phase in (Phase.SEATED, Phase.INSERT) and self.engaged(data):
            if jumped:
                self.safe_stop(data, f"car moved {moved:.1f} mm while plug engaged: holding, no further force")
            return  # once engaged the target is frozen
        self.socket_pos = np.array(position, dtype=float)
        self.rot_d = np.array(rotation, dtype=float).reshape(3, 3)
        self.axis = self.rot_d[:, 2].copy()
        if not jumped:
            if self.phase == Phase.ALIGN:
                self.ref_pos = self._standoff_point()  # small correction: keep servoing onto the axis
            return
        if data.time - self._last_jump > 1.0:  # one continuous car movement counts as one re-plan
            self.replans += 1
        self._last_jump = float(data.time)
        if self.replans > self.limits.max_replans:
            self.safe_stop(data, "target keeps moving: giving up")
        elif self.phase == Phase.INSERT:
            self.ref_pos = data.site_xpos[self.plug].copy()
            self._set_phase(data, Phase.RETRACT, f"car moved {moved:.1f} mm before engagement: backing out to re-align")
        elif self.phase == Phase.ALIGN:
            self._set_phase(data, Phase.APPROACH, f"car moved {moved:.1f} mm: re-planning approach")
        else:
            self._log(data, f"target updated ({moved:.1f} mm): re-planning")

    def _standoff_point(self) -> np.ndarray:
        return self.socket_pos - self.axis * self.limits.standoff

    def _move_ref_towards(self, goal: np.ndarray, speed: float, dt: float) -> bool:
        delta = goal - self.ref_pos
        dist = float(np.linalg.norm(delta))
        step = speed * dt
        if dist <= step:
            self.ref_pos = goal.copy()
            return True
        self.ref_pos = self.ref_pos + delta / dist * step
        return False

    # ----------------------------------------------------------------------------------------
    def step(self, data: mujoco.MjData) -> None:
        """Update phase logic and write actuator set-points. Call once before each mj_step."""
        m, lim, dt = self.m, self.limits, self.m.opt.timestep
        al = self._estimated_alignment(data)
        true_al = plug_alignment_error(m, data)
        cf = connector_contact_force(m, data)
        cab = cable_metrics(m, data)
        alpha = dt / (lim.force_filter_tau + dt)
        self.force_filt += alpha * (cf.total_normal - self.force_filt)
        force = self.force_filt
        self.peak["contact_force"] = max(self.peak["contact_force"], force)
        self.peak["contact_force_raw"] = max(self.peak["contact_force_raw"], cf.total_normal)
        self.peak["cable_tension"] = max(self.peak["cable_tension"], cab.max_tension)
        self.peak["strain_relief"] = max(self.peak["strain_relief"], cab.strain_relief_load)

        # Hard safety limits apply in every active phase.
        if self.phase in (Phase.APPROACH, Phase.ALIGN, Phase.INSERT, Phase.SEATED):
            if force > lim.contact_force_max and self.engaged(data):
                self.safe_stop(data, f"contact force {force:.1f} N while engaged: holding (not chasing)")
            elif self.phase == Phase.SEATED:
                pass
            elif force > lim.contact_force_max:
                self._start_retract(data, f"contact force {force:.1f} N > {lim.contact_force_max} N")
            elif cab.max_tension > lim.cable_tension_max:
                self._start_retract(data, f"cable tension {cab.max_tension:.1f} N > {lim.cable_tension_max} N")
            elif cab.strain_relief_load > lim.strain_relief_max:
                self._start_retract(
                    data, f"strain relief load {cab.strain_relief_load:.1f} N > {lim.strain_relief_max} N"
                )

        if self.hold and self.phase in (Phase.APPROACH, Phase.ALIGN):
            self._aligned_since = None  # perception unreliable: hold position, no progress
        elif self.phase == Phase.APPROACH:
            if self._move_ref_towards(self._standoff_point(), lim.approach_speed, dt):
                self._set_phase(data, Phase.ALIGN, "reference reached standoff")

        elif self.phase == Phase.ALIGN:
            mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_SITE, self.plug, self._vel, 0)
            ok = (
                float(np.linalg.norm(self._vel[3:])) < lim.align_max_speed
                and al.lateral < lim.align_lateral_tol
                and al.angle < lim.align_angle_tol
                and abs(al.axial + lim.standoff) < lim.align_axial_tol
                and cf.count == 0
            )
            if ok:
                self._aligned_since = self._aligned_since if self._aligned_since is not None else data.time
                if data.time - self._aligned_since >= lim.align_hold_time:
                    self._set_phase(
                        data,
                        Phase.INSERT,
                        f"aligned: lateral {al.lateral * 1e3:.2f} mm, angle {math.degrees(al.angle):.2f} deg",
                    )
            else:
                self._aligned_since = None
            if self.phase == Phase.ALIGN and data.time - self.phase_start > lim.align_timeout:
                self._fail_attempt(data, "alignment timeout")

        elif self.phase == Phase.INSERT:
            if al.axial >= -(PIN_LENGTH - lim.seat_depth):
                self._set_phase(data, Phase.SEATED, f"seated, engagement {(al.axial + PIN_LENGTH) * 1e3:.1f} mm")
            elif (al.lateral > lim.insert_lateral_max or al.angle > lim.insert_angle_max) and self.engaged(data):
                self.safe_stop(
                    data,
                    f"plug pushed off axis while engaged (lateral {al.lateral * 1e3:.2f} mm): holding, no further force",
                )
            elif al.lateral > lim.insert_lateral_max or al.angle > lim.insert_angle_max:
                self._start_retract(
                    data,
                    f"alignment lost: lateral {al.lateral * 1e3:.2f} mm, angle {math.degrees(al.angle):.2f} deg",
                )
            elif force > lim.contact_force_soft:
                self._paused_since = self._paused_since if self._paused_since is not None else data.time
                if data.time - self._paused_since > lim.soft_pause_timeout:
                    self._start_retract(data, f"stuck at {force:.1f} N (jam)")
            else:
                self._paused_since = None
                goal = self.socket_pos - self.axis * 0.0005
                self._move_ref_towards(goal, lim.insert_speed, dt)

        elif self.phase == Phase.RETRACT:
            if self._move_ref_towards(self._standoff_point(), lim.retract_speed, dt):
                if self.retries > lim.max_retries:
                    self._set_phase(data, Phase.ABORTED, "retries exhausted; holding at standoff")
                else:
                    self._set_phase(data, Phase.ALIGN, f"retry {self.retries}")

        elif self.phase == Phase.SEATED:
            # Hold the plug in place with gentle insertion bias, no further advance.
            pass

        if self.phase not in (Phase.ESTOP, Phase.SAFE_STOP):
            self._track(data)
        data.ctrl[self.act] = self.q_cmd

        if not self.telemetry or data.time - self.telemetry[-1].time >= 0.02 - 1e-9:
            self.telemetry.append(
                Telemetry(
                    float(data.time),
                    self.phase.value,
                    true_al.lateral * 1e3,
                    true_al.axial * 1e3,
                    math.degrees(true_al.angle),
                    force,
                    cf.axial_resistance,
                    cab.max_tension,
                    cab.strain_relief_load,
                )
            )

    def _estimated_alignment(self, data) -> Alignment:
        d = self.rot_d.T @ (data.site_xpos[self.plug] - self.socket_pos)
        ang = orientation_error(data.site_xmat[self.plug].reshape(3, 3), self.rot_d)
        return Alignment(float(np.hypot(d[0], d[1])), float(d[2]), float(np.linalg.norm(ang)))

    def _start_retract(self, data, why: str) -> None:
        self.retries += 1
        # Never advance further: reference jumps back to the current plug position.
        self.ref_pos = data.site_xpos[self.plug].copy()
        self._set_phase(data, Phase.RETRACT, why)

    def _fail_attempt(self, data, why: str) -> None:
        self.retries += 1
        if self.retries > self.limits.max_retries:
            self._set_phase(data, Phase.ABORTED, why)
        else:
            self._log(data, f"{why}; retry {self.retries}")
            self.phase_start = data.time

    def _track(self, data) -> None:
        m = self.m
        ep = self.ref_pos - data.site_xpos[self.plug]
        er = orientation_error(data.site_xmat[self.plug].reshape(3, 3), self.rot_d)
        mujoco.mj_jacSite(m, data, self.jacp, self.jacr, self.plug)
        jac = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
        lam = 1e-4
        jinv = jac.T @ np.linalg.inv(jac @ jac.T + lam * np.eye(6))
        dq_err = jinv @ np.concatenate([ep, er])  # joint-space image of the pose error
        # Integral part (removes steady-state sag from servo compliance and cable weight).
        dq = jinv @ np.concatenate([self.gain_pos * ep, self.gain_rot * er])
        # Weak posture term in the null space keeps the redundant DOF near the home pose.
        null = np.eye(len(dq)) - jinv @ jac
        dq += null @ (self.posture_gain * (self.q_home - self.q_ref))
        q = data.qpos[self.qadr]
        self.q_ref = self.q_ref + dq * m.opt.timestep
        self.q_ref = np.clip(self.q_ref, q - self.lead, q + self.lead)
        self.q_ref = np.clip(self.q_ref, self.q_lo, self.q_hi)
        # Proportional lead plus model-based Coulomb friction compensation: the original servos
        # need |ctrl - q| > frictionloss / kp before a joint moves at all, which otherwise causes
        # stick-slip hunting around the target.
        fc = self.fric_offset * np.tanh(dq_err / self.fric_eps)
        self.q_cmd = np.clip(self.q_ref + self.lead_gain * dq_err + fc, self.q_lo, self.q_hi)
