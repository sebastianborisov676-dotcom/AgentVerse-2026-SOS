"""Full charging mission: drive the base in from a distance, dock, then align and insert."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from autocharge.base_controller import DriveController, DriveLimits, DrivePhase
from autocharge.controller import InsertionController, InsertionLimits
from autocharge.scene import ARM_JOINTS, SceneParams, arm_dof_indices

# Drive-in scenario: the robot starts 3 m from its docking pose, 0.5 m to the side and turned
# 15 degrees, with a 4.5 m cable already plugged into the charger.
DRIVE_IN_PARAMS = SceneParams(mobile_base=True, base_start=(-3.0, 0.5, 15.0), cable_segments=64)

# A long cable dragging on the floor loads the connector more (~13 N vs ~5 N when parked next to
# the charger), so the arm needs more servo authority than the fixed-base default.
MISSION_INSERTION_LIMITS = dataclasses.replace(InsertionLimits(), joint_lead_max=0.4)


class ChargingMission:
    """Sequence the base drive and the arm insertion. Call ``step`` before every ``mj_step``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        drive_limits: DriveLimits | None = None,
        insertion_limits: InsertionLimits | None = None,
        socket_estimate=None,
        hold_gain: float = 3.0,
        obstacles=None,
    ) -> None:
        self.m = model
        self.drive = DriveController(model, data, limits=drive_limits, obstacles=obstacles)
        self.insertion: InsertionController | None = None
        self._ins_limits = insertion_limits or MISSION_INSERTION_LIMITS
        self._socket_estimate = socket_estimate
        # Arm carry pose: held with an integral joint-space loop while the base drives, so the
        # cable's drag does not push the connector around.
        self.qadr, _ = arm_dof_indices(model)
        jids = [model.joint(n).id for n in ARM_JOINTS]
        self.arm_act = np.array([InsertionController._actuator_for(model, j) for j in jids])
        self.q_carry = data.qpos[self.qadr].copy()
        self.q_cmd = data.ctrl[self.arm_act].copy()
        self.q_lo, self.q_hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.hold_gain = hold_gain

    @property
    def phase(self) -> str:
        if self.insertion is not None:
            return self.insertion.phase.value
        return self.drive.phase.value

    @property
    def done(self) -> bool:
        if self.drive.phase == DrivePhase.STOPPED:
            return True
        return self.insertion is not None and self.insertion.done

    @property
    def events(self) -> list[tuple[float, str]]:
        return self.drive.events + (self.insertion.events if self.insertion else [])

    def step(self, data: mujoco.MjData) -> None:
        self.drive.step(data)  # keeps the brakes applied after docking
        if self.insertion is None:
            err = self.q_carry - data.qpos[self.qadr]
            self.q_cmd = np.clip(self.q_cmd + self.hold_gain * err * self.m.opt.timestep, self.q_lo, self.q_hi)
            data.ctrl[self.arm_act] = self.q_cmd
        if self.drive.phase == DrivePhase.DOCKED and self.insertion is None:
            # The arm starts its own alignment only once the base is parked.
            self.insertion = InsertionController(
                self.m, data, limits=self._ins_limits, socket_estimate=self._socket_estimate
            )
            self.insertion.q_ref = self.q_cmd.copy()
            self.insertion.q_cmd = self.q_cmd.copy()
        if self.insertion is not None:
            self.insertion.step(data)
