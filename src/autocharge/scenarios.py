"""Scenario manager: changes the world independently of the robot.

The robot is never told what the scenario is. The manager only writes the simulator state
(the car's mocap pose and the visibility of the inlet marker); the agent has to notice the change
through its camera or force sensing.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True)
class CarPose:
    dx: float = 0.0  # m, along the socket axis (away from the robot is +)
    dy: float = 0.0  # m, sideways
    dz: float = 0.0  # m, inlet height change
    dyaw_deg: float = 0.0  # parking angle, about a vertical axis through the nominal inlet

    @staticmethod
    def random(rng: np.random.Generator) -> CarPose:
        return CarPose(
            dx=float(rng.uniform(-0.03, 0.03)),
            dy=float(rng.uniform(-0.04, 0.04)),
            dz=float(rng.uniform(-0.03, 0.03)),
            dyaw_deg=float(rng.uniform(-4.0, 4.0)),
        )


@dataclasses.dataclass(frozen=True)
class CarMove:
    at: float  # s, simulation time the move starts
    to: CarPose  # absolute pose offset after the move
    duration: float = 0.8  # s, smooth roll


@dataclasses.dataclass(frozen=True)
class Occlusion:
    start: float  # s
    end: float  # s (use a large number for "until the end")


class ScenarioManager:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        park: CarPose | None = None,
        moves: tuple[CarMove, ...] = (),
        occlusions: tuple[Occlusion, ...] = (),
        trigger_moves_on: str | None = None,
        light_scale: float = 1.0,
        flap_closed: bool = False,
    ) -> None:
        self.m = model
        self._light0 = (model.light_diffuse.copy(), model.light_ambient.copy(),
                        model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
        self.set_lighting(light_scale)
        self.set_flap(flap_closed)
        self.mocap = model.body_mocapid[model.body("vehicle").id]
        self.marker = model.geom("aruco_marker").id
        self.marker_mat = int(model.geom_matid[self.marker])
        self.marker_rgba = model.geom_rgba[self.marker].copy()
        mujoco.mj_forward(model, data)
        sid = model.site("socket_face").id
        self.pivot = data.site_xpos[sid].copy()  # nominal inlet position: rotation centre
        self.pivot[2] = 0.0
        self.current = park or CarPose()
        self.moves = sorted(moves, key=lambda mv: mv.at)
        self.occlusions = occlusions
        self.log: list[tuple[float, str]] = []
        self._active: tuple[float, CarPose, CarMove] | None = None
        self._covered = False
        self.apply(data, self.current)
        if park and park != CarPose():
            self.log.append((round(float(data.time), 3), f"car parked at dx={park.dx * 100:+.1f} cm "
                             f"dy={park.dy * 100:+.1f} cm dz={park.dz * 100:+.1f} cm yaw={park.dyaw_deg:+.1f} deg"))

    def apply(self, data, pose: CarPose) -> None:
        yaw = math.radians(pose.dyaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        origin = self.pivot + rz @ (-self.pivot) + np.array([pose.dx, pose.dy, pose.dz])
        data.mocap_pos[self.mocap] = origin
        data.mocap_quat[self.mocap] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def cover_marker(self, covered: bool) -> None:
        if covered == self._covered:
            return
        self._covered = covered
        if covered:  # something blocks the camera's view of the inlet area
            self.m.geom_matid[self.marker] = -1
            self.m.geom_rgba[self.marker] = [0.62, 0.05, 0.06, 1.0]
        else:
            self.m.geom_matid[self.marker] = self.marker_mat
            self.m.geom_rgba[self.marker] = self.marker_rgba

    def set_lighting(self, scale: float) -> None:
        """Scale every light (scene lights and headlight); 1.0 = normal."""
        d0, a0, hd0, ha0 = self._light0
        self.m.light_diffuse[:] = d0 * scale
        self.m.light_ambient[:] = a0 * scale
        self.m.vis.headlight.diffuse[:] = hd0 * scale
        self.m.vis.headlight.ambient[:] = ha0 * scale
        self.light_scale = scale

    def set_flap(self, closed: bool) -> None:
        """Closed charging flap: a solid panel in front of the inlet (same colour as the car)."""
        g = self.m.geom("charge_flap").id
        # bit 4 alone collides with nothing (and keeps the geom in the collision structures)
        self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 3 if closed else 4
        self.m.geom_rgba[g, 3] = 1.0 if closed else 0.0
        self.flap_closed = closed

    def set_obstacle(self, on: bool) -> None:
        """A panel appears between the plug and the car (collides with the arm and connector)."""
        g = self.m.geom("obstacle").id
        self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 3 if on else 4
        self.m.geom_rgba[g, 3] = 1.0 if on else 0.0
        self.obstacle_on = on

    def set_sticker(self, on: bool) -> None:
        """A sticker over one corner of the inlet marker (partial occlusion of the camera's target)."""
        self.m.geom_rgba[self.m.geom("marker_sticker").id, 3] = 1.0 if on else 0.0
        self.sticker_on = on

    PROPS = {
        "bay": ("bay_line_l", "bay_line_r", "bay_line_back", "lane_arrow"),
        "carpark": ("pillar_1", "pillar_2", "parked_car", "parked_car_wheel"),
    }

    def set_props(self, *groups: str) -> None:
        """Show (and make solid, where they have collision) the chosen environment prop groups."""
        on = {n for g in groups for n in self.PROPS[g]}
        for names in self.PROPS.values():
            for n in names:
                g = self.m.geom(n).id
                self.m.geom_rgba[g, 3] = 1.0 if n in on else 0.0
                if self.m.geom_contype[g] in (3, 4):  # props that collide when present
                    self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 3 if n in on else 4
        self.props = groups

    def set_car_colour(self, rgba) -> None:
        """Repaint the car body (the robot's camera sees this too)."""
        if not hasattr(self, "_car_rgba0"):
            self._car_rgba0 = self.m.geom_rgba[self.m.geom("car_body").id].copy()
        self.m.geom_rgba[self.m.geom("car_body").id] = rgba

    def restore(self) -> None:
        """Undo model-level changes (lighting, flap, marker cover) so the model can be reused."""
        self.set_lighting(1.0)
        self.set_flap(False)
        self.set_obstacle(False)
        self.set_sticker(False)
        self.set_props()
        if hasattr(self, "_car_rgba0"):
            self.set_car_colour(self._car_rgba0)
        self.cover_marker(False)

    def nudge(self, data, dx=0.0, dy=0.0, dz=0.0, dyaw_deg=0.0) -> None:
        """Immediate external push (used by the live demo's keyboard controls)."""
        cp = self.current
        self.current = CarPose(cp.dx + dx, cp.dy + dy, cp.dz + dz, cp.dyaw_deg + dyaw_deg)
        self.apply(data, self.current)
        self.log.append((round(float(data.time), 3), f"car pushed by hand to dx={self.current.dx * 100:+.1f} "
                         f"dy={self.current.dy * 100:+.1f} cm yaw={self.current.dyaw_deg:+.1f} deg"))

    def step(self, data) -> None:
        t = float(data.time)
        if self._active is None and self.moves and t >= self.moves[0].at:
            mv = self.moves.pop(0)
            self._active = (t, self.current, mv)
            self.log.append((round(t, 3), f"car rolls to dx={mv.to.dx * 100:+.1f} dy={mv.to.dy * 100:+.1f} cm "
                             f"yaw={mv.to.dyaw_deg:+.1f} deg"))
        if self._active is not None:
            t0, a, mv = self._active
            u = min(1.0, (t - t0) / mv.duration)
            u = 0.5 - 0.5 * math.cos(math.pi * u)
            pose = CarPose(*(av + u * (bv - av) for av, bv in zip(dataclasses.astuple(a), dataclasses.astuple(mv.to),
                                                                  strict=True)))
            self.apply(data, pose)
            if u >= 1.0:
                self.current, self._active = mv.to, None
        covered = any(o.start <= t < o.end for o in self.occlusions)
        if covered != self._covered:
            self.log.append((round(t, 3), "inlet view blocked" if covered else "inlet view clear"))
        self.cover_marker(covered)
