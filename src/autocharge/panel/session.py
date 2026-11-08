"""Single-thread-owned simulation, recording, checkpoints, and rendered views."""

import copy
import dataclasses
import io
import hashlib
import json
import math
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from autocharge.metrics import cable_metrics, plug_alignment_error
from autocharge.training.telemetry import EpisodeTelemetry
from .adaptive import AdaptiveMission
from .scenario import Config, build_scenario

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
CAMERAS = ("external", "robot_rgb", "depth", "connector_cam", "connector_side")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Recording:
    def __init__(self, config, model):
        self.config = dataclasses.asdict(config)
        self.model = model
        self.frames = []
        self.actions = []
        self.checkpoints = {}
        self.scene_context = None
        self.created = time.time()
        self.id = "attempt_" + str(time.time_ns())

    def capture(self, session, checkpoint=False):
        data = session.data
        state = np.empty(mujoco.mj_stateSize(session.model, STATE))
        mujoco.mj_getState(session.model, data, state, STATE)
        frame = dict(time=float(data.time), state=state, telemetry=session.snapshot())
        self.frames.append(frame)
        if checkpoint:
            # Full MjData copy preserves solver warmstarts and derived fields; controller
            # memory is copied with model identity pinned. No serialized Python pickle.
            self.checkpoints[len(self.frames) - 1] = (
                copy.copy(data),
                copy.deepcopy(session.controller, {id(session.model): session.model}),
                copy.deepcopy(session.telemetry, {id(session.model): session.model}),
            )
        return len(self.frames) - 1

    def save(self, folder):
        if not self.frames:
            raise ValueError("No recorded frames")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (self.id + ".npz")
        mujoco.mj_saveModel(self.model, str(folder / (self.id + ".mjb")), None)
        source = Path(__file__).parents[1]
        digest = hashlib.sha256()
        for file in sorted(source.rglob("*.py")):
            digest.update(file.relative_to(source).as_posix().encode())
            digest.update(file.read_bytes())
        metadata = dict(
            id=self.id,
            config=self.config,
            created=self.created,
            mujoco=mujoco.__version__,
            schema_version=1,
            numpy=np.__version__,
            source_sha256=digest.hexdigest(),
            kind="visual playback; controller checkpoints are in-process only",
            actions=self.actions,
            telemetry=[f["telemetry"] for f in self.frames],
        )
        np.savez_compressed(
            path,
            states=np.stack([f["state"] for f in self.frames]),
            metadata=np.array(json.dumps(metadata, default=plain)),
        )
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["mujoco"] != mujoco.__version__:
                raise ValueError("Recording MuJoCo version mismatch")
            model = mujoco.MjModel.from_binary_path(str(path.with_suffix(".mjb")))
            obj = cls(Config(**metadata["config"]), model)
            obj.id = metadata["id"]
            obj.created = metadata["created"]
            obj.actions = metadata["actions"]
            states = archive["states"].copy()
            if states.shape[1] != mujoco.mj_stateSize(model, STATE):
                raise ValueError("Recording state/model mismatch")
            obj.frames = [
                dict(time=t["time"], state=state, telemetry=t)
                for t, state in zip(metadata["telemetry"], states, strict=True)
            ]
            return obj

    def playback(self, index):
        frame = self.frames[index]
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, frame["state"], STATE)
        mujoco.mj_forward(self.model, data)
        return data


class Session:
    def __init__(self, output="outputs/panel"):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.renderer = None
        self.render_model = None
        self.config = Config()
        self.status = "empty"
        self.error = None
        self.camera = "external"
        self.azimuth = 100.0
        self.elevation = -25.0
        self.distance = 5.7
        self.records = {}
        self.record = None
        self.playing = None
        self.play_index = 0
        self.last_capture = -1.0
        self.last_checkpoint = -1.0
        self.frame_error = None
        self.speed = 1.0
        self.view_camera = None

    def configure(self, values):
        config = Config(**values).validate()
        # Transactional reset: an invalid request preserves the previous experiment.
        model, data, info, obstacles, startup = build_scenario(config)
        controller = AdaptiveMission(
            model, data, config.level, config.seed, config.moving_car
        )
        if self.record is not None:
            self.record.actions.append(
                dict(time=float(self.data.time), action="archive_for_reset")
            )
            self.record.save(self.output / "runs")
            self.records[self.record.id] = self.record
        self.config = config
        self.model = model
        self.data = data
        self.info = info
        self.obstacles = obstacles
        self.startup = startup
        self.controller = controller
        self.telemetry = EpisodeTelemetry(model)
        self.status = "ready"
        self.error = None
        self.playing = None
        self.record = Recording(config, model)
        self.record.scene_context = (info, obstacles, startup)
        self.records[self.record.id] = self.record
        self.last_capture = self.last_checkpoint = float(data.time)
        self.record.capture(self, checkpoint=True)
        self.record.actions.append(
            dict(
                time=float(data.time),
                action="configure",
                config=dataclasses.asdict(config),
            )
        )
        return self.snapshot()

    def action(self, name, body):
        if self.record is not None:
            self.record.actions.append(
                dict(
                    time=float(self.data.time),
                    action=name,
                    parameters=copy.deepcopy(body),
                )
            )
        if name == "configure":
            return self.configure(body)
        if name == "restart":
            return self.configure(dataclasses.asdict(self.config))
        if name == "new_seed":
            config = dataclasses.asdict(self.config)
            config["seed"] = int(body.get("seed", self.config.seed + 1))
            return self.configure(config)
        if name == "camera":
            camera = body.get("camera", self.camera)
            if camera not in CAMERAS:
                raise ValueError("Unknown camera")
            self.camera = camera
            for key, low, high in (
                ("azimuth", -360, 360),
                ("elevation", -85, -5),
                ("distance", 1, 12),
            ):
                if key in body:
                    value = float(body[key])
                    if not math.isfinite(value):
                        raise ValueError("Invalid camera value")
                    setattr(self, key, float(np.clip(value, low, high)))
            return self.snapshot()
        if name == "playback":
            record = self.get_record(body.get("id", self.record.id))
            index = int(body.get("index", 0))
            if not 0 <= index < len(record.frames):
                raise ValueError("Playback index out of range")
            self.status = "playback"
            self.playing = record
            self.play_index = index
            return self.snapshot()
        if name == "live":
            self.playing = None
            self.status = "paused"
            return self.snapshot()
        if name == "restore":
            source = self.get_record(
                body.get("id", self.playing.id if self.playing else self.record.id)
            )
            if not source.checkpoints or source.scene_context is None:
                raise ValueError(
                    "Disk recording is visual playback only; no live controller checkpoints"
                )
            index = int(body.get("index", self.play_index))
            candidates = [
                i
                for i, cp in source.checkpoints.items()
                if i <= index and not cp[1].done
            ]
            if not candidates:
                raise ValueError("No complete checkpoint available")
            index = max(candidates)
            data, controller, telemetry = source.checkpoints[index]
            self.model = source.model
            self.config = Config(**copy.deepcopy(source.config))
            self.info, self.obstacles, self.startup = copy.deepcopy(
                source.scene_context
            )
            self.error = None
            self.data = copy.copy(data)
            self.controller = copy.deepcopy(controller, {id(self.model): self.model})
            self.telemetry = copy.deepcopy(telemetry, {id(self.model): self.model})
            # Archive the original future before branching; do not overwrite Attempt A.
            old = self.record
            old.save(self.output / "runs")
            self.record = Recording(self.config, self.model)
            self.record.scene_context = (self.info, self.obstacles, self.startup)
            self.records[self.record.id] = self.record
            self.playing = None
            self.status = "paused"
            self.last_capture = self.last_checkpoint = float(self.data.time)
            self.record.actions.append(
                dict(
                    action="checkpoint_restore",
                    source=source.id,
                    frame=index,
                    time=float(self.data.time),
                )
            )
            self.record.capture(self, checkpoint=True)
            return self.snapshot()
        if name == "preview_placement":
            if self.camera != "external" or self.view_camera is None:
                raise ValueError(
                    "Ground placement requires the live external orbit view"
                )
            if self.status == "running" or self.playing is not None:
                raise ValueError(
                    "Pause and return to live view before placing obstacles"
                )
            if self.config.level < 5 or body.get("mode") not in ("manual", "hybrid"):
                raise ValueError(
                    "Ground placement requires Manual/Hybrid mode at Level 5+"
                )
            from .placement import ground_point, placement_check, project
            from autocharge.training.levels import Obstacle

            point = ground_point(self.view_camera, float(body["u"]), float(body["v"]))
            value = dict(body["obstacle"])
            value.update(x=round(float(point[0]), 2), y=round(float(point[1]), 2))
            candidate = Obstacle(**value)
            selected = int(body.get("selected", -1))
            check = placement_check(
                self.model,
                self.data,
                candidate,
                self.telemetry.contacts.robot,
                [Obstacle(**o) for o in body["manual"]] if "manual" in body else [o for i, o in enumerate(self.obstacles) if i != selected],
            )
            check["obstacle"] = value
            check["footprint_uv"] = project(
                self.view_camera, [[x, y, 0.025] for x, y in check["footprint"]]
            )
            return check
        if name == "save_layout":
            identifier = body.get("name", "layout")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", identifier):
                raise ValueError(
                    "Use letters, digits, dash or underscore for layout name"
                )
            folder = self.output / "layouts"
            folder.mkdir(exist_ok=True)
            path = folder / (identifier + "_" + str(time.time_ns()) + ".json")
            content = dict(
                level=self.config.level,
                seed=self.config.seed,
                mode="manual",
                source_mode=self.config.mode,
                moving_car=self.config.moving_car,
                random_start=self.config.random_start,
                obstacles=[dataclasses.asdict(o) for o in self.obstacles],
            )
            path.write_text(json.dumps(content, indent=2))
            return dict(path=str(path), layout=content)
        if name == "compare":
            a = self.get_record(body["a"])
            b = self.get_record(body["b"])
            return dict(
                a=a.id,
                b=b.id,
                config_a=a.config,
                config_b=b.config,
                outcome_a=a.frames[-1]["telemetry"],
                outcome_b=b.frames[-1]["telemetry"],
            )
        if self.model is None:
            raise ValueError("Configure a valid scenario first")
        if name in ("start", "resume"):
            if self.controller.done:
                raise ValueError("Attempt ended; restart first")
            if self.playing is not None:
                raise ValueError("Return to live simulation before resuming")
            self.status = "running"
        elif name == "pause":
            self.status = "paused"
        elif name == "vehicle_motion":
            self.controller.request_vehicle_motion(self.data)
        elif name == "save_run":
            return dict(path=str(self.record.save(self.output / "runs")))
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def get_record(self, identifier):
        if identifier in self.records:
            return self.records[identifier]
        if not re.fullmatch(r"attempt_[0-9]+", identifier):
            raise ValueError("Invalid attempt identifier")
        path = self.output / "runs" / (identifier + ".npz")
        if not path.exists():
            raise ValueError("Recording not found")
        self.records[identifier] = Recording.load(path)
        return self.records[identifier]

    def tick(self, steps=10):
        if self.status != "running":
            return
        for _ in range(steps):
            self.controller.step(self.data)
            mujoco.mj_step(self.model, self.data)
            self.telemetry.sample(self.data, self.controller.mission.drive.lim)
            active = self.telemetry.contacts.active
            if any(
                k[0] in ("obstacle_robot", "unsafe_robot_environment", "robot_self")
                for k in active
            ):
                self.controller.stop(self.data, "unsafe robot contact: stopped")
            if not np.isfinite(self.data.qpos).all() or any(
                w.number for w in self.data.warning
            ):
                self.controller.stop(self.data, "physics warning/nonfinite state")
            if self.controller.done or self.data.time > 90.0:
                self.status = "finished"
                if self.data.time > 90.0:
                    self.controller.stop(self.data, "90 s experiment time limit")
                self.record.capture(self, checkpoint=True)
                self.record.save(self.output / "runs")
                break
        if self.data.time - self.last_capture >= 0.1 - 1e-9:
            checkpoint = self.data.time - self.last_checkpoint >= 2.0 - 1e-9
            self.record.capture(self, checkpoint)
            self.last_capture = float(self.data.time)
            if checkpoint:
                self.last_checkpoint = float(self.data.time)

    def snapshot(self):
        if self.model is None:
            return dict(
                status=self.status,
                error=self.error,
                config=dataclasses.asdict(self.config),
                frames=0,
                records=[],
            )
        mission = self.controller.mission
        cab = cable_metrics(self.model, self.data)
        al = plug_alignment_error(self.model, self.data)
        state = dict(
            status=self.status,
            time=float(self.data.time),
            phase=self.controller.phase,
            config=dataclasses.asdict(self.config),
            pose=mission.drive.pose(self.data),
            start_pose=self.info.params.base_start,
            nominal_distance=2.24 if self.config.level == 11 else 2.5 + 0.04 * (self.config.level - 1),
            cable_length=self.info.cable_length,
            tension=cab.max_tension,
            strain_relief=cab.strain_relief_load,
            alignment=dict(
                lateral_mm=al.lateral * 1000,
                axial_mm=al.axial * 1000,
                angle_deg=math.degrees(al.angle),
            ),
            detected_cells=len(self.controller.sensors.cells),
            ray_hits=self.controller.sensors.hit_count,
            sensor_range=2.2,
            replans=self.controller.replans,
            route_version=self.controller.route_version,
            route=np.vstack(
                [mission.drive.path[::10], mission.drive.path[-1]]
            ).tolist(),
            port_speed=self.controller.sensors.port_speed,
            vehicle_moving=self.controller.vehicle.moving,
            failure=self.controller.failed
            or (
                mission.events[-1][1]
                if self.controller.done and self.controller.phase != "seated"
                else None
            ),
            observations=dict(
                ray_points=[[o.x, o.y] for o in self.controller.scan_points],
                port_pose=self.controller.sensors.last_port[0].tolist()
                if self.controller.sensors.last_port
                else None,
                port_timestamp=self.controller.sensors.port_time,
            ),
            events=sorted(
                self.controller.events + mission.events, key=lambda event: event[0]
            )[-80:],
            contacts=self.telemetry.contacts.report(),
            cable=self.telemetry.report()["cable"],
            obstacles=[dataclasses.asdict(o) for o in self.obstacles],
            frames=len(self.record.frames) if self.record else 0,
            attempt=self.record.id if self.record else None,
            checkpoint_frames=list(self.record.checkpoints) if self.record else [],
            camera=self.camera,
            view_camera=self.view_camera,
            camera_error=self.frame_error,
            error=self.error,
        )
        return state

    def state(self):
        # Keep six recent live attempts; older runs remain selectable on disk.
        if len(self.records) > 6:
            protected = {self.record.id} | (
                {self.playing.id} if self.playing else set()
            )
            for key in list(self.records):
                if len(self.records) <= 6:
                    break
                if key not in protected:
                    del self.records[key]
        state = self.snapshot()
        state["records"] = [
            dict(
                id=r.id,
                frames=len(r.frames),
                level=r.config["level"],
                seed=r.config["seed"],
                kind="live checkpoints" if r.checkpoints else "visual playback only",
                checkpoints=[
                    dict(frame=i, time=r.frames[i]["time"])
                    for i, cp in r.checkpoints.items()
                    if not cp[1].done
                ],
            )
            for r in self.records.values()
        ]
        saved = self.output / "runs"
        if saved.exists():
            known = {r["id"] for r in state["records"]}
            state["records"] += [
                dict(
                    id=p.stem,
                    frames=None,
                    level=None,
                    seed=None,
                    kind="disk visual playback only",
                    checkpoints=[],
                )
                for p in sorted(saved.glob("attempt_*.npz"))
                if p.stem not in known
            ]
        if self.playing is not None:
            state["playback"] = dict(
                id=self.playing.id,
                index=self.play_index,
                frames=len(self.playing.frames),
                telemetry=self.playing.frames[self.play_index]["telemetry"],
                kind="live checkpoints"
                if self.playing.checkpoints
                else "visual playback only",
            )
        return state

    def render(self):
        model = self.playing.model if self.playing is not None else self.model
        if model is None:
            return None
        data = (
            self.playing.playback(self.play_index)
            if self.playing is not None
            else self.data
        )
        if self.render_model is not model:
            if self.renderer:
                self.renderer.close()
            self.renderer = mujoco.Renderer(model, 540, 900)
            self.render_model = model
        renderer = self.renderer
        if self.camera == "external":
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [-1.0, -0.2, 0.55]
            cam.distance = self.distance
            cam.azimuth = self.azimuth
            cam.elevation = self.elevation
        elif self.camera == "connector_side":
            # Operator-only free view; original connector/RGB perception cameras are untouched.
            tip = data.site_xpos[model.site("plug_tip").id]
            socket = data.site_xpos[model.site("socket_face").id]
            axis = data.site_xmat[model.site("socket_face").id].reshape(3, 3)[:, 2]
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (tip + socket) / 2
            cam.distance = 0.36
            cam.azimuth = math.degrees(math.atan2(axis[1], axis[0])) - 90
            cam.elevation = -18
        else:
            cam = "robot_rgb" if self.camera == "depth" else self.camera
        if self.camera == "depth":
            renderer.enable_depth_rendering()
        else:
            renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        self.view_camera = None
        if self.camera == "external":
            gl = mujoco.mjv_averageCamera(
                renderer.scene.camera[0], renderer.scene.camera[1]
            )
            width = float(gl.frustum_width) or float(
                gl.frustum_top - gl.frustum_bottom
            ) * (900 / 540)
            self.view_camera = dict(
                pos=gl.pos.tolist(),
                forward=gl.forward.tolist(),
                up=gl.up.tolist(),
                near=float(gl.frustum_near),
                width=width,
                center=float(gl.frustum_center),
                top=float(gl.frustum_top),
                bottom=float(gl.frustum_bottom),
                view=dict(
                    azimuth=self.azimuth,
                    elevation=self.elevation,
                    distance=self.distance,
                ),
            )
        if self.camera == "external":
            points = (
                self.controller.mission.drive.path
                if self.playing is None
                else np.array(
                    self.playing.frames[self.play_index]["telemetry"]["route"]
                )
            )
            for point in points[:: max(1, len(points) // 50)]:
                scene = renderer.scene
                if scene.ngeom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.018, 0.018, 0.018]),
                    np.array([point[0], point[1], 0.025]),
                    np.eye(3).ravel(),
                    np.array([0.15, 0.65, 0.95, 0.9], dtype=np.float32),
                )
                scene.ngeom += 1
        pixels = renderer.render()
        if self.camera == "depth":
            normalized = np.clip(pixels / 5.0, 0, 1)
            pixels = np.repeat(
                ((1 - normalized) * 255).astype(np.uint8)[:, :, None], 3, axis=2
            )
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=85)
        self.frame_error = None
        return stream.getvalue()

    def close(self):
        if self.record and self.record.frames:
            self.record.save(self.output / "runs")
        if self.renderer:
            self.renderer.close()
