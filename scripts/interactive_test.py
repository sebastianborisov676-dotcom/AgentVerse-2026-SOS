"""Judge-controlled autonomy challenges: YOU control the world, the robot controls itself.

    python scripts/interactive_test.py

Click the "robot view" window so it has keyboard focus.

    1..9, 0, - load challenge 1..11 (resets the robot, sets up the car)
    K / N / M  drop a bin bag on the route / park neighbour cars / make the car roll now
    SPACE      start charging (the robot only gets its camera image and its own sensors)
    W/S A/D    car 1 cm further/closer, left/right      (hold SHIFT: 10 cm)
    R/F  Q/E   car 1 cm up/down, turn 1 degree
    B          block / unblock the camera's view          L  cycle lighting 100/40/15/3 %
    P          open / close the charging flap             O  put / remove an obstacle
    I          stick / remove a sticker on the marker     U  unplug and reset the arm
    G          show / hide the standard-bay comparison ghost
    X          emergency stop                             ESC quit

Three verdicts per attempt, from the independent checker (physical socket pose, never shown
to the robot): DOCKING, SAFETY, AUTONOMY EVIDENCE. Logs: outputs/challenge_logs/.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from watch_tests import draw_overlays, ghost_geometry  # noqa: E402

from autocharge import build_model, reset_to_home  # noqa: E402
from autocharge.autonomy import AgentState, AutonomousAgent  # noqa: E402
from autocharge.evaluation import check_docking  # noqa: E402
from autocharge.overlay import robot_camera_overlay, status_panel  # noqa: E402
from autocharge.perception import PortPerception  # noqa: E402
from autocharge.scenarios import CarMove, CarPose, Occlusion, ScenarioManager  # noqa: E402
from autocharge.scene import SceneParams  # noqa: E402

LIGHTS = (1.0, 0.4, 0.15, 0.03)
LIMITS = {"dx": (-0.10, 3.50), "dy": (-1.20, 1.20), "dz": (-0.06, 0.30), "yaw": (-25.0, 25.0)}


@dataclasses.dataclass
class Challenge:
    n: int
    name: str
    tell: str  # what the judge should do
    expect: str  # dock | refuse | touch-stop
    car: CarPose = dataclasses.field(default_factory=CarPose)
    flap: bool = False
    obstacle: bool = False
    sticker: bool = False
    auto: str = ""  # automatic world changes after SPACE (scenario manager, robot not told)
    colour: tuple = (0.62, 0.05, 0.06, 1.0)  # car paint (the robot's camera sees it)
    props: tuple = ()  # environment prop groups: "bay", "carpark"
    light: float = 1.0
    cam: tuple = (0.6, -0.3, 0.6, 3.0, 60.0, -24.0)  # viewer lookat x,y,z, distance, azimuth, elevation
    bags: tuple = ()  # rubbish bags on the floor, world (x, y)
    neighbours: bool = False  # three parked cars around the target


CHALLENGES = [
    Challenge(1, "Standard parking", "Red car parked normally in the EV bay, clear approach. SPACE", "dock",
              colour=(0.62, 0.05, 0.06, 1.0), cam=(0.9, -0.4, 0.4, 5.5, 235.0, -38.0)),
    Challenge(2, "Socket on the far side", "Car faces the other way: inlet on the far side. SPACE", "refuse",
              car=CarPose(1.97, 3.10, 0.0, 180.0), colour=(0.92, 0.92, 0.9, 1.0), cam=(1.6, 1.0, 0.4, 7.0, 230.0, -45.0)),
    Challenge(3, "Angled parking", "Car parked at a random angle (+5/+10/-8/random). SPACE", "dock",
              auto="angle", colour=(0.68, 0.7, 0.74, 1.0), cam=(1.4, 0.8, 0.3, 6.0, 270.0, -70.0)),
    Challenge(4, "Car moves during approach", "SPACE. Car rolls by itself at 3 s, or press M any time", "dock",
              auto="roll", colour=(0.92, 0.92, 0.9, 1.0), cam=(0.55, -0.32, 0.8, 1.6, 90.0, -14.0)),
    Challenge(5, "Bin bag blocking the approach", "Car 1.5 m away, black bag on the route. SPACE: detour?", "dock",
              car=CarPose(1.5, 0.0, 0.0, 0.0), bags=((0.8, 0.0),), colour=(0.62, 0.05, 0.06, 1.0), cam=(0.9, -0.3, 0.3, 4.0, 250.0, -55.0)),
    Challenge(6, "Three cars around the target", "Cars in front, behind and on the far side. SPACE", "dock",
              car=CarPose(0.9, -0.6, 0.0, 0.0), neighbours=True, colour=(0.62, 0.05, 0.06, 1.0), cam=(1.6, -0.5, 0.3, 8.0, 240.0, -60.0)),
    Challenge(7, "Closed charging flap", "Flap is shut (P toggles). SPACE: it must not fake success", "refuse", flap=True,
              colour=(0.62, 0.05, 0.06, 1.0), cam=(0.70, -0.32, 0.85, 0.85, 30.0, -10.0)),
    Challenge(8, "Night + partial view", "Lights at 15%, sticker on the marker for the first 3 s. SPACE", "dock",
              sticker=True, light=0.15, auto="night-sticker", colour=(0.07, 0.07, 0.08, 1.0), cam=(0.45, -0.35, 0.75, 1.5, 200.0, -15.0)),
    Challenge(9, "Beyond the cable's reach", "Car 3.3 m away in a distant bay. SPACE: must stop safely", "refuse",
              car=CarPose(3.3, 0.1, 0.0, 0.0), colour=(0.68, 0.7, 0.74, 1.0), cam=(1.4, -0.6, 0.4, 7.5, 70.0, -40.0)),
    Challenge(10, "Multiple obstacles (random)", "Random bags, neighbours and angle each load. SPACE", "any",
              auto="multi", colour=(0.1, 0.25, 0.7, 1.0), cam=(1.0, -0.4, 0.3, 6.5, 225.0, -55.0)),
    Challenge(11, "Judge's mystery (press -)", "Combine anything: WASD QE, K bag, N cars, M move, B, L, P, I", "any",
              colour=(0.62, 0.05, 0.06, 1.0), cam=(0.9, -0.4, 0.4, 6.0, 135.0, -35.0)),
]


class IdleView:
    def __init__(self, perception):
        self.perception, self.last_obs, self.ctrl = perception, None, None
        self.filter = type("F", (), {"pos": None})()


def verdicts(ch: Challenge, agent: AutonomousAgent, model, data, extra: dict) -> dict:
    v = check_docking(model, data)
    claimed = agent.state == AgentState.SUCCESS
    peak = agent.ctrl.peak["contact_force"] if agent.ctrl else 0.0
    events = [e for _, e in agent.events]
    first_motion = next((t for t, e in agent.decisions if e.startswith(("inlet within", "inlet out of arm"))), None)
    first_seen = next((o["t"] for o in agent.obs_log if o["detected"]), None)
    safety_ok = ((not claimed or v.docked) and peak < 70.0 and extra.get("collisions", 0) == 0
                 and not (agent.touched and agent.state != AgentState.SAFE_STOP))
    evidence = []
    if first_motion is not None:
        evidence.append("moved only after a camera detection" if first_seen is not None and first_seen <= first_motion
                        else "MOVED WITHOUT DETECTION")
    else:
        evidence.append("never moved (no reliable view or unreachable)")
    if extra["car_moves_during_run"]:
        evidence.append(f"car moved {extra['car_moves_during_run']}x during run -> "
                        f"{sum('car moved' in e for e in events)} re-plan events")
    if any(e.startswith("hold") for e in events):
        evidence.append(f"held while blind (plug moved {extra['blind_motion_mm']:.1f} mm)")
    if any("LiDAR: obstacle" in e or "re-planned route" in e for e in events):
        evidence.append("LiDAR saw an obstacle and the route was planned around it")
    if agent.repositions:
        evidence.append(f"drove base {extra['base_travel_m']:.2f} m to reach the car")
    evidence_ok = "MOVED WITHOUT DETECTION" not in evidence[0] and extra["blind_motion_mm"] < 5.0
    if ch.expect == "dock":
        outcome_ok = v.docked
    elif ch.expect == "refuse":
        outcome_ok = not v.docked and not claimed
    elif ch.expect == "touch-stop":
        outcome_ok = agent.touched is not None and agent.state == AgentState.SAFE_STOP
    else:
        outcome_ok = v.docked or agent.state in (AgentState.SAFE_STOP, AgentState.FAILED)
    return {
        "docking": ("DOCKED" if v.docked else "NOT DOCKED") + f" ({v.reason})",
        "docking_expected_ok": outcome_ok,
        "safety_ok": safety_ok,
        "safety": f"collisions {extra.get('collisions', 0)}, peak plug force {peak:.0f} N, "
                  f"false success: {'YES' if claimed and not v.docked else 'no'}",
        "evidence_ok": evidence_ok,
        "evidence": "; ".join(evidence),
        "robot_state": agent.state.value,
    }


def header(width: int, ch: Challenge, car: CarPose, light: float, toggles: str, res: dict | None) -> np.ndarray:
    rows = 4 + (3 if res else 0)
    img = np.full((14 + 20 * rows, width, 3), (60, 40, 20), np.uint8)

    def put(i, text, col=(230, 230, 230), scale=0.45, thick=1):
        cv2.putText(img, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)

    put(0, f"CHALLENGE {ch.n}: {ch.name}", (255, 255, 255), 0.55, 2)
    put(1, ch.tell[:68], (255, 230, 160))
    exp = {"dock": "should dock", "refuse": "should REFUSE safely", "touch-stop": "should STOP on touch",
           "any": "dock or stop safely"}[ch.expect]
    put(2, f"expected: {exp}   light {int(light * 100)}%  {toggles}")
    put(3, f"car vs bay: {car.dx * 100:+.0f} far {car.dy * 100:+.0f} left {car.dz * 100:+.0f} up cm {car.dyaw_deg:+.0f} deg")
    if res:
        for i, (key, label, ok) in enumerate((("docking", "DOCKING", res["docking_expected_ok"]),
                                              ("safety", "SAFETY", res["safety_ok"]),
                                              ("evidence", "AUTONOMY", res["evidence_ok"]))):
            y = 22 + 20 * (4 + i)
            col = (40, 170, 40) if ok else (40, 40, 220)
            cv2.rectangle(img, (8, y - 14), (100, y + 4), col, -1)
            cv2.putText(img, f"{label} {'OK' if ok else 'X'}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(img, res[key][:62], (106, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 200), 1)
    return img


def main() -> int:
    print(__doc__)
    model, _ = build_model(SceneParams(mobile_base=True, base_start=(0.0, 0.0, 0.0), cable_segments=64))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    log_dir = ROOT / "outputs" / "challenge_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from mujoco import viewer as mj_viewer

    perception = PortPerception(model)
    plug_site = model.site("plug_face").id
    base_b = model.body("base_link").id
    steps = int(round((1 / 30) / model.opt.timestep))
    with mj_viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.6, -0.3, 0.6]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 60.0, -24.0
        with v.lock():
            ghost = ghost_geometry(model, data)
            sc = ScenarioManager(model, data)
        st = {"ch": CHALLENGES[0], "agent": None, "res": None, "trail": [], "prev": [], "light": 0,
              "blind": False, "moves": 0, "blind_mm": 0.0, "base0": None, "attempt": 0}
        idle = IdleView(perception)
        obstacle_geoms = {g for g in range(model.ngeom)
                          if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                              ("bin_bag", "nb_", "pcar_", "bollard", "charger_pedestal", "wall_", "obstacle"))}
        robot_root = model.body("base_link").id
        robot_bodies = set()
        for bb in range(model.nbody):
            aa = bb
            while aa not in (0, robot_root):
                aa = model.body_parentid[aa]
            if aa == robot_root:
                robot_bodies.add(bb)

        rng = np.random.default_rng()

        def random_spot() -> CarPose:
            return CarPose(float(rng.uniform(-0.05, 0.9)), float(rng.uniform(-0.45, 0.45)),
                           float(rng.uniform(-0.03, 0.08)), float(rng.uniform(-12, 12)))

        def plan_auto(ch: Challenge) -> list:
            """(seconds after SPACE, action, argument) - executed by the scenario manager only."""
            if ch.auto == "night-sticker":
                return [(3.0, "sticker", False)]
            if ch.auto == "roll":
                return [(3.0, "move", (0.0, 0.06, 0.0, 4.0))]
            if ch.auto == "blackout":
                return [(2.0, "block", True), (4.5, "block", False)]
            if ch.auto == "jumps":
                return [(2.0 + i, "move", (float(rng.uniform(-0.02, 0.02)), float(rng.choice([-1, 1]) * 0.025), 0.0,
                                           float(rng.uniform(-2, 2)))) for i in range(4)]
            if ch.auto == "mystery":
                pool = [(float(rng.uniform(1.5, 4.0)), "move", (0.0, float(rng.uniform(-0.05, 0.05)), 0.0,
                                                               float(rng.uniform(-4, 4)))),
                        (float(rng.uniform(1.5, 3.5)), "block", True),
                        (float(rng.uniform(1.0, 3.0)), "light", 0.15)]
                picks = [pool[i] for i in sorted(rng.choice(len(pool), size=int(rng.integers(1, 4)), replace=False))]
                out = []
                for t, a, arg in picks:
                    out.append((t, a, arg))
                    if a == "block":
                        out.append((t + 2.0, "block", False))
                return out
            return []

        def bag_on_route():
            """World point ~45% of the way from the robot to the inlet (the environment knows the truth)."""
            mujoco.mj_forward(model, data)
            a = data.xpos[base_b][:2]
            b = data.site_xpos[model.site("socket_face").id][:2]
            p = a + 0.45 * (b - a)
            return float(p[0]), float(p[1])

        def load(ch: Challenge, keep_car: bool = False):
            if keep_car:
                car = sc.current
            elif ch.auto in ("random-park", "mystery"):
                car = random_spot()
            elif ch.auto == "angle":
                car = CarPose(0.0, 0.0, 0.0, float(rng.choice([5.0, 10.0, -8.0, float(rng.uniform(-12, 12))])))
            elif ch.auto == "multi":
                car = CarPose(float(rng.uniform(0.8, 1.5)), float(rng.uniform(-0.6, 0.2)), 0.0, float(rng.uniform(-8, 8)))
            else:
                car = ch.car
            sc.restore()
            reset_to_home(model, data, settle=0.0)
            sc.current = car
            sc.apply(data, car)
            sc.set_flap(ch.flap)
            sc.set_obstacle(ch.obstacle)
            sc.set_sticker(ch.sticker)
            mujoco.mj_forward(model, data)
            bags = list(ch.bags)
            neigh = ch.neighbours
            if ch.auto == "multi" and not keep_car:
                neigh = bool(rng.integers(2))
                bags = [bag_on_route()] if rng.integers(2) else []
            sc.set_bags(*bags)
            sc.set_neighbours(data, neigh)
            st["light"] = min(range(len(LIGHTS)), key=lambda i: abs(LIGHTS[i] - ch.light))
            sc.set_lighting(LIGHTS[st["light"]])
            sc.set_props(*ch.props)
            colours = [(0.92, 0.92, 0.9, 1.0), (0.68, 0.7, 0.74, 1.0), (0.1, 0.25, 0.7, 1.0), (0.62, 0.05, 0.06, 1.0)]
            sc.set_car_colour(colours[int(rng.integers(len(colours)))] if ch.auto == "mystery" else ch.colour)
            lx, ly, lz, dist, az, el = ch.cam
            v.cam.lookat[:] = [lx, ly, lz]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, az, el
            sc.occlusions = ()
            mujoco.mj_forward(model, data)
            for _ in range(250):
                sc.step(data)
                mujoco.mj_step(model, data)
            st.update(collisions=0)
            st.update(ch=ch, agent=None, res=None, blind=False, moves=0, blind_mm=0.0, base0=None,
                      auto=plan_auto(ch), t_go=None, start_ghost=None, start_car=None)
            st["prev"], st["trail"] = st["trail"], []
            sc.log.append((round(float(data.time), 2), f"challenge {ch.n} loaded"))

        with v.lock():
            load(CHALLENGES[0])
        while v.is_running():
            t0 = time.time()
            key = cv2.waitKeyEx(1)
            ch_key = chr(key) if 0 <= key < 256 else ""
            agent = st["agent"]
            with v.lock():
                big = ch_key.isupper()
                step_m = 0.10 if big else 0.01
                mv = {"w": (step_m, 0, 0, 0), "s": (-step_m, 0, 0, 0), "a": (0, step_m, 0, 0), "d": (0, -step_m, 0, 0),
                      "r": (0, 0, 0.01, 0), "f": (0, 0, -0.01, 0), "q": (0, 0, 0, 1.0), "e": (0, 0, 0, -1.0)
                      }.get(ch_key.lower())
                if mv:
                    cp = sc.current
                    nx = float(np.clip(cp.dx + mv[0], *LIMITS["dx"]))
                    ny = float(np.clip(cp.dy + mv[1], *LIMITS["dy"]))
                    nz = float(np.clip(cp.dz + mv[2], *LIMITS["dz"]))
                    nyaw = float(np.clip(cp.dyaw_deg + mv[3], *LIMITS["yaw"]))
                    sc.nudge(data, nx - cp.dx, ny - cp.dy, nz - cp.dz, nyaw - cp.dyaw_deg)
                    if agent is not None and not agent.done:
                        st["moves"] += 1
                elif ch_key in "1234567890" and ch_key:
                    load(CHALLENGES[(int(ch_key) - 1) % 10])
                elif ch_key == "-":
                    load(CHALLENGES[10])
                elif ch_key.lower() == "k":
                    has = getattr(sc, "bags", ())
                    sc.set_bags(*(() if has else (bag_on_route(),)))
                    sc.log.append((round(float(data.time), 2), "bin bag removed" if has else "bin bag dropped on the route"))
                elif ch_key.lower() == "n":
                    sc.set_neighbours(data, not getattr(sc, "neighbours", False))
                    sc.log.append((round(float(data.time), 2), "neighbour cars " + ("parked" if sc.neighbours else "gone")))
                elif ch_key.lower() == "m":
                    sc.moves = [CarMove(at=float(data.time), to=CarPose(sc.current.dx, sc.current.dy + float(rng.choice([-1, 1])) * 0.05,
                                                                       sc.current.dz, sc.current.dyaw_deg + float(rng.uniform(-4, 4))))]
                    if agent is not None and not agent.done:
                        st["moves"] += 1
                    sc.log.append((round(float(data.time), 2), "you: car set rolling"))
                elif ch_key == " " and agent is None:
                    st["attempt"] += 1
                    st["agent"] = AutonomousAgent(model, data, perception=perception)
                    st["base0"] = data.xpos[base_b][:2].copy()
                    st["t_go"] = float(data.time)
                    st["start_car"] = sc.current
                    st["start_ghost"] = [(sz, pos, mat, np.array([1.0, 0.75, 0.1, 0.18 if i == 0 else 0.6]))
                                         for i, (sz, pos, mat, _) in enumerate(ghost_geometry(model, data))]
                    st["prev"], st["trail"] = st["trail"], []
                    sc.log.append((round(float(data.time), 2), "SPACE: robot started"))
                elif ch_key.lower() == "b":
                    st["blind"] = not st["blind"]
                    sc.occlusions = (Occlusion(0.0, 1e9),) if st["blind"] else ()
                    sc.log.append((round(float(data.time), 2), "camera blocked" if st["blind"] else "camera clear"))
                elif ch_key.lower() == "l":
                    st["light"] = (st["light"] + 1) % len(LIGHTS)
                    sc.set_lighting(LIGHTS[st["light"]])
                    sc.log.append((round(float(data.time), 2), f"lighting {int(LIGHTS[st['light']] * 100)}%"))
                elif ch_key.lower() == "p":
                    sc.set_flap(not sc.flap_closed)
                    sc.log.append((round(float(data.time), 2), "flap closed" if sc.flap_closed else "flap opened"))
                elif ch_key.lower() == "o":
                    sc.set_obstacle(not sc.obstacle_on)
                    sc.log.append((round(float(data.time), 2), "obstacle placed" if sc.obstacle_on else "obstacle removed"))
                elif ch_key.lower() == "i":
                    sc.set_sticker(not sc.sticker_on)
                    sc.log.append((round(float(data.time), 2), "sticker on" if sc.sticker_on else "sticker removed"))
                elif ch_key.lower() == "g":
                    st["show_ghost"] = not st.get("show_ghost", False)
                elif ch_key.lower() == "u":
                    load(st["ch"], keep_car=True)
                elif ch_key.lower() == "x" and agent is not None and agent.ctrl is not None:
                    agent.ctrl.emergency_stop(data)
                elif key == 27:
                    break

                agent = st["agent"]
                if agent is not None and st.get("t_go") is not None:
                    while st["auto"] and data.time - st["t_go"] >= st["auto"][0][0]:
                        _, action, arg = st["auto"].pop(0)
                        if action == "move":
                            cp = sc.current
                            sc.nudge(data, *[float(v) for v in arg])
                            if not agent.done:
                                st["moves"] += 1
                            sc.log[-1] = (sc.log[-1][0], "AUTO: car moved by itself to "
                                          f"{sc.current.dx * 100:+.0f}/{sc.current.dy * 100:+.0f} cm "
                                          f"{sc.current.dyaw_deg:+.0f} deg")
                            del cp
                        elif action == "block":
                            st["blind"] = bool(arg)
                            sc.occlusions = (Occlusion(0.0, 1e9),) if arg else ()
                            sc.log.append((round(float(data.time), 2), "AUTO: camera blocked" if arg else "AUTO: camera clear"))
                        elif action == "sticker":
                            sc.set_sticker(bool(arg))
                            sc.log.append((round(float(data.time), 2), "AUTO: sticker removed" if not arg else "AUTO: sticker on"))
                        elif action == "light":
                            sc.set_lighting(float(arg))
                            sc.log.append((round(float(data.time), 2), f"AUTO: lights dimmed to {int(arg * 100)}%"))
                for _ in range(steps):
                    p_before = data.site_xpos[plug_site].copy()
                    sc.step(data)
                    if agent is not None:
                        agent.step(data)
                    mujoco.mj_step(model, data)
                    if agent is not None and agent.ctrl is not None and agent.ctrl.hold:
                        st["blind_mm"] += float(np.linalg.norm(data.site_xpos[plug_site] - p_before)) * 1e3
                if agent is None and int(data.time * 10) != int((data.time - steps * model.opt.timestep) * 10):
                    idle.last_obs = perception.observe(data)
                p = data.site_xpos[plug_site].copy()
                if agent is not None and (not st["trail"] or np.linalg.norm(p - st["trail"][-1]) > 0.004):
                    st["trail"].append(p)
                if agent is not None and not agent.done:
                    for ci in range(data.ncon):
                        c = data.contact[ci]
                        g1, g2 = int(c.geom[0]), int(c.geom[1])
                        if g1 < 0 or g2 < 0:
                            continue
                        if (g1 in obstacle_geoms and model.geom_bodyid[g2] in robot_bodies) or \
                                (g2 in obstacle_geoms and model.geom_bodyid[g1] in robot_bodies):
                            st["collisions"] = st.get("collisions", 0) + 1
                if agent is not None and agent.done and st["res"] is None:
                    extra = {"car_moves_during_run": st["moves"], "blind_motion_mm": st["blind_mm"],
                             "collisions": st.get("collisions", 0),
                             "base_travel_m": float(np.linalg.norm(data.xpos[base_b][:2] - st["base0"]))}
                    st["res"] = verdicts(st["ch"], agent, model, data, extra)
                    stamp = datetime.datetime.now().strftime("%H%M%S")
                    rec = {"challenge": st["ch"].n, "name": st["ch"].name, "attempt": st["attempt"],
                           "car": dataclasses.asdict(sc.current), "verdicts": st["res"], "extra": extra,
                           "decisions": agent.events, "world_changes": sc.log, "observations": agent.obs_log}
                    (log_dir / f"challenge{st['ch'].n}_{stamp}.json").write_text(json.dumps(rec, indent=1, default=str))
                shown_ghosts = list(ghost) if st.get("show_ghost") else []
                if st.get("start_ghost") and st.get("start_car") is not None and sc.current != st["start_car"]:
                    shown_ghosts += st["start_ghost"]  # where the car was when the robot started
                draw_overlays(v, shown_ghosts, st["trail"], st["prev"])
                if agent is not None and getattr(agent, "lidar_points", None) is not None:
                    for pt in agent.lidar_points[::2]:
                        if v.user_scn.ngeom >= v.user_scn.maxgeom:
                            break
                        mujoco.mjv_initGeom(v.user_scn.geoms[v.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.02, 0, 0]), np.array([pt[0], pt[1], 0.40]), np.eye(3).ravel(),
                                            np.array([1.0, 0.15, 0.1, 0.9], np.float32))
                        v.user_scn.ngeom += 1
                    path = agent.nav_path or []
                    for a3, b3 in zip(path[:-1], path[1:], strict=False):
                        if v.user_scn.ngeom >= v.user_scn.maxgeom:
                            break
                        gg = v.user_scn.geoms[v.user_scn.ngeom]
                        mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                            np.array([0.2, 0.5, 1.0, 0.9], np.float32))
                        mujoco.mjv_connector(gg, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, np.array([a3[0], a3[1], 0.03]),
                                             np.array([b3[0], b3[1], 0.03]))
                        v.user_scn.ngeom += 1
                if agent is not None and agent.filter.pos is not None and v.user_scn.ngeom < v.user_scn.maxgeom:
                    mujoco.mjv_initGeom(v.user_scn.geoms[v.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.012, 0, 0]), agent.filter.pos, np.eye(3).ravel(),
                                        np.array([0.1, 0.9, 1.0, 0.9], np.float32))
                    v.user_scn.ngeom += 1
            v.sync()

            toggles = " ".join(t for t, on in (("BLIND", st["blind"]), ("FLAP", sc.flap_closed),
                                                ("OBSTACLE", sc.obstacle_on), ("STICKER", sc.sticker_on)) if on)
            shown = agent if agent is not None else idle
            cam = robot_camera_overlay(shown, data, 0.5)
            world = [f"you: {t:.1f}s {txt}" for t, txt in sc.log[-3:]]
            if agent is not None:
                panel = status_panel(agent, data, cam.shape[1], 300, world)
            else:
                panel = np.full((300, cam.shape[1], 3), (30, 24, 20), np.uint8)
                cv2.putText(panel, "WAITING - set up the car, press SPACE", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (255, 255, 255), 2)
                cv2.putText(panel, "keys 1..0 = challenges 1..10", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                for i, ln in enumerate(world):
                    cv2.putText(panel, ln[:60], (14, 110 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            frame = np.vstack([header(cam.shape[1], st["ch"], sc.current, LIGHTS[st["light"]], toggles, st["res"]),
                               cam, panel])
            cv2.imshow("AutoCharge AI - robot view (click here for keys)", frame)
            time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
