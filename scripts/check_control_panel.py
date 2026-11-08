#!/usr/bin/env python3
"""Local HTTP/render smoke check; starts and stops its own ephemeral loopback server."""

import json
import os
import pathlib
import sys
import threading
import time
import urllib.request

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from autocharge.panel.server import make_server


def main():
    folder = pathlib.Path("outputs/panel_http_check")
    folder.mkdir(parents=True, exist_ok=True)
    server = make_server(port=0, output=folder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    checks = []

    def get(path):
        with urllib.request.urlopen(base + path, timeout=90) as response:
            return response.read()

    def state():
        return json.loads(get("/api/state"))

    def post(name, body=None):
        request = urllib.request.Request(
            base + "/api/" + name,
            json.dumps(body or {}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)

    def wait(predicate):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = state()
            if predicate(result):
                return result
            time.sleep(0.1)
        raise AssertionError("Timed out waiting for simulator")

    try:
        html = get("/").decode()
        assert "OBSTACLE EDITOR" in html and "timeline" in html
        get("/app.js")
        get("/style.css")
        checks.append("static interface and assets")
        current = wait(lambda s: s["status"] in ("ready", "invalid"))
        assert current["status"] == "ready", current.get("error")
        checks.append("default scene initialization")
        post("start")
        running = wait(lambda s: s.get("time", 0) > current["time"] + 0.04)
        post("pause")
        paused = wait(lambda s: s["status"] == "paused")
        time.sleep(0.25)
        assert state()["time"] == paused["time"]
        checks.append("start and exact pause")
        post("resume")
        wait(lambda s: s.get("time", 0) > paused["time"] + 0.02)
        post("pause")
        checks.append("resume")
        a = post(
            "configure",
            dict(
                level=5,
                seed=42,
                mode="manual",
                manual=[dict(kind="box", x=-1.5, y=2.0, size=0.2, angle=0)],
            ),
        )
        b = post(
            "configure",
            dict(
                level=5,
                seed=42,
                mode="manual",
                manual=[dict(kind="box", x=-1.5, y=2.2, size=0.2, angle=0)],
            ),
        )
        comparison = post("compare", dict(a=a["attempt"], b=b["attempt"]))
        assert comparison["config_a"]["seed"] == comparison["config_b"]["seed"] == 42
        assert comparison["config_a"]["manual"] != comparison["config_b"]["manual"]
        checks.append("manual obstacle edit and A/B comparison")
        saved = post("save_layout", dict(name="http_check"))
        assert pathlib.Path(saved["path"]).exists()
        checks.append("layout persistence")
        restart = post("restart")
        assert restart["start_pose"] == b["start_pose"]
        checks.append("identical restart")
        previous = None
        for camera in (
            "external",
            "robot_rgb",
            "depth",
            "connector_cam",
            "connector_side",
        ):
            post("camera", dict(camera=camera))
            wait(lambda s: s.get("camera") == camera)
            deadline = time.monotonic() + 60
            while True:
                with urllib.request.urlopen(
                    base + "/api/frame", timeout=90
                ) as response:
                    image = response.read()
                    ready = (
                        response.headers.get("X-Autocharge-Camera") == camera
                        and response.headers.get("X-Autocharge-Attempt")
                        == restart["attempt"]
                    )
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("Stale viewport frame")
                time.sleep(0.1)
            assert image[:2] == b"\xff\xd8" and len(image) > 1000
            assert image != previous
            (folder / (camera + ".jpg")).write_bytes(image)
            previous = image
            checks.append(camera + " rendered viewport")
        post("save_run")
        current = wait(lambda s: s.get("attempt") == restart["attempt"])
        playback = post("playback", dict(id=current["attempt"], index=0))
        assert playback["status"] == "playback"
        wait(lambda s: "playback" in s)
        checks.append("recorded visual playback")
        restored = post("restore", dict(index=0))
        assert restored["status"] == "paused"
        checks.append("complete checkpoint restore")
        result = dict(
            passed=len(checks),
            checks=checks,
            cameras="actual MuJoCo EGL JPEGs",
            browser_automation=False,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        server.worker.close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
