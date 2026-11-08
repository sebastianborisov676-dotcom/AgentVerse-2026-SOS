#!/usr/bin/env python3
"""Actual Chromium controls, screenshots and network evidence for the local panel."""

import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


class BrowserCheck:
    def __init__(self, page, url, folder):
        self.page = page
        self.url = url
        self.folder = folder
        self.results = []
        self.requests = []
        self.errors = []
        page.on("pageerror", lambda error: self.errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: self.requests.append(
                dict(url=request.url, error=request.failure)
            ),
        )
        page.on(
            "response",
            lambda response: (
                self.requests.append(dict(url=response.url, status=response.status))
                if response.request.method == "POST"
                else None
            ),
        )
        page.goto(url)
        page.wait_for_function(
            "typeof state!=='undefined' && state?.pose", timeout=120000
        )

    def state(self):
        return self.page.evaluate("state")

    def click(self, selector, action, expected=200):
        self.page.wait_for_function("!busy", timeout=120000)
        with self.page.expect_response(
            lambda r: r.url.endswith("/api/" + action) and r.request.method == "POST",
            timeout=120000,
        ) as pending:
            self.page.locator(selector).click()
        response = pending.value
        result = response.json()
        self.page.wait_for_function("!busy", timeout=120000)
        assert expected is None or response.status == expected, (
            action,
            response.status,
            result,
        )
        if (
            response.status == 200
            and result.get("attempt")
            and action in ("configure", "restart", "restore")
        ):
            self.page.wait_for_function(
                "(id)=>state?.attempt===id && lastImageAttempt===id",
                arg=result["attempt"],
                timeout=60000,
            )
        self.page.wait_for_timeout(600)
        return result

    def slider(self, id, value):
        self.page.locator("#" + id).evaluate(
            '(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}',
            str(value),
        )

    def editor(self):
        p = self.page
        p.select_option("#level", "9")
        p.select_option("#mode", "manual")
        p.fill("#seed", "42")
        while p.locator("#manualList option").count():
            p.select_option("#manualList", index=0)
            self.click("#remove", "configure")
        self.click("#apply", "configure")
        for x, y, code in [(-1.7, 2.86, 200), (-1.7, 0.44, 200), (-2.4, 0.44, 400)]:
            self.slider("x", x)
            self.slider("y", y)
            self.click("#add", "configure", code)
        assert p.locator("#manualList option").count() == 3
        assert p.locator("#obstacleCount").inner_text() == "Active: 2"
        assert "NOT APPLIED" in p.locator("#draftStatus").inner_text()
        assert "/api/configure" in p.locator("#message").inner_text()
        active = self.state()["attempt"]
        self.click("#apply", "configure", 400)
        assert self.state()["attempt"] == active
        with p.expect_download() as download:
            saved = self.click("#saveLayout", "save_layout")
        path = self.folder / "saved_active_layout.json"
        download.value.save_as(path)
        assert len(json.loads(path.read_text())["obstacles"]) == 2
        assert len(saved["layout"]["obstacles"]) == 2
        with p.expect_response(
            lambda r: r.url.endswith("/api/configure") and r.request.method == "POST",
            timeout=120000,
        ) as load:
            p.locator("#loadLayout").set_input_files(str(path.resolve()))
        assert load.value.status == 200
        p.wait_for_function("!busy", timeout=120000)
        assert p.locator("#manualList option").count() == 2
        new_seed = self.click("#newSeed", "configure", None)
        if "error" in new_seed:
            assert self.state()["config"]["seed"] == 42
            assert "Active scene" in p.locator("#message").inner_text()
        else:
            assert self.state()["config"]["seed"] == 43
        p.select_option("#manualList", index=1)
        self.click("#remove", "configure")
        self.click("#newSeed", "configure")
        assert self.state()["config"]["seed"] == 44
        pose = self.state()["start_pose"]
        self.click("#restart", "restart")
        assert self.state()["start_pose"] == pose
        self.results.append(
            dict(
                check="editor",
                passed=True,
                active_vs_draft=True,
                saved_obstacles=2,
                new_seed=44,
                restart_pose=pose,
            )
        )
        p.screenshot(path=str(self.folder / "editor_after.png"), full_page=True)

    def path_box(self):
        p = self.page
        path = self.folder / "path_box_test.json"
        path.write_text(
            json.dumps(
                dict(
                    level=9,
                    seed=42,
                    obstacles=[dict(kind="box", x=-1.7, y=0.44, size=0.22, angle=0)],
                )
            )
        )
        p.fill("#seed", "42")
        with p.expect_response(
            lambda r: r.url.endswith("/api/configure") and r.request.method == "POST",
            timeout=120000,
        ) as load:
            p.locator("#loadLayout").set_input_files(str(path.resolve()))
        assert load.value.status == 200, load.value.text()
        p.wait_for_function("!busy", timeout=120000)
        self.click("#start", "start")
        p.wait_for_function("state.replans>0 || state.failure", timeout=120000)
        if self.state()["status"] == "running":
            self.click("#pause", "pause")
        state = self.state()
        assert state["detected_cells"] > 0
        assert state["replans"] > 0 or state["failure"]
        self.results.append(dict(check="path_box", passed=True, state=state))
        p.screenshot(path=str(self.folder / "path_box_browser.png"), full_page=True)

    def seek(self, frame):
        self.page.wait_for_function("!busy", timeout=120000)
        with self.page.expect_response(
            lambda r: r.url.endswith("/api/playback") and r.request.method == "POST",
            timeout=30000,
        ) as response:
            self.slider("timeline", frame)
        assert response.value.status == 200
        self.page.wait_for_function(
            '(f)=>!busy && state.status==="playback" && state.playback.index===f',
            arg=frame,
            timeout=30000,
        )
        return response.value.json()

    def checkpoint_before(self):
        p = self.page
        p.select_option("#level", "9")
        p.select_option("#mode", "random")
        p.fill("#seed", "42")
        self.click("#apply", "configure")
        self.click("#start", "start")
        p.wait_for_function('state.status==="finished"', timeout=360000)
        finished = self.state()
        source = finished["attempt"]
        frame = finished["checkpoint_frames"][3]
        (self.folder / "panel_level9_reference.json").write_text(
            json.dumps(finished, indent=2)
        )
        self.seek(frame)
        restored = self.click("#restore", "restore")
        before = restored["time"]
        self.click("#resume", "resume")
        p.wait_for_function("(t)=>state.time>t+.3", arg=before, timeout=60000)
        self.click("#pause", "pause")
        advanced = self.state()["time"]
        p.select_option("#attemptB", source)
        p.wait_for_function(
            "(id)=>!busy && state.playback?.id===id", arg=source, timeout=30000
        )
        self.seek(frame)
        disabled = p.locator("#restore").is_disabled()
        assert disabled
        self.results.append(
            dict(
                check="checkpoint_before",
                source=source,
                frame=frame,
                restored_time=before,
                resumed_time=advanced,
                second_restore_disabled=disabled,
                original_preserved=True,
            )
        )
        p.screenshot(path=str(self.folder / "checkpoint_before.png"), full_page=True)

    def checkpoint(self):
        import numpy as np

        p = self.page
        p.select_option("#level", "9")
        p.select_option("#mode", "random")
        p.fill("#seed", "42")
        self.click("#apply", "configure")
        self.click("#start", "start")
        p.wait_for_function('state.status==="finished"', timeout=480000)
        original = self.state()
        source = original["attempt"]
        descriptor = next(r for r in original["records"] if r["id"] == source)
        checkpoint = descriptor["checkpoints"][3]
        frame = checkpoint["frame"]
        start = checkpoint["time"]
        (self.folder / "panel_level9_reference.json").write_text(
            json.dumps(original, indent=2)
        )
        paths = []
        outcomes = []
        for changed in (False, False, True):
            if p.locator("#attemptB").input_value() != source:
                p.select_option("#attemptB", source)
                p.wait_for_function(
                    "(id)=>!busy && state.playback?.id===id", arg=source, timeout=30000
                )
            p.select_option("#checkpointSelect", str(frame))
            p.wait_for_function(
                "(f)=>state.playback?.index===f", arg=frame, timeout=30000
            )
            restored = self.click("#restore", "restore")
            assert abs(restored["time"] - start) < 1e-10
            if changed:
                self.click("#vehicle", "vehicle_motion")
            self.click("#resume", "resume")
            p.wait_for_function("(t)=>state.time>=t+2.", arg=start, timeout=120000)
            self.click("#pause", "pause")
            saved = self.click("#saveRun", "save_run")
            paths.append(saved["path"])
            outcomes.append(self.state())

        def frames(path):
            with np.load(path, allow_pickle=False) as a:
                metadata = json.loads(str(a["metadata"]))
                states = a["states"].copy()
            return {
                round(t["time"], 8): (state, t)
                for t, state in zip(metadata["telemetry"], states, strict=True)
            }

        a, b, c = map(frames, paths)
        common = sorted(set(a) & set(b) & set(c))
        common = [t for t in common if start + 0.2 < t < start + 1.9]
        assert common
        error = max(float(np.max(np.abs(a[t][0] - b[t][0]))) for t in common)
        divergence = max(float(np.max(np.abs(a[t][0] - c[t][0]))) for t in common)
        assert error == 0 and divergence > 0
        self.results.append(
            dict(
                check="checkpoint",
                passed=True,
                source=source,
                checkpoint=checkpoint,
                branches=paths,
                compared_times=common,
                identical_state_max_error=error,
                changed_input_max_difference=divergence,
                original_preserved=any(
                    r["id"] == source for r in self.state()["records"]
                ),
                outcomes=[
                    {
                        k: o[k]
                        for k in (
                            "attempt",
                            "time",
                            "phase",
                            "events",
                            "tension",
                            "strain_relief",
                        )
                    }
                    for o in outcomes
                ],
            )
        )
        p.screenshot(path=str(self.folder / "checkpoint_after.png"), full_page=True)

    def placement_before(self):
        p = self.page
        before = self.state()["obstacles"]
        rect = p.locator("#orbitArea").bounding_box()
        p.mouse.click(
            rect["x"] + rect["width"] * 0.5, rect["y"] + rect["height"] * 0.65
        )
        p.wait_for_timeout(500)
        assert self.state()["obstacles"] == before
        assert p.locator("#placeGround").count() == 0
        self.results.append(
            dict(
                check="placement_before",
                no_placement_control=True,
                ground_click_did_not_place=True,
            )
        )
        p.screenshot(path=str(self.folder / "placement_before.png"), full_page=True)

    def load_file(self, path):
        with self.page.expect_response(
            lambda r: r.url.endswith("/api/configure") and r.request.method == "POST",
            timeout=120000,
        ) as response:
            self.page.locator("#loadLayout").set_input_files(str(Path(path).resolve()))
        assert response.value.status == 200, response.value.text()
        self.page.wait_for_function("!busy", timeout=120000)
        result = response.value.json()
        self.page.wait_for_function(
            "(id)=>state.attempt===id && lastImageAttempt===id",
            arg=result["attempt"],
            timeout=60000,
        )
        return result

    def world_click(self, x, y):
        point = self.page.evaluate(
            """([x,y])=>{const v=state.view_camera,a=imageArea(),r=[v.forward[1]*v.up[2]-v.forward[2]*v.up[1],v.forward[2]*v.up[0]-v.forward[0]*v.up[2],v.forward[0]*v.up[1]-v.forward[1]*v.up[0]],p=[x-v.pos[0],y-v.pos[1],-v.pos[2]],dot=(a,b)=>a.reduce((s,n,i)=>s+n*b[i],0),d=dot(p,v.forward),u=.5+(dot(p,r)*v.near/d-v.center)/v.width,w=(v.top-dot(p,v.up)*v.near/d)/(v.top-v.bottom);return [a.rect.x+a.left+u*a.width,a.rect.y+a.top+w*a.height];}""",
            [x, y],
        )
        p = self.page
        with p.expect_response(
            lambda r: (
                r.url.endswith("/api/preview_placement") and r.request.method == "POST"
            ),
            timeout=30000,
        ) as response:
            p.mouse.click(*point)
        assert response.value.status == 200, response.value.text()
        p.wait_for_function("!busy")
        return response.value.json()

    def placement(self):
        p = self.page
        for name, count in [("box_avoidance", 1), ("stone_slope", 2)]:
            with p.expect_download() as pending:
                p.locator(f'a[href="/sample-layouts/{name}.json"]').click()
            path = self.folder / (name + "_download.json")
            pending.value.save_as(path)
            self.load_file(path)
            assert len(self.state()["obstacles"]) == count
            assert not self.state()["config"]["moving_car"]
            self.results.append(
                dict(check="load_" + name, passed=True, obstacles=count)
            )
        p.wait_for_function(
            'state.view_camera && state.status==="ready"', timeout=30000
        )
        p.select_option("#manualList", index=0)
        p.click("#placeGround")
        preview = self.world_click(-2.8, 1.4)
        assert preview["valid"], preview
        assert abs(float(p.locator("#x").input_value()) + 2.8) < 0.02
        assert abs(float(p.locator("#y").input_value()) - 1.4) < 0.02
        p.screenshot(path=str(self.folder / "placement_preview.png"), full_page=True)
        self.click("#applyPlacement", "configure")
        assert abs(self.state()["obstacles"][0]["x"] + 2.8) < 0.02
        active = self.state()["attempt"]
        p.select_option("#manualList", index=0)
        pose = self.state()["pose"]
        bad = self.world_click(pose[0], pose[1])
        assert not bad["valid"] and p.locator("#applyPlacement").is_disabled()
        assert self.state()["attempt"] == active
        before = self.state()["view_camera"]["view"]["azimuth"]
        rect = p.locator("#orbitArea").bounding_box()
        p.mouse.move(rect["x"] + rect["width"] * 0.5, rect["y"] + rect["height"] * 0.5)
        p.mouse.down()
        p.mouse.move(
            rect["x"] + rect["width"] * 0.5 + 60,
            rect["y"] + rect["height"] * 0.5 + 12,
            steps=8,
        )
        p.mouse.up()
        p.wait_for_function(
            "(a)=>Math.abs(state.view_camera?.view.azimuth-a)>1",
            arg=before,
            timeout=30000,
        )
        self.results.append(
            dict(
                check="ground_placement",
                passed=True,
                minimum_edge_clearance_m=preview["required_clearance_m"],
                accepted=preview,
                rejected=bad,
                orbit_preserved=True,
            )
        )
        # The downloaded box must cause actual observation-driven adaptation.
        self.load_file(self.folder / "box_avoidance_download.json")
        self.click("#start", "start")
        p.wait_for_function("state.replans>0 || state.failure", timeout=60000)
        if self.state()["status"] == "running":
            self.click("#pause", "pause")
        observed = self.state()
        assert observed["ray_hits"] > 0
        assert observed["replans"] > 0, observed["failure"]
        self.results.append(dict(check="sample_avoidance", passed=True, state=observed))
        p.screenshot(
            path=str(self.folder / "sample_avoidance_browser.png"), full_page=True
        )
        self.click("#live", "live")
        source = json.loads((self.folder / "checkpoint_browser.json").read_text())[
            "checks"
        ][0]["source"]
        p.select_option("#attemptB", source)
        p.wait_for_function(
            "(id)=>!busy && state.playback?.id===id", arg=source, timeout=30000
        )
        assert p.locator("#restore").is_disabled()
        self.results.append(dict(check="disk_visual_only", passed=True, source=source))

    def camera_before(self):
        import numpy as np

        p = self.page
        reference = json.loads(
            (self.folder / "panel_level9_reference.json").read_text()
        )
        source = reference["attempt"]
        with np.load(
            self.folder / "browser" / "runs" / (source + ".npz"), allow_pickle=False
        ) as data:
            frames = json.loads(str(data["metadata"]))["telemetry"]
        frame = min(range(len(frames)), key=lambda i: abs(frames[i]["time"] - 24.2))
        p.select_option("#attemptB", source)
        p.wait_for_function(
            "(id)=>!busy && state.playback?.id===id", arg=source, timeout=30000
        )
        self.seek(frame)
        with p.expect_response(
            lambda r: r.url.endswith("/api/camera") and r.request.method == "POST"
        ) as response:
            p.select_option("#camera", "connector_cam")
        assert response.value.status == 200
        p.wait_for_function(
            '(i)=>lastImageCamera==="connector_cam" && lastImageIndex===String(i)',
            arg=frame,
            timeout=60000,
        )
        p.locator("#orbitArea").screenshot(
            path=str(self.folder / "connector_existing.png")
        )
        p.screenshot(
            path=str(self.folder / "connector_existing_browser.png"), full_page=True
        )
        self.results.append(
            dict(
                check="camera_before",
                source=source,
                frame=frame,
                time=frames[frame]["time"],
                alignment=frames[frame]["alignment"],
            )
        )

    def camera(self):
        import numpy as np

        p = self.page
        reference = json.loads(
            (self.folder / "panel_level9_reference.json").read_text()
        )
        source = reference["attempt"]
        with np.load(
            self.folder / "browser" / "runs" / (source + ".npz"), allow_pickle=False
        ) as data:
            frames = json.loads(str(data["metadata"]))["telemetry"]
        frame = min(range(len(frames)), key=lambda i: abs(frames[i]["time"] - 24.2))
        p.select_option("#attemptB", source)
        p.wait_for_function(
            "(id)=>!busy && state.playback?.id===id", arg=source, timeout=30000
        )
        self.seek(frame)
        with p.expect_response(
            lambda r: r.url.endswith("/api/camera") and r.request.method == "POST"
        ) as response:
            p.select_option("#camera", "connector_side")
        assert response.value.status == 200
        p.wait_for_function(
            '(i)=>lastImageCamera==="connector_side" && lastImageIndex===String(i)',
            arg=frame,
            timeout=60000,
        )
        p.locator("#orbitArea").screenshot(path=str(self.folder / "connector_side.png"))
        p.screenshot(
            path=str(self.folder / "connector_side_browser.png"), full_page=True
        )
        self.results.append(
            dict(
                check="operator_camera",
                source=source,
                frame=frame,
                time=frames[frame]["time"],
                original_camera_unchanged=True,
                approx_axial_tip_gap_mm=-frames[frame]["alignment"]["axial_mm"] - 36,
            )
        )

    def references(self):
        p = self.page
        for level in (1, 9):
            p.wait_for_function("!busy", timeout=30000)
            if p.locator("#camera").input_value() != "external":
                with p.expect_response(
                    lambda r: (
                        r.url.endswith("/api/camera") and r.request.method == "POST"
                    )
                ) as response:
                    p.select_option("#camera", "external")
                assert response.value.status == 200
                p.wait_for_function("!busy", timeout=30000)
            p.select_option("#level", str(level))
            p.select_option("#mode", "random")
            p.fill("#seed", "42")
            p.locator("#movingCar").check()
            p.locator("#randomStart").check()
            self.click("#apply", "configure")
            self.click("#start", "start")
            p.wait_for_function('state.status==="finished"', timeout=480000)
            final = self.state()
            assert final["phase"] == "seated", final["failure"]
            assert not any(final["cable"]["violation_episodes"].values())
            self.click("#saveRun", "save_run")
            (self.folder / f"panel_level{level}_final.json").write_text(
                json.dumps(final, indent=2)
            )
            p.screenshot(
                path=str(self.folder / f"panel_level{level}_final.png"), full_page=True
            )
            self.results.append(
                dict(
                    check=f"final_level_{level}_seed42",
                    passed=True,
                    time=final["time"],
                    attempt=final["attempt"],
                    events=final["events"],
                )
            )

    def final(self):
        self.editor()
        self.placement()
        self.references()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--stage", default="editor")
    parser.add_argument("--output", type=Path, default=Path("outputs/panel_revision"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 1500, "height": 1200}, accept_downloads=True
        )
        check = BrowserCheck(page, args.url, args.output)
        try:
            getattr(check, args.stage)()
        finally:
            report = dict(
                checks=check.results, requests=check.requests, page_errors=check.errors
            )
            (args.output / (args.stage + "_browser.json")).write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    dict(
                        stage=args.stage,
                        checks=[
                            {
                                k: v
                                for k, v in c.items()
                                if k in ("check", "passed", "time")
                            }
                            for c in check.results
                        ],
                        requests=len(check.requests),
                        page_errors=check.errors,
                    ),
                    indent=2,
                )
            )
            browser.close()


if __name__ == "__main__":
    main()
