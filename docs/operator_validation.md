# Operator controls, replay, cameras and parallel validation

All work remains local on `feature/charger-alignment`; no commits, pushes, merges or PRs.
Robot assets, the existing robot driver and insertion controller, connector geometry,
insertion tolerances, and 60 N / 45 N cable limits are preserved. Changes here concern
operator state/requests, placement validation, views, replay bookkeeping and experiment
orchestration. The existing planar docking-trim approximation is unchanged.

## Actual browser reproduction and fixes

Playwright with Chromium performs real DOM selections, clicks, file uploads/downloads,
mouse ground clicks, drags, timeline changes and Resume. It does not manipulate physics
through hidden browser-test endpoints. Read-only state and recorded files provide the
assertions. Evidence is under `outputs/panel_revision/`.

### Obstacle editor

The requested Level 9 / seed 42 sequence accepted boxes at (-1.70, 2.86) and (-1.70, 0.44).
The third box at (-2.40, 0.44) caused **POST /api/configure -> HTTP 400**, originally
`Obstacle intersects protected start/dock area`. Three entries remained in the draft,
while two remained active. Applying all three together over an empty scene can likewise
leave zero active objects; a rejected transactional build does not partially populate it.
A transport-level `Failed to fetch` was **not reproduced** on the direct WSL URL. HTTP
400 is not a fetch transport rejection; the report does not pretend they are identical.

Fixes:

- Separate draft count / NOT APPLIED state and active-physics count.
- Active scene level, seed, object count and attempt ID remain visible.
- Errors name the POST endpoint, HTTP status/reason, and scene retained. Transport errors
  explicitly report a connection problem and the retained scene.
- Commands publish committed scene state before acknowledgement; obsolete polls cannot
  replace a newer command result. Controls lock during pending requests.
- JPEGs carry scene/camera/playback-frame identities. The browser waits for a matching
  frame instead of showing an old scene's image as the newly applied layout.
- **Save active layout** saves only accepted active geometry and settings; it no longer
  silently rebuilds a pending invalid draft. Load restores the file's seed and scenario flags.
- Selecting Random mode sends no stale manual objects to the generator.

Browser clicks verified Add, rejected Build, Save/download, Load/upload, New seed and
Identical restart. A physically invalid new seed remains a rejection, not a silent reroll.
The exact path box (-1.70, 0.44) generated 24 ray returns and four occupied cells. The
controller stopped at 0.502 s with `no observed free route: Final docking corridor blocked`.
This is a real sensor-driven conservative planning stop, not a claimed avoidance success.
The less restrictive sample below triggers an actual replan.

### Checkpoint restore

Before the fix, the first browser restore resumed time from 6.50 to 6.86 s, but the original
attempt became non-restorable in the UI after branching despite retaining checkpoints.
The fix supports retained in-memory source attempts, restores their matching model/config/
scene context and controller data, and creates a separate branch. The original is saved
unchanged. Six records are cached; evicted or disk-loaded records are explicitly visual-only.

The live checkpoint selector lists available resumable checkpoints; disk recordings and
terminal-only checkpoints do not enable Restore. Selecting a checkpoint shows the recorded
state; Restore then Resume advances actual live physics. Restoring also synchronizes the
editor with the restored scene, rather than leaving another scene's settings in the form.

Browser verification finished Level 9 seed 42, then restored the **6.50 s** checkpoint into
three branches. At matching timestamps **6.7–8.3 s**, the two identical-input continuations
had exactly equal integration states, poses, phases, events, forces, contacts, route counts
and observations. In the third branch, a vehicle-motion request before Resume caused
motion detection/braking at **6.8 s**. At 8.3 s the unchanged branch was driving at
X=-1.358 m with 7.305 N cable tension; the changed branch was waiting for target motion at
X=-1.678 m with 4.513 N tension. This validates same-machine cases, not arbitrary
cross-platform or indefinitely long determinism.

Evidence: `checkpoint_browser.json`, `checkpoint_telemetry_comparison.json` and the three
paired NPZ/MJB recordings listed there. Restarting the Python server loses controller
checkpoints; disk playback remains available and is correctly labelled.

## Ground placement and sample layouts

Select Manual or Hybrid, select an obstacle, enable **Click ground**, then click a ground
pixel in the external orbit view. A translucent projected footprint and clearance values
appear. **Apply preview** rebuilds the scene; pending previews do not change physics.
A drag beyond four pixels still orbits. The pixel-to-ground mapping uses MuJoCo's actual
rendered camera/frustum and accounts for image letterboxing. Sky/out-of-bounds clicks and
invalid placements explain their rejection.

Minimum clearance is **0.15 m**, measured between conservative XY-projected geometry
edges, not object centres. Robot collision-geometry bounds and connector geometry supply
the protective envelopes. Sensor geometry supplies the buffer:

```
map half-cell diagonal: 0.12 / sqrt(2) = 0.08485 m
half inter-ray gap: 2.2 * sin((240 / (81-1)) / 2 degrees) = 0.05759 m
sum rounded up to centimetres: 0.15 m
```

This is an explicit conservative operator-placement rule, not a safety certification or
an assertion that the sensor resolves every thin/occluded obstacle. Reset geometry and
cable contacts are checked separately. Preview evaluates the displayed live pose; applying
revalidates the reset pose. No object is teleported into a running contact scene.

Ready-to-load files:

- `layouts/box_avoidance.json`: Level 9 / seed 42, one 0.44 m square box at (-1.70, 0.80).
  Automatic vehicle movement is disabled to isolate the obstacle. Through the panel's Load
  and Start controls it produced **22 ray returns, six occupied cells and one route replan**.
  The test pauses after the decision; a full avoidance mission is not claimed for this layout.
- `layouts/stone_slope.json`: the same seeded start, with a stone and a one-metre-footprint
  static wedge. The base avoids these shapes; terrain climbing is not simulated.

Both were downloaded from the panel and uploaded using its actual Load control. A stone
was moved by a ground click to (-2.80, 1.40), previewed and applied. A click overlapping the
robot yielded zero edge clearance, disabled Apply and retained the current scene. Dragging
still changed camera azimuth. Screenshots and JSON: `placement_browser.json`,
`placement_preview.png`, `sample_avoidance_browser.png`.

## Connector camera

At **24.2 s**, the existing `connector_cam` shows the socket but not its own physical plug
tip. Its pose and the robot RGB perception camera were left unchanged. A separate
**Connector side · operator** free view aims between the real `plug_tip` and `socket_face`
sites. At the browser viewport's actual size it shows socket on the left, plug on the right
and approximately 185 mm axial tip gap during the docking approach.

Open `outputs/panel_revision/connector_side.png` or `connector_side_browser.png`.
For interactive review:

```bash
.venv/bin/python scripts/run_control_panel.py --port 8770 --output outputs/panel_revision/browser
```

Select saved attempt `1794119334711774896` under Attempt B, scrub to **frame 237 / 24.2 s**,
and select **Connector side · operator**. This saved attempt is visual playback only.
`connector_existing.png` records the original view for comparison.

## Shared headless pipeline and concurrency

The old runner used `CourseEnv`, a privileged-map baseline. The current default is:

```
run_course_batch -> panel.Session -> build_scenario + AdaptiveMission
```

It uses identical fixed cable, pose randomization, actual ray observations, marker updates,
moving-car logic, supervision, telemetry and `tick(10)` physics/control stepping as the
panel worker. Headless runs omit only rendering and browser input. `--engine legacy`
explicitly selects the older baseline; historical results must not be conflated.
Each process has its own model/data/controller and case directory. Workers are bounded
from one to four; this evaluation uses two. Logs, scenario settings, results and recordings
are isolated per case. Wall and simulated-time limits remain enabled.

The required initial check ran Level 9 seeds **42 and 43 simultaneously**, then seed 42
alone. Both seated. Seed 42 matched the browser in configuration, event timings, final
41.002 s time and final saved integration state. Concurrent and solo seed 42 additionally
matched cable/contact telemetry exactly. See `repeatability_comparison.json`,
`concurrent_level9.json`, `solo_level9.json` and their separate case directories.

Repeat the all-level evaluation in WSL (choose a new output filename to retain earlier results):

```bash
cd /home/sebastianborisov676/Events/AgentVerse
.venv/bin/python scripts/run_course_batch.py --engine panel \
  --levels 1 2 3 4 5 6 7 8 9 10 --seeds 42 43 --workers 2 \
  --max-time 90 --wall-time 180 --output outputs/panel_repeat_all_levels.json
```

The evaluated command used `--output outputs/panel_revision/all_levels.json`.
All twenty full missions docked and seated, with no failures, obstacle contacts, contacts
classified unsafe by the existing supervisor, operational cable-limit violations or MuJoCo
warnings. Maximum operational tension was 26.03 N and strain-relief load 20.92 N.
Counts below are force-active contact-pair onsets, not impact counts. Expected socket
insertion, cable/floor and cable/pedestal contacts remain separately available in JSON.
Level 1 seed 43 had eight cable/base contact onsets (peak normal force 7.65 N), below cable
limits; the result is not advertised as entirely contact-free or hardware safety validation.

| Level | Seed | Docking | Obstacle contacts | Cable/robot | Unsafe robot | Cable violations | Failure |
|---|---|---|---:|---:|---:|---:|---|
| 1 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 1 | 43 | Seated | 0 | 8 | 0 | 0 | None |
| 2 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 2 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 3 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 3 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 4 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 4 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 5 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 5 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 6 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 6 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 7 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 7 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 8 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 8 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 9 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 9 | 43 | Seated | 0 | 0 | 0 | 0 | None |
| 10 | 42 | Seated | 0 | 0 | 0 | 0 | None |
| 10 | 43 | Seated | 0 | 0 | 0 | 0 | None |

Counts are force-active contact-pair onsets, not impact counts. Obstacle contacts combine robot/obstacle and cable/obstacle; expected socket insertion, floor and pedestal contacts are separate in JSON.

Primary machine-readable results: `outputs/panel_revision/all_levels.json`; per-case
`scenario.json`, `console.log`, `result.json` and `runs/` live beneath the matching
`all_levels/level_<level>_seed_<seed>_case_<index>/` directories.

## Validation and remaining limits

The closing checks rerun the full pytest suite, HTTP/render integration checks, actual
Chromium editor/placement controls, and fresh Level 1/Level 9 seed-42 browser missions.
The full suite passed **57 tests**; the HTTP/render checker passed **14 checks**. The final
Chromium sequence passed eight checks (editor, both layout Loads, ground placement,
observation-driven avoidance, disk-only restore labelling and both full references), with
29 control requests and no page errors. Fresh browser references seated at **36.846 s**
(Level 1) and **41.002 s** (Level 9). Both matched their headless configurations, event
timings and final integration states exactly on this machine.

Final outputs are `final_browser.json`, `panel_level1_final.json`, `panel_level9_final.json`,
`final_reference_comparison.json` and screenshots under `outputs/panel_revision/`;
the HTTP check writes `outputs/panel_http_check/result.json`.

For repeatable browser checks, install optional `requirements-browser.txt` and Chromium:

```bash
.venv/bin/python -m pip install -r requirements-browser.txt
PLAYWRIGHT_BROWSERS_PATH=/tmp/agentverse-browser .venv/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=/tmp/agentverse-browser .venv/bin/python scripts/verify_panel_browser.py --stage final
```

The check expects the local panel at `http://127.0.0.1:8770`; `--url` overrides it.
The earlier checkpoint evidence requires a live server while branching; disk files
intentionally cannot restore controller memory.

Remaining limits: the original transport error was not reproduced; rendering may use Mesa
software fallback; placement is conservative projected geometry; the path-box case can
stop on the planner's conservative final corridor; perception remains ideal rays/fiducials;
base/vehicle mechanics remain planar prototypes; and exact determinism is established only
for the tested same-machine comparisons. No cable, insertion or collision-safety thresholds
were relaxed to obtain passing results.


## Level 6 barrier editor follow-up

The original server on port 8765 was inspected without restarting or reloading it.
Its active Level 6 / seed 42 hybrid config has four manual entries plus twelve generated
obstacles. Its recorded error is `Obstacle intersects protected start/dock area`.
The 23-entry browser-only draft was inaccessible through the available server API; it
was not claimed recovered. `outputs/editor_barrier_fix/initial_8765.json` preserves the
active snapshot, not that draft.

The supplied slope (-1.54, 0.26), size 0.5, and barrier (-1.46, 0.27), size 0.35,
were independently submitted to `/api/configure` on an isolated panel. HTTP 400 reports
0.081 m centre separation against the existing conservative footprint rule of 1.107 m.
See `pair_request_response.json`. This confirms the pair conflicts, but does not identify
which of the inaccessible original draft entries first violated the protected dock rule.

The editor now checks the entire manual draft during ground preview, reports numbered
geometry failures beside Apply preview, selects and marks offending entries, and keeps
them available to Move/Remove. A failed Add retains one pending entry; repeated attempts
update it instead of appending duplicates. Apply preview adds its exact captured obstacle,
independent of the list selection. A successful build clears the preview; a failed build
retains it and the active scene. Save still exports only validated active physics.

Actual Chromium verification on port 8771 exercised Level 6 / seed 42 hybrid mode:
one click at (-1.46, 3.60), rotation -87 degrees, then Apply created exactly one barrier.
Save and file-input Load preserved it. A protected-dock Add at (-0.30, 0.10) produced
HTTP 400 with obstacle 14 identified inline; repeating Add retained 14 draft entries.
Removing that entry through the list and Remove button built successfully.
Screenshots, JSON responses and the exported layout are in `outputs/editor_barrier_fix/`.
`scripts/verify_barrier_editor.py` repeats these checks against an isolated panel on 8771.

Robot geometry, connector, controller, insertion tolerances and cable limits are unchanged.
The old user panel was intentionally left running with its browser draft untouched;
port 8771 serves the updated implementation for inspection in a separate tab.

An additional browser check loaded a draft with an earlier protected-dock violation,
then clicked the ground twice. Both `/api/preview_placement` responses identified
obstacle 14, Apply remained disabled, the offending list entry was selected/marked,
and the draft stayed at 14 entries. Evidence: `earlier_draft_preview.json` and
`earlier_draft_rejection.png`. No rejected previews were appended to the draft.

To preserve the original tab's still-open browser draft before any reload, its developer
console can copy a Load-compatible JSON object without changing the scene:

```javascript
copy(JSON.stringify({...config(), obstacles: structuredClone(manual)}, null, 2))
```

Save that clipboard text as a separate draft JSON file, then use Load in the updated
panel. Invalid layouts remain in the editor for correction and do not replace physics.
The ordinary Save active layout button intentionally cannot back up unapplied drafts.

Validation for this follow-up: the full previously collected suite passed **57 tests in
441.25 s**; the newly appended numbered-geometry regression passed separately (**1 test**).
Final collection contains 58 tests. Python compilation passed. Browser results and the
machine-readable summary are in `outputs/editor_barrier_fix/validation_summary.json`.
Modified in this follow-up: `src/autocharge/panel/static/app.js`,
`src/autocharge/panel/session.py`, `src/autocharge/training/course.py`,
`tests/test_course.py`, `scripts/verify_barrier_editor.py`, and this document.
