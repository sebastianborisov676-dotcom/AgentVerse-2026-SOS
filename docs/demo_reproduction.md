# Verified Level 11 local reference

The simulation source, generated scene definitions, panel assets, layouts and tests
are versioned. Downloaded third-party models and generated recordings are separate
local artifacts, excluded by `.gitignore`.

## External models

The verified model checkout is Google DeepMind MuJoCo Menagerie revision
`98a70c569c4c1b6da467a20012f766ee1f8dc521`, with no local changes.
Restore it on a fresh checkout using:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie checkout 98a70c569c4c1b6da467a20012f766ee1f8dc521
git -C models/mujoco_menagerie sparse-checkout set franka_emika_panda google_robot
```

Do not replace an existing modified model checkout. The model source and licences
remain in that external checkout; application-specific scene geometry is generated
by the versioned Python source.

## Verified environment and launch

Observed versions: Python 3.14.4, MuJoCo 4.0.0, NumPy 2.5.4, Pillow 13.0.0,
pytest 9.2.0 and optional Playwright 1.64.1. The existing panel requirements specify
Pillow below 13, so installing those ranges does not exactly reproduce the observed
environment. For the exact observed package versions in an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install mujoco==4.0.0 numpy==2.5.4 Pillow==13.0.0 pytest==9.2.0 playwright==1.64.1
.venv/bin/python scripts/run_control_panel.py --port 8772 --output outputs/master_demo
```

Open http://127.0.0.1:8772 and select Level 11, seed 42, Random mode,
Randomize start enabled and Car challenge enabled; Build scenario, then Start.
The reference playback reaches SEATED at 62.756 simulated seconds. See
[master challenge](master_challenge.md) for event ordering and simulation limits.

## Preserved reference artifacts (not committed)

Both files remain under `outputs/master_demo/runs/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `attempt_1794134385345455365.npz` | 6892786 | `fdf1c1ad8bc0224c773f9c7857ce32d7417a0e5fa9a56873d3d2c31b31a21abe` |
| `attempt_1794134385345455365.mjb` | 22496079 | `ac55f71bdf77a505e6dc0d6e1e397cdbc6a78665b860ac115d1ed855bd96febe` |

A fresh clone can re-simulate the demo but does not contain these playback files.
The browser verification script also expects local `outputs/master_demo/selected_demo.json`
and its referenced recording; retain the existing output directory to repeat that check.

## Existing final verification

All 65 tests passed in four isolated groups (17, 16, 16, 16), each exit code zero.
Saved results: `outputs/master_tests_final.json`; detailed logs:
`outputs/master_tests_group_0.log` through `outputs/master_tests_group_3.log`.
Python compilation and the Level 11 Chromium verification passed.
No simulation code was changed while preparing this commit and these checks were
not rerun. The earlier 63-pass/1-failure regression is superseded by the final run.
Mission evaluation completed 5 of 7 episodes; Level 6 seed 42 and Level 11 seed 43
stopped at cable-load limits. Full local evidence is `outputs/master_verification.json`.
