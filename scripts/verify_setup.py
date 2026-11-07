
"""
Team SOS — AutoCharger AI
Initial environment verification.

Checks:
1. Python and package versions
2. Franka Panda model files
3. MuJoCo model loading
4. Robot joints and actuators
5. Basic physics simulation
"""

import sys
from pathlib import Path

import mujoco
import numpy as np


# Locate the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the Franka Panda model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)


def verify_dependencies():
    """Check the installed Python packages."""

    print("\n[1] Checking dependencies")

    print(f"Python: {sys.version.split()[0]}")
    print(f"MuJoCo: {mujoco.__version__}")
    print(f"NumPy: {np.__version__}")

    print("PASS: Dependencies imported successfully")


def verify_model():
    """Check that the Franka Panda model exists."""

    print("\n[2] Checking robot model")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Franka Panda scene not found: {MODEL_PATH}\n"
            "Run: bash scripts/setup.sh"
        )

    print(f"Model found: {MODEL_PATH}")
    print("PASS: Robot model exists")


def load_robot():
    """Load the model and initialise simulation data."""

    print("\n[3] Loading MuJoCo robot")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("PASS: Robot loaded successfully")

    return model, data


def verify_robot_structure(model):
    """Inspect the robot's joints and actuators."""

    print("\n[4] Inspecting robot structure")

    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Degrees of freedom: {model.nv}")

    print("\nAvailable actuators:")

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        print(f"  {actuator_id}: {name}")

    if model.nu == 0:
        raise RuntimeError(
            "No actuators found in the robot model."
        )

    print("PASS: Robot has controllable actuators")


def verify_physics(model, data):
    """Run one physics step to check the simulator."""

    print("\n[5] Testing physics simulation")

    initial_time = data.time

    mujoco.mj_step(model, data)

    if data.time <= initial_time:
        raise RuntimeError(
            "Simulation time did not advance."
        )

    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError(
            "Invalid robot joint positions detected."
        )

    if not np.all(np.isfinite(data.qvel)):
        raise RuntimeError(
            "Invalid robot joint velocities detected."
        )

    print(f"Simulation time: {data.time:.4f} seconds")
    print("PASS: Physics simulation is working")


def main():
    """Run all environment verification checks."""

    print("=" * 50)
    print("TEAM SOS — AUTOCHARGER AI")
    print("Environment Verification")
    print("=" * 50)

    try:
        verify_dependencies()
        verify_model()

        model, data = load_robot()

        verify_robot_structure(model)
        verify_physics(model, data)

    except Exception as error:
        print(f"\nFAIL: {error}")
        return 1

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("Environment ready for development")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
