#!/usr/bin/env bash
set -euo pipefail

# Move to the project root
cd "$(dirname "$0")/.."

echo "Setting up Team SOS AutoCharger AI..."

# Check prerequisites
command -v python3 >/dev/null || {
    echo "Python 3 is required."
    exit 1
}

command -v git >/dev/null || {
    echo "Git is required."
    exit 1
}

# Create the virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate environment (Linux/macOS use bin/, Windows Git Bash uses Scripts/)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-all.txt

# Download robot models (Franka Panda + Google Robot) from MuJoCo Menagerie.
# The model files are used unmodified; AutoCharger adds its parts at load time.
MODEL_REPO="models/mujoco_menagerie"
MODELS="franka_emika_panda google_robot"

if [ ! -d "$MODEL_REPO/.git" ]; then
    mkdir -p models

    if [ -d "$MODEL_REPO" ]; then
        echo "Existing incomplete model directory: $MODEL_REPO (delete it and re-run)."
        exit 1
    fi

    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git \
      "$MODEL_REPO"
fi

# Idempotent: also adds google_robot to checkouts made by older versions of this script.
git -C "$MODEL_REPO" sparse-checkout set $MODELS

# Verify the completed installation
echo "Verifying Team SOS environment..."
python scripts/verify_setup.py

echo "Dependencies installed and model downloaded."
echo "Setup completed successfully."
