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

# Activate environment
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-all.txt

# Download Franka Panda model
MODEL_REPO="models/mujoco_menagerie"
SCENE="$MODEL_REPO/franka_emika_panda/scene.xml"

if [ ! -f "$SCENE" ]; then
    mkdir -p models

    if [ -d "$MODEL_REPO" ]; then
        echo "Existing incomplete model directory."
        exit 1
    fi

    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git \
      "$MODEL_REPO"

    git -C "$MODEL_REPO" sparse-checkout set franka_emika_panda
fi

# Verify the completed installation
echo "Verifying Team SOS environment..."
python scripts/verify_setup.py

echo "Dependencies installed and model downloaded."
echo "Setup completed successfully."
