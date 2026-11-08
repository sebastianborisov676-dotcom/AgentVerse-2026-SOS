#!/usr/bin/env python3
"""Run the local robotics operator panel at http://127.0.0.1:8765."""

import argparse
import os
import pathlib
import sys

# Select before importing MuJoCo. Override explicitly with MUJOCO_GL=glfw/osmesa if needed.
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from autocharge.panel.server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default="outputs/panel")
    args = parser.parse_args()
    server = make_server(port=args.port, output=args.output)
    print(f"Local control panel: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.worker.close()


if __name__ == "__main__":
    main()
