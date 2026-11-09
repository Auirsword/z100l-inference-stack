#!/usr/bin/env python3
"""Start the production Strata serve container from strata/serve-z100l.json.

Keeps the model in VRAM with the whole KV resident (no KV streaming), single
stream (parallel 1, other requests queue), and enables the slot save/restore
endpoint so the live session can be written to disk and read back.
"""
import subprocess
import sys
import time
import urllib.request

NAME = "strata-serve"
PORT = 8080
import os
CONFIG = os.environ.get("STRATA_CONFIG", "/data/z100l-strata-port/strata/serve-z100l.json")
SAVE_DIR = "/work/sessions"

subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
command = [
    "docker", "run", "-d", "--name", NAME, "--restart", "unless-stopped",
    "--network=host", "--device=/dev/kfd", "--device=/dev/mkfd", "--device=/dev/dri",
    "--group-add", "video", "--security-opt", "seccomp=unconfined",
    "--security-opt", "apparmor=unconfined", "--ulimit", "memlock=-1",
    "--ipc=host", "--shm-size=16G",
    "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
    "-v", "/data/build:/work", "-v", "/data/models:/models",
    "-v", "/data/z100l-strata-port:/data/z100l-strata-port:ro",
    "-v", "/data/build/conversations:/conversations",
    "-w", "/work/Strata", "z100l-dev:latest",
    "/work/venv-strata/bin/python", "-m", "serve.server", "--engine", "strata",
    "--config", CONFIG, "--port", str(PORT), "--slot-save-path", SAVE_DIR,
]
print(subprocess.run(command, capture_output=True, text=True).stdout.strip())

deadline = time.monotonic() + 900
ready = False
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=3):
            ready = True
            break
    except Exception:  # noqa: BLE001
        time.sleep(3)
print("ready", ready)
sys.exit(0 if ready else 1)
