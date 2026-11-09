#!/usr/bin/env python3
"""Switch the production API backend between the 4-card PP config and the
single-card three-tier (VRAM -> RAM -> GGUF/disk) baseline."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NAME = "strata-serve"
PORT = 8080
CONFIGS = {
    "single": Path("/data/z100l-strata-port/strata/serve-z100l-single.json"),
    "pp4": Path("/data/z100l-strata-port/strata/serve-z100l-pp4.json"),
}


def start(mode):
    config = CONFIGS[mode]
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
        "--config", str(config), "--port", str(PORT),
    ]
    print(subprocess.run(command, capture_output=True, text=True).stdout.strip())


def wait_ready(timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=3):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return False


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    start(mode)
    print(json.dumps({"mode": mode, "ready": wait_ready(900)}))
