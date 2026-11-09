#!/usr/bin/env python3
"""Create the production 4-card PP serve config and (re)start the serve container."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/data/build/tuning")
import qwen_pp  # noqa: E402

CONFIG = Path("/data/z100l-strata-port/strata/serve-z100l-pp4.json")
PORT = 8080
NAME = "strata-serve"


def build(context, kv_resident, parallel):
    payload = qwen_pp.build_config(CONFIG, PORT, [0, 1, 2, 3], context, kv_resident, 4, True, 2)
    payload["parallel"] = parallel
    payload["model_name"] = "LLM-server-flash"
    payload["aliases"] = ["qwen3.8-flash-next-iq3s-z100l"]
    CONFIG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def start():
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
        "--config", str(CONFIG), "--port", str(PORT),
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
    context = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
    kv = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
    build(context, kv, 4)
    start()
    ready = wait_ready(600)
    print(json.dumps({"ready": ready, "context": context, "kv_resident": kv}))
