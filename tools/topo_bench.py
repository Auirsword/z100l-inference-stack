#!/usr/bin/env python3
"""Run the DS4 engine in several topologies and record single-stream decode.

Each topology starts the serve container with its own config, waits until
/v1/models answers (or the container dies), runs a fixed 64-token completion
twice, then stops the container.  Results go to /data/build/ds4-topo/results.jsonl
together with the engine log tail (CPU lane %, expert cache hit).
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/data/build/ds4-topo")
ROOT.mkdir(parents=True, exist_ok=True)
REPO = "/data/z100l-strata-port"
PACK = "/models/deepseek-v41-flash-1017/patched/Maya-S/maya-ds4"
BASE_PORT = 18100

BASE_ARGS = [
    "--pack", PACK, "--max-context", "32768", "--threads", "off",
    "--prefill-chunk", "0", "--reserve-mib", "3072", "--dspark", "none",
    "--vision", "off", "--mmap-experts", "auto", "--parallel", "1", "--suffix-draft", "0",
]

# name, gpus, extra engine args, extra env
TOPOLOGIES = [
    ("pp4", "0,1,2,3", ["--tp", "1"], {"DS4_PROF": "1"}),
    ("tp4", "0,1,2,3", ["--tp", "4"], {"DS4_TP_GGUF_UNEVEN": "1", "DS4_PROF": "1"}),
    ("ep4", "0,1,2,3", ["--tp", "1", "--ep", "4"], {"DS4_SYNC": "1", "DS4_PROF": "1"}),
    ("tp2", "0,1", ["--tp", "2"], {"DS4_TP_GGUF_UNEVEN": "1"}),
    ("tp2pp2", "0,1,2,3", ["--tp", "2"], {}),
]
PROFILE = "/work/ds4-experiments/pp4/expert-profile.bin"


def config_for(name, gpus, extra, port, extra_env):
    (ROOT / name).mkdir(parents=True, exist_ok=True)
    args = list(BASE_ARGS) + ["--gpus", gpus] + extra
    if Path("/data/build/ds4-experiments/pp4/expert-profile.bin").exists():
        args += ["--expert-profile", PROFILE]
    env = {
        "STRATA_DS4_SLOT_KEEP": "1",
        "STRATA_DS4_SLOT_DIR": "/work/ds4-topo/%s/slots" % name,
        "STRATA_DS4_SLOT_CAP": "1",
        "STRATA_DS4_SLOT_RAM_GB": "1",
        "STRATA_DS4_SLOT_GB": "16",
    }
    env.update(extra_env)
    payload = {
        "exe": "/work/maya-ds4-z100l/build-906/maya-ds4",
        "args": args,
        "cwd": "/work/maya-ds4-z100l",
        "tokenizer": PACK + "/tokenizer",
        "backend": "hip",
        "model_name": "LLM-server-flash",
        "gpu": [int(x) for x in gpus.split(",")],
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "reasoning_effort": "high",
        "lib_dirs": [],
        "host": "127.0.0.1",
        "port": port,
        "log": "/work/ds4-topo/%s/engine.log" % name,
        "env": env,
    }
    path = ROOT / ("%s.json" % name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def start(name, config, port):
    subprocess.run(["docker", "rm", "-f", "ds4-topo-" + name], capture_output=True)
    command = [
        "docker", "run", "-d", "--name", "ds4-topo-" + name,
        "--network=host", "--ipc=host", "--shm-size=32G",
        "--device=/dev/kfd", "--device=/dev/mkfd", "--device=/dev/dri",
        "-v", "/opt/hyhal:/opt/hyhal:ro", "-v", "/opt/dtk:/opt/dtk:ro",
        "-v", "/data/build:/work", "-v", "/data/models:/models:ro",
        "-v", REPO + ":/repo:ro",
        "-w", "/work/maya-ds4-z100l",
        "--group-add", "video", "--ulimit", "memlock=-1",
        "--security-opt", "seccomp=unconfined", "--security-opt", "apparmor=unconfined",
        "z100l-dev:latest",
        "/work/ds4-venv/bin/python", "-m", "serve.server", "--engine", "strata",
        "--config", config.replace("/data/build", "/work"),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def container_running(name):
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "ds4-topo-" + name],
                         capture_output=True, text=True).stdout.strip()
    return out == "true"


def wait_ready(port, name, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=3):
                return True
        except Exception:  # noqa: BLE001
            if not container_running(name):
                return False
            time.sleep(5)
    return False


def bench(port, tokens=64, runs=5):
    out = subprocess.run([sys.executable, "/data/z100l-strata-port/tools/ds4_bench.py", str(port), str(tokens), str(runs)],
                         capture_output=True, text=True)
    return out.stdout.strip(), out.stderr.strip()[-300:]


def engine_tail(name, lines=8):
    path = ROOT / name / "engine.log"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [line for line in text if re.search(r"done:|CPU lane|expert cache", line)]
    return keep[-lines:]


def main():
    names = sys.argv[1:] or [t[0] for t in TOPOLOGIES]
    for index, (name, gpus, extra, extra_env) in enumerate(TOPOLOGIES):
        if name not in names:
            continue
        port = BASE_PORT + index
        print("=== %s (gpus=%s %s)" % (name, gpus, " ".join(extra)), flush=True)
        config = config_for(name, gpus, extra, port, extra_env)
        start(name, config, port)
        ready = wait_ready(port, name)
        result = {"topology": name, "gpus": gpus, "extra": extra, "port": port,
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ready": ready}
        if ready:
            stdout, stderr = bench(port)
            result["bench"] = stdout
            result["stderr"] = stderr
        result["engine"] = engine_tail(name)
        with (ROOT / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(name, "ready=%s" % ready, result.get("bench", "")[:200], flush=True)
        subprocess.run(["docker", "stop", "-t", "20", "ds4-topo-" + name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", "ds4-topo-" + name], capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
