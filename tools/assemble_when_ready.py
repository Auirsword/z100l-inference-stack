#!/usr/bin/env python3
"""Assemble the Maya-S derived package as soon as the six base shards are complete.

Policy (handoff §12): GGUF shards are usable before their SHA-256 is checked.
This watcher only waits for full file size and the absence of `.aria2` control
files, then runs the pinned patch into a NEW timestamped directory.  It never
deletes or overwrites anything; a failed run goes to `<name>.partial` and the
next attempt uses a new name.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/models/deepseek-v41-flash-1017")
PLAN = ROOT / "download-plan.json"
BASE = ROOT / "base/Maya-S"
PATCHED = ROOT / "patched"
MARKER = ROOT / "assemble-state.json"
LOG = Path("/data/build/tuning/ds4-assemble.log")
PATCH_SCRIPT = "/repo/strata/ds4-z100l/patch_maya_s.py"


def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), message)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def shards():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    return [job for job in plan["jobs"] if job.get("group") == "base"]


def all_complete():
    for job in shards():
        path = BASE / Path(job["name"]).name
        if not path.exists():
            return False, "%s missing" % path.name
        if Path(str(path) + ".aria2").exists():
            return False, "%s still has a control file" % path.name
        size = path.stat().st_size
        if size != job["size"]:
            return False, "%s %d/%d" % (path.name, size, job["size"])
    return True, "six shards complete"


def assemble():
    state = json.loads(MARKER.read_text()) if MARKER.exists() else {}
    if state.get("status") == "done":
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ")
    out = PATCHED / ("Maya-S-" + stamp)
    partial = PATCHED / ("Maya-S-" + stamp + ".partial")
    PATCHED.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm", "--name", "ds4-patch", "--network=none", "--ipc=host",
        "--shm-size=8G", "-v", "/data/build:/work", "-v", "/data/models:/models",
        "-v", "/data/z100l-strata-port:/repo:ro", "z100l-dev:latest",
        "/work/ds4-venv/bin/python", PATCH_SCRIPT,
        "--base", "/models/deepseek-v41-flash-1017/base/Maya-S",
        "--patch-file", "/models/deepseek-v41-flash-1017/Maya/DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S.gguf",
        "--patch-script", "/models/deepseek-v41-flash-1017/Maya/patch-Maya-DeepSeek-V4.1-Flash-1017.py",
        "--out", "/models/deepseek-v41-flash-1017/patched/" + partial.name,
        "--maya-root", "/work/maya-ds4-z100l",
        "--drafter", "/models/deepseek-v41-flash-1017/dspark/dspark.json",
        "--python", "/work/ds4-venv/bin/python",
    ]
    log("assembling into %s" % out)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        MARKER.write_text(json.dumps({"status": "failed", "at": stamp,
                                      "error": (result.stderr or "")[-2000:]}), encoding="utf-8")
        log("assembly failed rc=%d: %s" % (result.returncode, (result.stderr or "")[-500:]))
        return
    if partial.exists():
        partial.rename(out)
    MARKER.write_text(json.dumps({"status": "done", "at": stamp, "out": str(out)}), encoding="utf-8")
    log("assembled %s" % out)


def main():
    once = "--once" in sys.argv
    while True:
        ready, detail = all_complete()
        if ready:
            log("all shards complete, assembling")
            assemble()
            return 0
        log("waiting: " + detail)
        if once:
            return 1
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
