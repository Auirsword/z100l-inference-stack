#!/usr/bin/env python3
"""DS4 VRAM / OOM guard.

Watches, every few seconds, what each GPU of the running DS4 instance has free (the engine's own /metrics, no NVML
needed) and alarms before an allocation can fail:

    free < GUARD_WARN_MIB   -> WARN   (the reserve is being eaten; something allocates after start)
    free < GUARD_CRIT_MIB   -> CRIT   (an allocation is likely to fail: OOM)
    free == 0 / card gone   -> CRIT
    /metrics unreachable N  -> CRIT   (the engine died or hangs)
    "no VRAM"/"out of memory"/"cudaMalloc failed" in the container's log -> CRIT

State (one JSON a line, every SAMPLE_S seconds) -> /data/build/vram-guard.jsonl
Alerts (only when state changes)               -> /var/log/ds4-vram-guard.log  (+ syslog, tag ds4-vram-guard)

Run as ds4-vram-guard.service.  Nothing here touches the engine: one small HTTP GET every GUARD_INTERVAL seconds.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

SERVE = "/data/z100l-strata-port/strata/ds4-z100l/serve"
STATE = os.environ.get("GUARD_STATE", "/data/build/vram-guard.jsonl")
ALERTS = os.environ.get("GUARD_ALERTS", "/var/log/ds4-vram-guard.log")
INTERVAL = float(os.environ.get("GUARD_INTERVAL", "5"))
SAMPLE_S = float(os.environ.get("GUARD_SAMPLE_S", "30"))
WARN_MIB = int(os.environ.get("GUARD_WARN_MIB", "512"))
CRIT_MIB = int(os.environ.get("GUARD_CRIT_MIB", "192"))
MISS_LIMIT = int(os.environ.get("GUARD_MISS_LIMIT", "6"))
CONTAINER = os.environ.get("GUARD_CONTAINER", "ds4-serve")
MB = 1 << 20
KEY = None


def key():
    global KEY
    if KEY is None:
        try:
            KEY = open(SERVE + "/API_KEY.txt").read().strip()
        except OSError:
            KEY = ""
    return KEY


def metrics():
    req = urllib.request.Request("http://127.0.0.1:18100/metrics",
                                 headers={"Authorization": "Bearer " + key()})
    with urllib.request.urlopen(req, timeout=6) as r:
        return json.loads(r.read().decode())


def reserve_mib():
    """the reserve the running instance was started with (--reserve-mib), 0 when it cannot be read"""
    try:
        cfg = os.path.realpath(SERVE + "/serve.json")
        d = json.load(open(cfg))
        a = d.get("args", [])
        for i, x in enumerate(a):
            if x == "--reserve-mib" and i + 1 < len(a):
                return int(a[i + 1])
    except Exception:
        pass
    return 0


def say(level, msg):
    line = "%s %-5s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), level, msg)
    print(line, flush=True)
    try:
        with open(ALERTS, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    try:
        subprocess.run(["logger", "-t", "ds4-vram-guard", "-p",
                        "user.%s" % ("err" if level != "WARN" else "warning"), msg], check=False)
    except Exception:
        pass


def service_uptime_s():
    """seconds since the unit last became active (0 when it cannot be read)"""
    try:
        out = subprocess.run(["systemctl", "show", "-p", "ActiveEnterTimestampMonotonic", "--value", "ds4-serve"],
                             capture_output=True, text=True).stdout.strip()
        us = int(out)
        if us <= 0:
            return 0
        up = float(open("/proc/uptime").read().split()[0])
        return max(0.0, up - us / 1e6)
    except Exception:
        return 0


def service_active():
    try:
        return __import__("subprocess").run(["systemctl", "is-active", "ds4-serve"],
                                            capture_output=True, text=True).stdout.strip() == "active"
    except Exception:
        return False


def log_errors():
    """the container's own log, last minute, for allocation failures"""
    try:
        out = subprocess.run(["docker", "logs", "--since", "90s", CONTAINER],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    txt = (out.stdout or "") + (out.stderr or "")
    hits = []
    for pat in ("no VRAM", "out of memory", "OutOfMemory", "cudaMalloc", "hipErrorMemoryAllocation",
                "a launch failed", "cannot allocate"):
        for line in txt.splitlines():
            if pat.lower() in line.lower():
                hits.append(line.strip()[:180])
    return hits[:4]


def main():
    res = reserve_mib()
    mins = {}          # card -> min free seen since this process started
    last = {}          # card -> last state ("ok"/"warn"/"crit")
    miss = 0
    ever_ok = False    # (a restart or a load gives "connection refused" for minutes: that is not a hang)
    last_sample = 0.0
    say("INFO", "guard up: interval %.0fs, WARN %d MiB, CRIT %d MiB, instance reserve %d MiB, state %s"
        % (INTERVAL, WARN_MIB, CRIT_MIB, res, STATE))
    while True:
        try:
            m = metrics()
            if miss >= MISS_LIMIT or not ever_ok:
                say("INFO", "engine answers again" if ever_ok else "engine answers (first time since start)")
            miss = 0
            ever_ok = True
        except Exception as e:
            miss += 1
            up = service_active()
            if not up:
                if miss == 1:
                    say("INFO", "engine stopped (service inactive) - waiting")
                ever_ok = False
            elif miss == MISS_LIMIT:
                up = service_uptime_s()
                if up < 900:   # (a restart: the engine loads for minutes before /metrics answers)
                    say("INFO", "engine still loading (%.0f s since the unit started)" % up)
                else:
                    say("CRIT", "engine unreachable %d times while the service is active: %s" % (miss, str(e)[:120]))
            time.sleep(INTERVAL)
            continue
        cards = m.get("gpu_map") or []
        now = time.time()
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "reserve_mib": res, "gpus": {}}
        for g in cards:
            dev = g.get("dev")
            free = int(g.get("free") or 0)
            fmib = free // MB
            mins[dev] = min(mins.get(dev, fmib), fmib)
            st = "crit" if fmib < CRIT_MIB else ("warn" if fmib < WARN_MIB else "ok")
            row["gpus"][dev] = {"free_mib": fmib, "min_mib": mins[dev], "state": st,
                                "slots": g.get("slots"), "pool_mib": int((g.get("parts") or {}).get("pool", 0)) // MB}
            if st != last.get(dev):
                if st == "crit":
                    say("CRIT", "CUDA%s free %d MiB (< %d): an allocation will fail - OOM" % (dev, fmib, CRIT_MIB))
                elif st == "warn":
                    say("WARN", "CUDA%s free %d MiB (< %d): the reserve is being eaten" % (dev, fmib, WARN_MIB))
                else:
                    say("INFO", "CUDA%s back to %d MiB free" % (dev, fmib))
                last[dev] = st
        if res and not any(v["state"] != "ok" for v in row["gpus"].values()):
            # the invariant: free stays at the reserve on every card (the helper cards keep their 768 MiB margin)
            low = min((v["free_mib"] for v in row["gpus"].values()), default=0)
            if low < res - 768:
                say("WARN", "free %d MiB is below the reserve %d MiB - 768: something allocates after start" % (low, res))
        errs = log_errors()
        if errs:
            say("CRIT", "allocation errors in %s: %s" % (CONTAINER, " | ".join(errs)))
        if now - last_sample >= SAMPLE_S:
            last_sample = now
            try:
                with open(STATE, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
