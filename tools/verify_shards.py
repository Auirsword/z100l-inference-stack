#!/usr/bin/env python3
"""Verify completed Maya-S shards as soon as they finish.

aria2 writes a `.aria2` control file next to an incomplete file and removes it
on completion, so "no control file and full size" marks a candidate.  Each
candidate is hashed once against download-plan.json and cached by mtime/size.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path("/data/models/deepseek-v41-flash-1017")
PLAN = ROOT / "download-plan.json"
REPORT = ROOT / "verify-report.json"
CACHE = ROOT / "verify-cache.json"


def load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sha256(path, chunk=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def base_jobs():
    plan = load(PLAN, {})
    return [job for job in plan.get("jobs", []) if job.get("group") == "base"]


def scan(verify):
    cache = load(CACHE, {})
    rows = []
    for job in base_jobs():
        path = ROOT / "base/Maya-S" / Path(job["name"]).name
        row = {"name": Path(job["name"]).name, "size": job["size"], "sha256": job["sha256"]}
        if not path.exists():
            row["status"] = "missing"
        else:
            actual = path.stat().st_size
            row["bytes"] = actual
            control = Path(str(path) + ".aria2")
            if control.exists() or actual != job["size"]:
                row["status"] = "downloading"
                row["percent"] = round(100.0 * actual / job["size"], 2) if job["size"] else 0.0
            else:
                key = "%s:%d:%d" % (path, actual, path.stat().st_mtime_ns)
                if cache.get("key") == key and cache.get("digest"):
                    digest = cache["digest"]
                else:
                    digest = sha256(path) if verify else None
                    if digest:
                        cache = {"key": key, "path": str(path), "digest": digest}
                        CACHE.write_text(json.dumps(cache), encoding="utf-8")
                if digest is None:
                    row["status"] = "unverified"
                elif digest == job["sha256"]:
                    row["status"] = "ok"
                else:
                    row["status"] = "mismatch"
                    row["actual_sha256"] = digest
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int, default=0, help="seconds between passes, 0 = once")
    parser.add_argument("--no-verify", action="store_true", help="only classify, do not hash")
    args = parser.parse_args()
    while True:
        rows = scan(not args.no_verify)
        summary = {}
        for row in rows:
            summary[row["status"]] = summary.get(row["status"], 0) + 1
        report = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summary": summary, "files": rows}
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"summary": summary,
                          "files": [(r["name"][-26:], r["status"]) for r in rows]}, ensure_ascii=False),
              flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
