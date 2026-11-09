#!/usr/bin/env bash
set -euo pipefail
root=${1:?new output directory required}
probe=${PROBE:-/data/build/ds4-fused-review/hip/fused_ar_probe}
sequences=${SEQUENCES:-8}
wire_fp16=${WIRE_FP16:-0}
script_dir=$(cd "$(dirname "$0")" && pwd)
case "$sequences" in
  ''|*[!0-9]*) printf 'SEQUENCES must be an integer in [1,256]\n' >&2; exit 2 ;;
esac
if [ "$sequences" -lt 1 ] || [ "$sequences" -gt 256 ]; then
  printf 'SEQUENCES must be in [1,256]\n' >&2
  exit 2
fi
case "$wire_fp16" in
  0|1) ;;
  *) printf 'WIRE_FP16 must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${QUEUED:-0}" in
  0|1) ;;
  *) printf 'QUEUED must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${MEMBERS:-all}" in
  all) modes=(2 4) ;;
  2|4) modes=("$MEMBERS") ;;
  *) printf 'MEMBERS must be all, 2 or 4\n' >&2; exit 2 ;;
esac
case "${TIMEOUT_CYCLES:-}" in
  ''|*[!0-9]*) if [ -n "${TIMEOUT_CYCLES:-}" ]; then printf 'TIMEOUT_CYCLES must be a positive integer\n' >&2; exit 2; fi ;;
  0) printf 'TIMEOUT_CYCLES must be a positive integer\n' >&2; exit 2 ;;
esac
test ! -e "$root"
test -x "$probe"
command -v fuser >/dev/null
if fuser /dev/kfd >/dev/null 2>&1; then
  printf 'refusing fused AR probe: another process has /dev/kfd open\n' >&2
  fuser -v /dev/kfd >&2 || true
  exit 3
fi
mkdir -p "$root"
test ! -e "$root/thermal.stop"
sha256sum "$probe" > "$root/probe.sha256"
sha256sum "$script_dir"/fused_ar.{hpp,cu} "$script_dir"/fused_ar_probe.cu > "$root/source.sha256"
printf 'fused-ar-probe\n' > "$root/phase.txt"
python3 /data/z100l-strata-port/tuning/quad_gpu/telemetry.py \
  --out "$root/telemetry.csv" --seconds 86400 --interval 1 --phase-file "$root/phase.txt" \
  --stop-file "$root/thermal.stop" --fan-state-file /run/z100l-fan-controller.json \
  > "$root/telemetry.log" 2>&1 &
monitor=$!
trap 'kill -TERM "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
for members in "${modes[@]}"; do
  phase=tp$members
  out=$root/$phase
  mkdir "$out"
  printf '%s\n' "$phase" > "$root/phase.txt"
  args=(--json /work/ds4-fused-output/report.json --sequences "$sequences")
  if [ "${QUEUED:-0}" = 1 ]; then args+=(--queued --skew-cycles 0); fi
  if [ -n "${TIMEOUT_CYCLES:-}" ]; then args+=(--timeout-cycles "$TIMEOUT_CYCLES"); fi
  if [ "$wire_fp16" = 1 ]; then
    args+=(--fp16-wire)
  fi
  if [ "$members" = 2 ]; then
    args+=(--device 0=0000:03:00.0 --device 1=0000:43:00.0)
  else
    args+=(--device 0=0000:03:00.0 --device 1=0000:43:00.0 --device 2=0000:83:00.0 --device 3=0000:c3:00.0)
  fi
  test ! -e "$root/thermal.stop"
  cat /sys/devices/system/edac/mc/mc0/ue_count > "$out/ue-before.txt"
  test "$(<"$out/ue-before.txt")" = 0
  cat /sys/devices/system/edac/mc/mc0/ce_count > "$out/ce-before.txt"
  date -Is > "$out/started.txt"
  container=ds4-fused-ar-$phase
  set +e
  timeout -s INT 300 docker run --rm --name "$container" --network=none --ipc=host \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v /opt/dtk:/opt/dtk:ro \
    -v /data/build:/work -v "$out:/work/ds4-fused-output" -v "$probe:$probe:ro" \
    --group-add video --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    z100l-dev:latest bash -lc 'source /opt/dtk/env.sh && exec "$@"' _ "$probe" "${args[@]}" \
    > "$out/stdout.json" 2> "$out/trace.log"
  status=$?
  set -e
  cat /sys/devices/system/edac/mc/mc0/ue_count > "$out/ue-after.txt"
  cat /sys/devices/system/edac/mc/mc0/ce_count > "$out/ce-after.txt"
  date -Is > "$out/finished.txt"
  printf '%s\n' "$status" > "$out/exit-status.txt"
  if [ "$status" != 0 ]; then
    printf 'fused AR %s failed with status %s; evidence retained in %s\n' "$phase" "$status" "$out" >&2
    exit "$status"
  fi
  python3 - "$out" "$members" "$sequences" "$wire_fp16" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
members = int(sys.argv[2])
expected = int(sys.argv[3])
wire_fp16 = bool(int(sys.argv[4]))
report = json.loads((out / "report.json").read_text())
assert report["schema"] == "ds4-fused-ar-probe-v1"
assert report["success"] and not report["error"]
assert report["members"] == members
assert report["fp16_wire"] == wire_fp16
assert [device["mapping_ok"] for device in report["devices"]] == [True] * members
if report["queue_mode"]:
    assert report["queue_depth"] == expected and not report["intermediate_outputs_checked"]
    assert report["sequences"] == [{"seq": expected, "ok": True}]
    assert report["pinned_input_bytes"] == members * expected * (report["elements_per_member"] * 4 + 128)
    assert not report["pinned_inputs_retained"] and report["pinned_input_status"] == "released"
else:
    assert not report["queue_mode"]
    assert len(report["sequences"]) == report["requested_sequences"] == expected
assert all(sequence["ok"] for sequence in report["sequences"])
assert all(check["ok"] for check in report["checks"])
assert (out / "ue-after.txt").read_text().strip() == "0"
print(json.dumps({"mode": f"TP{members}", "fp16_wire": wire_fp16,
                  "sequences": len(report["sequences"]), "checks": len(report["checks"]), "passed": True}))
PY
done
kill -TERM "$monitor"
wait "$monitor"
trap - EXIT
python3 - "$root" "$wire_fp16" "${QUEUED:-0}" "${modes[*]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
wire_fp16 = bool(int(sys.argv[2]))
queued = bool(int(sys.argv[3]))
scope = ("standalone queued TP probe; final output and guards only; intermediate outputs are not checked; "
         "not model TP integration or end-to-end Maya equivalence" if queued else
         "standalone FP16-wire TP2/TP4 probe; not model TP integration or end-to-end Maya equivalence"
         if wire_fp16 else
         "standalone FP32 TP2/TP4 probe; not model TP integration or FP16 equivalence")
result = {"schema": "ds4-fused-ar-hardware-v1", "passed": True, "fp16_wire": wire_fp16,
          "queue_mode": queued, "intermediate_outputs_checked": not queued,
          "scope": scope, "production_api_enabled": False, "modes": {}}
for mode in [f"tp{member}" for member in sys.argv[4].split()]:
    report = json.loads((root / mode / "report.json").read_text())
    assert report["success"] and not report["error"]
    assert report["fp16_wire"] == wire_fp16
    result["modes"][mode] = {"sequences": report["sequences"], "checks": report["checks"],
                             "devices": report["devices"], "ue_before": (root / mode / "ue-before.txt").read_text().strip(),
                             "ue_after": (root / mode / "ue-after.txt").read_text().strip(),
                             "ce_before": (root / mode / "ce-before.txt").read_text().strip(),
                             "ce_after": (root / mode / "ce-after.txt").read_text().strip()}
result["telemetry"] = json.loads((root / "telemetry.csv.metadata.json").read_text())
assert result["telemetry"]["status"] == "stopped"
(root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"passed": True, "fp16_wire": wire_fp16,
                  "modes": {mode: len(value["checks"]) for mode, value in result["modes"].items()}}))
PY
