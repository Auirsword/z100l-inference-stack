#!/usr/bin/env bash
# ds4-switch.sh <single|tp2|tp3|tp4> [--no-restart]
# Pick the DS4 serve topology (one binary, one config; a restart applies it).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOPO=${1:-}
case "$TOPO" in
  single|tp2|tp3|tp4) ;;
  *) echo "usage: $0 <single|tp2|tp3|tp4> [--no-restart]" >&2; exit 2 ;;
esac
CFG="$HERE/serve-$TOPO.json"
test -f "$CFG" || { echo "missing $CFG" >&2; exit 1; }
ln -sf "serve-$TOPO.json" "$HERE/serve.json"
mkdir -p /data/kv/ds4
echo "topology: $TOPO (config $CFG)"
if [ "${2:-}" != "--no-restart" ]; then
  systemctl restart ds4-serve.service
  systemctl --no-pager --lines=0 status ds4-serve.service | head -5
fi
