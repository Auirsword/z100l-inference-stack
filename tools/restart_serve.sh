#!/usr/bin/env bash
# Safe restart: save the live session to disk, restart, then restore it.
#   restart_serve.sh            save + restart + restore the file it just saved
#   restart_serve.sh --no-save  plain restart
set -euo pipefail
TOOLS=/data/z100l-strata-port/tools
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NAME="session-$STAMP.bin"
if [[ "${1:-}" != "--no-save" ]]; then
  echo "== saving the live session to $NAME"
  "$TOOLS/session_ctl.sh" save "$NAME" || echo "  (save failed, continuing)"
fi
echo "== restarting the serve container"
python3 "$TOOLS/start_serve.py"
if [[ "${1:-}" != "--no-save" ]]; then
  echo "== restoring $NAME"
  "$TOOLS/session_ctl.sh" restore "$NAME" || echo "  (restore failed; the next request re-prefills)"
fi
