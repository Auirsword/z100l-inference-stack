#!/usr/bin/env bash
# Save or restore the live KV session between requests (engine SAVE/RESTORE).
#   session_ctl.sh save [name]     write the current slot-0 conversation to disk
#   session_ctl.sh restore <name>  load it back into slot 0
#   session_ctl.sh list            show the files in the save directory
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8080}
DIR=${SESSION_DIR:-/data/build/sessions}
action=${1:-list}
case "$action" in
  list)
    ls -lh "$DIR" 2>/dev/null || echo "no saved sessions in $DIR"
    ;;
  save)
    name=${2:-session-$(date -u +%Y%m%dT%H%M%SZ).bin}
    curl -sS -X POST "$BASE/slots/0?action=save" \
      -H 'Content-Type: application/json' -d "{\"filename\":\"$name\"}"
    echo
    ;;
  restore)
    name=${2:?usage: $0 restore NAME}
    curl -sS -X POST "$BASE/slots/0?action=restore" \
      -H 'Content-Type: application/json' -d "{\"filename\":\"$name\"}"
    echo
    ;;
  *)
    echo "usage: $0 {save [name]|restore name|list}" >&2
    exit 2
    ;;
esac
