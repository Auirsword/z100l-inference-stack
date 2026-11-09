#!/usr/bin/env bash
# check-ds4.sh — 一条命令确认 DS4 API 可用（在任意装有 curl 的机器上运行）
# 用法: ./check-ds4.sh [http://host:18100] [api_key]
set -u
URL=${1:-http://192.168.0.123:18100}
KEY=${2:-}
H=()
[ -n "$KEY" ] && H=(-H "Authorization: Bearer $KEY")

echo "== $URL"
for p in /health /v1/models /metrics; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "${H[@]}" "$URL$p")
  printf '  %-12s %s\n' "$p" "$code"
done

echo "== 一次真实请求（8 token）"
curl -s --max-time 120 "${H[@]}" -H 'Content-Type: application/json' \
  "$URL/v1/chat/completions" \
  -d '{"model":"LLM-server-flash","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8,"temperature":0}' \
  | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print("  reply:", d["choices"][0]["message"]["content"][:60])
except Exception as e:
    print("  FAILED:", e)'
