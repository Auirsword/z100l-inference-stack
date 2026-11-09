#!/usr/bin/env python3
"""Minimal single-stream benchmark against a DS4/Qwen OpenAI-compatible server."""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PROMPT = ("Explain in three short sentences why a mixture-of-experts model is "
          "bound by memory bandwidth rather than arithmetic throughput.")

for index in range(RUNS):
    body = json.dumps({
        "model": "LLM-server-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": TOKENS,
        "stream": False,
        "temperature": 0,
    }).encode()
    started = time.monotonic()
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
        headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(request, timeout=1800).read())
    wall = time.monotonic() - started
    usage = data.get("usage", {})
    done = usage.get("completion_tokens") or 0
    print("run %d: wall=%.2fs prompt=%s completion=%s tok/s=%.2f" % (
        index + 1, wall, usage.get("prompt_tokens"), done,
        (done / wall) if wall and done else 0.0), flush=True)
