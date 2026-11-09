#!/usr/bin/env python3
"""Add DeepSeek-V4.1's vision tower to an existing ds4 pack (the GGUFs leave it out).

The tower is not in any GGUF: it lives in shard 1 of DeepSeek's own checkpoint (vision.* + aligner.*, BF16) and the
picture span's three marker rows (image_start/newline/end) in shard 2.  The project ships the recipe already:

    # 1. the tower's shard and its geometry (0.97 GB, the only big download)
    R=1bae0e1088993b9b82de08f59de5be5a34e42170
    curl -fL -o vision-src/config.json \\
      "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/config.json"
    curl -fL -o vision-src/model-00001-of-00048.safetensors \\
      "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/model-00001-of-00048.safetensors"
    sha256sum vision-src/model-00001-of-00048.safetensors   # 13bd1210...d52fddb0ec

    # 2. vision.bin + vision.json (266 tensors, every one checked against the shard; the marker rows come from
    #    shard 2 through 30 KB of HTTP range requests)
    python tools/ds4/vision_pack.py --dir vision-src \\
      --marks-url "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/model-00002-of-00048.safetensors"

    # 3. this script: the pack takes them (tensors.txt + ds4.txt's geometry)
    python add-vision-to-pack.py --vision vision-src --pack /models/.../maya-s/maya-ds4

then `--vision on` in the serve config; the engine reports `INFO vision=ds4` and the server offers images.

The alternative is re-running tools/ds4/pack_gguf.py with `--vision <folder>`, which rebuilds the whole pack; this
script only appends, so an existing pack keeps its byte-for-byte layout.
"""
import argparse
import json
import os
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True, help="vision_pack.py's folder (vision.bin + vision.json)")
    ap.add_argument("--pack", required=True, help="the pack folder (ds4.txt + tensors.txt)")
    a = ap.parse_args()
    vd = pathlib.Path(a.vision).resolve()
    pack = pathlib.Path(a.pack).resolve()
    info = json.loads((vd / "vision.json").read_text())
    model_dir = None
    for line in (pack / "ds4.txt").read_text().splitlines():
        if line.startswith("model_dir "):
            model_dir = pathlib.Path(line.split(" ", 1)[1]).resolve()
    if model_dir is None:
        print("ds4.txt has no model_dir line: not a gguf pack", file=sys.stderr)
        return 1
    try:
        rel = os.path.relpath(vd / "vision.bin", model_dir).replace(os.sep, "/")
    except ValueError:
        print("the vision folder must be on the model's drive", file=sys.stderr)
        return 1
    if " " in rel:
        print("the vision folder's path relative to the model has a space", file=sys.stderr)
        return 1

    tpath = pack / "tensors.txt"
    lines = tpath.read_text().splitlines()
    if any(l.startswith("file ") and l.endswith("vision/vision.bin") for l in lines):
        print("already patched")
    else:
        nfile = sum(1 for l in lines if l.startswith("file "))
        last = max(i for i, l in enumerate(lines) if l.startswith("file "))
        out = lines[:last + 1] + ["file %d %s" % (nfile, rel)] + lines[last + 1:]
        for name, dt, off, nb, shape in info["tensors"]:
            out.append("t %s %s %d %d %d %s" % (name, dt, nfile, off, nb, " ".join(map(str, shape))))
        tpath.write_text("\n".join(out) + "\n")
        print("tensors.txt: +%d tensors (file %d = %s)" % (len(info["tensors"]), nfile, rel))

    dpath = pack / "ds4.txt"
    d = dpath.read_text().splitlines()
    d = [l for l in d if not l.startswith("vision.") and not l.startswith("image_token_id ")]
    i = next((k for k, l in enumerate(d) if l.startswith("format ")), len(d))
    add = ["%s %s" % (k, v) for k, v in info["config"].items()]
    dpath.write_text("\n".join(d[:i] + add + d[i:]) + "\n")
    print("ds4.txt: %d geometry keys" % len(add))
    for line in add:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
