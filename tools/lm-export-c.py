#!/usr/bin/env python3
"""
lm-export-c — turn a trained checkpoint into one portable file the C kernel can load.

    python3 lm-export-c.py --run ../run-act --out ../run-act/model.lmk

The format
----------
    magic      "LMK1"
    version    u32
    n_tensors  u32
    per tensor: name_len u8 | name | ndim u8 | dims u32[ndim] | dtype u8 | nbytes u32 | data

dtype 0 is float32, 1 is int8 with a per-row scale stored as a float32 tensor named
"<name>.scale". Everything is little-endian.

Why not GGUF
------------
GGUF carries a quantisation zoo, a tokeniser runtime, a chat template and a sampler
library. This kernel needs none of that: one architecture, one file, 400 lines of C.
A format nobody else can read is a feature when the whole point is that the reader is
small enough to audit.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"LMK1"
VERSION = 1


def write_tensor(fh, name, arr, dtype):
    raw = arr.tobytes(order="C")
    nb = name.encode()
    fh.write(struct.pack("<B", len(nb)))
    fh.write(nb)
    fh.write(struct.pack("<B", arr.ndim))
    for d in arr.shape:
        fh.write(struct.pack("<I", int(d)))
    fh.write(struct.pack("<B", dtype))
    fh.write(struct.pack("<I", len(raw)))
    fh.write(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true",
                    help="export float32 from model.pt instead of int8 — for a parity test")
    a = ap.parse_args()

    cfg_path = os.path.join(a.run, "config.json")
    npz_path = os.path.join(a.run, "model.npz")
    if not os.path.exists(npz_path):
        sys.exit("no model.npz in %s" % a.run)
    cfg = json.load(open(cfg_path))

    tensors = []
    if a.fp32:
        import torch
        sd = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")
        for key in sorted(sd):
            arr = sd[key].detach().numpy()
            tensors.append((key.replace(".", "/"), arr.astype("<f4"), 0))
    else:
        z = np.load(npz_path)
        for key in sorted(z.files):
            arr = z[key]
            if key.endswith(".scale"):
                tensors.append((key, arr.astype("<f4"), 0))
            elif arr.dtype == np.int8:
                tensors.append((key, arr, 1))
            else:
                tensors.append((key, arr.astype("<f4"), 0))

    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", VERSION))
        fh.write(struct.pack("<I", len(tensors)))
        for name, arr, dtype in tensors:
            write_tensor(fh, name, arr, dtype)

    size = os.path.getsize(a.out)
    print("wrote %s  (%d tensors, %.2f MB)" % (a.out, len(tensors), size / 1e6))
    print("  vocab %d  dim %d  layers %d  heads %d  ff %d  seq %d  tokenizer %s"
          % (cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"],
             cfg.get("tokenizer", "bytes")))
    if cfg.get("tokenizer", "bytes") != "bytes":
        print("  note: this kernel decodes byte-level only; the BPE merges are not in the file")


if __name__ == "__main__":
    main()
