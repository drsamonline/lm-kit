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
"<name>.scale", 2 is int32, 3 is 4-bit packed two per byte with the same per-row scale.
Everything is little-endian.

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
    ap.add_argument("--q4", action="store_true",
                    help="4-bit with a per-row scale: half the file of int8, two values a byte")
    ap.add_argument("--fp32", action="store_true",
                    help="export float32 from model.pt instead of int8 — for a parity test")
    a = ap.parse_args()

    cfg_path = os.path.join(a.run, "config.json")
    npz_path = os.path.join(a.run, "model.npz")
    if not os.path.exists(npz_path):
        sys.exit("no model.npz in %s" % a.run)
    cfg = json.load(open(cfg_path))

    def pack_q4(arr):
        """int4 per weight, two per byte, one fp32 scale per row. Rows are padded to even length."""
        a2 = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
        rows, cols = a2.shape
        pad = cols % 2
        if pad:
            a2 = np.pad(a2, ((0, 0), (0, 1)))
            cols += 1
        scale = np.abs(a2).max(axis=1) / 7.0
        scale[scale == 0] = 1e-8
        q = np.clip(np.round(a2 / scale[:, None]), -8, 7).astype(np.int8) & 0xF
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        return packed.reshape(arr.shape[:-1] + (-1,)) if arr.ndim > 1 else packed[0], \
               scale.astype("<f4"), pad

    tensors = []
    if a.q4:
        import torch
        sd = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")
        for key in sorted(sd):
            arr = sd[key].detach().numpy().astype("<f4")
            if arr.ndim < 2 or min(arr.shape) < 2:
                tensors.append((key.replace(".", "/"), arr, 0))
                continue
            packed, scale, pad = pack_q4(arr)
            name = key.replace(".", "/")
            tensors.append((name, packed, 3))
            tensors.append((name + ".scale", scale, 0))
    elif a.fp32:
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

    # the shape travels with the weights: the kernel must not guess heads or context
    shape = np.array([cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"]],
                     dtype="<i4")
    tensors.append(("config/shape", shape, 2))

    # a BPE model carries its merges, or the kernel cannot tokenise
    if cfg.get("tokenizer", "bytes") == "bpe":
        bpe_path = os.path.join(a.run, "bpe.json")
        if not os.path.exists(bpe_path):
            sys.exit("tokenizer is bpe but %s is missing" % bpe_path)
        spec = json.load(open(bpe_path))
        merges = np.array(spec["merges"], dtype="<i4")
        tensors.append(("tokenizer/merges", merges, 2))

    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", VERSION))
        fh.write(struct.pack("<I", len(tensors)))
        for name, arr, dtype in tensors:
            write_tensor(fh, name, arr, dtype)

    if a.q4:
        print("  note: 4-bit was measured to break this model at 1.08 M parameters —")
        print("        per-row scales give '{\"api\": \"Dainininmintinininininininmanti', and")
        print("        32-element groups only reach '{\"api\": \"Deanminmi\"'. int8 survives.")
        print("        See docs/EVAL.md. Q4 belongs on the 20 M models, not the demo.")

    size = os.path.getsize(a.out)
    print("wrote %s  (%d tensors, %.2f MB)" % (a.out, len(tensors), size / 1e6))
    print("  vocab %d  dim %d  layers %d  heads %d  ff %d  seq %d  tokenizer %s"
          % (cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"],
             cfg.get("tokenizer", "bytes")))
    if cfg.get("tokenizer", "bytes") == "bpe":
        print("  merges %d embedded" % (len(tensors[-1][1]) if tensors else 0))


if __name__ == "__main__":
    main()
