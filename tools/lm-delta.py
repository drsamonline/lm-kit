#!/usr/bin/env python3
"""
lm-delta — how big is the difference between a base model and its continuation?

    python3 tools/lm-delta.py --base ../models/act --tuned ../models/act

The family design says one base plus eight small deltas, nine specialities for about 4 MB. That
only works if a delta is genuinely smaller than a model. This measures it instead of assuming it.

What it reports, per tensor and in total
----------------------------------------
  relative size   ||tuned - base|| / ||base||, so a delta that is 2% of the base is worth storing
  dead fraction   share of entries whose change is under 1% of the tensor's own scale
  int8 delta      what the delta costs if stored the same way the weights are

It writes nothing unless --out is given, and it does not decide whether a delta is worth shipping.
The number does.
"""

import argparse
import json
import os
import sys

import torch


def load(run):
    cfg_path = os.path.join(run, "config.json")
    pt = os.path.join(run, "model.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(pt)):
        sys.exit("%s has no config.json and model.pt" % run)
    return json.load(open(cfg_path)), torch.load(pt, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg_b, base = load(a.base)
    cfg_t, tuned = load(a.tuned)
    if cfg_b != cfg_t:
        sys.exit("the two checkpoints do not share a configuration")

    total_params = 0
    total_delta = 0.0
    total_norm = 0.0
    dead = 0
    rows = []

    for key in sorted(base):
        if key not in tuned:
            continue
        b = base[key].float()
        t = tuned[key].float()
        d = t - b
        n = b.numel()
        bn = b.norm().item()
        dn = d.norm().item()
        scale = b.abs().max().item() or 1e-8
        dead_frac = (d.abs() < 0.01 * scale).float().mean().item()

        total_params += n
        total_norm += bn ** 2
        total_delta += dn ** 2
        dead += dead_frac * n
        rows.append((key, n, dn / bn if bn else 0.0, dead_frac, scale))

    rel = (total_delta ** 0.5) / (total_norm ** 0.5)
    dead_frac = dead / max(total_params, 1)

    if a.verbose:
        print("%-28s %9s %8s %8s" % ("tensor", "params", "rel", "dead"))
        for key, n, r, df, _ in rows:
            print("%-28s %9d %7.1f%% %7.1f%%" % (key, n, 100 * r, 100 * df))

    int8_full = total_params
    int8_delta = total_params  # worst case: every entry changes
    sparse_delta = int(total_params * (1 - dead_frac) * 5)  # 4 bytes index-ish + 1 byte value

    print()
    print("parameters            {:,}".format(total_params))
    print("relative delta        {:.1f}% of the base".format(100 * rel))
    print("dead entries          {:.1f}% change by under 1% of scale".format(100 * dead_frac))
    print()
    print("stored as int8:")
    print("  the base            {:>7.2f} MB".format(int8_full / 1e6))
    print("  a full int8 delta   {:>7.2f} MB   no saving".format(int8_delta / 1e6))
    print("  a sparse delta      {:>7.2f} MB   at 5 bytes an entry".format(sparse_delta / 1e6))

    if a.out:
        torch.save({k: (tuned[k].float() - base[k].float()) for k in base}, a.out)
        print("\nwrote the raw delta to %s (%.2f MB)"
              % (a.out, os.path.getsize(a.out) / 1e6))

    print()
    if dead_frac < 0.5:
        print("Verdict: the delta is dense. Storing it costs about what the model costs, so")
        print("base + deltas is not a saving at this size. It needs low-rank (LoRA) training,")
        print("not a difference of two full fine-tunes.")
    else:
        print("Verdict: the delta is sparse enough to be worth storing as a delta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
