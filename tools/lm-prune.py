#!/usr/bin/env python3
"""
lm-prune — structured pruning of the feed-forward, measured before it is trusted.

    python3 tools/lm-prune.py --run ../tuned-run --keep 0.75 --out ../pruned-run

Removes whole feed-forward channels rather than individual weights: w1 and w3 lose rows, w2 loses
the matching columns, and `ff` shrinks in the config. That is a real size reduction, unlike masking
a weight to zero and calling it sparse.

Channels are ranked by the L2 norm of the row in w1 plus the row in w3 — a channel that is small in
both is doing little. MINI-LLM scores by feature-map sensitivity on real activations, which is
better; this is the cheap version, and the question it answers is whether the headroom exists at
all at this size.

The output is a checkpoint to be retrained, not a finished model. Pruning without a retrain always
loses more than it needs to.
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    sd = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
    return cfg, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--keep", type=float, default=0.75, help="fraction of ff channels to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg, sd = load(a.run)
    ff, d, layers = cfg["ff"], cfg["dim"], cfg["layers"]
    keep = max(1, int(round(ff * a.keep)))
    if keep >= ff:
        sys.exit("--keep %.2f keeps every channel" % a.keep)

    out = dict(sd)
    for l in range(layers):
        w1 = sd["blocks.%d.w1.weight" % l].float()
        w3 = sd["blocks.%d.w3.weight" % l].float()
        w2 = sd["blocks.%d.w2.weight" % l].float()

        score = w1.norm(dim=1) + w3.norm(dim=1)
        idx = torch.argsort(score, descending=True)[:keep].sort().values

        out["blocks.%d.w1.weight" % l] = w1[idx].clone()
        out["blocks.%d.w3.weight" % l] = w3[idx].clone()
        out["blocks.%d.w2.weight" % l] = w2[:, idx].clone()

    cfg_out = dict(cfg)
    cfg_out["ff"] = keep
    cfg_out["params"] = sum(v.numel() for k, v in out.items() if not k.endswith((".A", ".B")))

    before = cfg.get("params", 0)
    after = cfg_out["params"]
    print("ff  {} -> {}".format(ff, keep))
    print("parameters  {:,} -> {:,}  ({:.1f}% removed)".format(before, after, 100 * (1 - after / before)))
    print("int8 size   {:.2f} MB -> {:.2f} MB".format(before / 1e6, after / 1e6))
    print("pruning without a retrain loses more than it needs to: retrain with --resume")

    if a.dry_run:
        return 0

    os.makedirs(a.out, exist_ok=True)
    torch.save(out, os.path.join(a.out, "model.pt"))
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("bpe.json",):
        src = os.path.join(a.run, f)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(a.out, f))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
