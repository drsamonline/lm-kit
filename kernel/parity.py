#!/usr/bin/env python3
"""
parity.py — prove the C kernel produces the same tokens as PyTorch.

    python3 kernel/parity.py --run ../run-act --lmk ../run-act/model-fp32.lmk

Both sides generate greedily and the strings are compared exactly. Run it against an
**fp32** export: int8 weights are a different model by design, so comparing them would
measure the quantiser, not the kernel.

Exit code 1 on the first mismatch, with both strings printed.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")

PROMPTS = [
    "Job postings from 185 sources",
    "Movie and TV data",
    "Find the right free API for weather",
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def torch_generate(t, tok, model, prompt, n, seq):
    ids = [t.BOS] + tok.encode("IN: %s\nOUT: " % prompt)
    out = []
    with torch.no_grad():
        for _ in range(n):
            nxt = int(model(torch.tensor([ids[-seq:]]))[0, -1].argmax())
            if nxt == t.EOS:
                break
            ids.append(nxt)
            out.append(nxt)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lmk", required=True)
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()

    t = load("lm_train", os.path.join(TOOLS, "lm-train.py"))
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    tok = t.make_tokenizer(argparse.Namespace(
        tokenizer=cfg.get("tokenizer", "bytes"), bpe=os.path.join(a.run, "bpe.json")))
    model = t.TinyLM(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"])
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location="cpu"))
    model.eval()

    exe = os.path.join(HERE, "lm")
    if not os.path.exists(exe):
        sys.exit("build the kernel first: cd kernel && make")

    bad = 0
    for p in PROMPTS:
        py = torch_generate(t, tok, model, p, a.tokens, cfg["seq"])
        out = subprocess.run([exe, a.lmk, p, str(a.tokens)],
                             capture_output=True, text=True, timeout=300)
        c = out.stdout.strip("\n")
        if py == c:
            print("  parity  %s" % p)
        else:
            bad += 1
            print("  DIFFER  %s" % p)
            print("    pytorch: %r" % py)
            print("    c      : %r" % c)
    if bad:
        print("\n%d of %d prompts differ" % (bad, len(PROMPTS)))
        return 1
    print("\nall %d prompts match" % len(PROMPTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
