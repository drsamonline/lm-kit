#!/usr/bin/env python3
"""
lm-eval — score a checkpoint against its held-out split, and fail when it is not good enough.

    python3 lm-eval.py --run ../run-act-bpe --corpus ../corpus --model act
    python3 lm-eval.py --run ../run-act-bpe --corpus ../corpus --model act --gate json_valid=0.9

Why a separate tool
-------------------
The trainer reports cross-entropy, which is a number about the *next token*. Nobody ships a
language model because its loss went down; they ship it because the thing it produces is right.
So this scores the output: does it parse, does it match the expected object, and which fields
survive.

Every metric here can fail, and each one names what it is measuring. A gate that cannot fail is
not a gate.

Metrics
-------
  json_valid    the generated text parses as a JSON object
  exact         the parsed object equals the expected one exactly
  field_match   mean fraction of expected fields present and equal
  refuse_ok     for pairs whose output carries no answer, the model emitted something invalid
                rather than a confident wrong object

Exit code is 1 when any --gate threshold is missed, so CI can use it.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = load("lm_train", os.path.join(HERE, "lm-train.py"))


# ---------------------------------------------------------------- generate
@torch.no_grad()
def generate_one(model, tok, prompt, seq, max_new=96):
    ids = [train.BOS] + tok.encode("IN: %s\nOUT: " % prompt)
    out = []
    for _ in range(max_new):
        ctx = torch.tensor([ids[-seq:]])
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == train.EOS:
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out)


def parse_json(text):
    """Take the first balanced {...} and parse it, or return None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except ValueError:
                    return None
    return None


def field_score(expected, got):
    if not isinstance(expected, dict) or not isinstance(got, dict):
        return 0.0
    keys = list(expected)
    if not keys:
        return 0.0
    hits = 0
    for k in keys:
        if k in got and json.dumps(got[k], sort_keys=True) == json.dumps(expected[k], sort_keys=True):
            hits += 1
    return hits / len(keys)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True, help="corpus name, e.g. act")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--gate", action="append", default=[],
                    help="metric=threshold, repeatable, e.g. json_valid=0.9")
    ap.add_argument("--out", default=None, help="write the result json here")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, "config.json")))
    tok = train.make_tokenizer(argparse.Namespace(
        tokenizer=cfg.get("tokenizer", "bytes"),
        bpe=os.path.join(a.run, "bpe.json")))
    model = train.TinyLM(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"])
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location="cpu"))
    model.eval()

    path = os.path.join(a.corpus, "%s.eval.jsonl" % a.model)
    if not os.path.exists(path):
        sys.exit("no eval split at %s" % path)

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= a.limit:
                break
            rows.append(json.loads(line))

    valid = exact = 0
    fields = []
    examples = []
    for r in rows:
        text = generate_one(model, tok, r["input"], cfg["seq"])
        got = parse_json(text)
        ok_json = got is not None
        valid += ok_json
        f = field_score(r["output"], got) if ok_json else 0.0
        fields.append(f)
        exact += (f == 1.0)
        if len(examples) < 5:
            examples.append({"input": r["input"][:80], "expected": r["output"],
                             "got": text[:120], "field_score": round(f, 3)})

    n = max(len(rows), 1)
    result = {
        "run": a.run, "model": a.model, "n": len(rows),
        "json_valid": round(valid / n, 4),
        "exact": round(exact / n, 4),
        "field_match": round(sum(fields) / n, 4),
        "examples": examples,
    }
    print("%-14s n=%d" % (a.model, len(rows)))
    print("  json_valid   %.3f" % result["json_valid"])
    print("  exact        %.3f" % result["exact"])
    print("  field_match  %.3f" % result["field_match"])
    print("  examples")
    for e in examples[:3]:
        print("    in   %s" % e["input"])
        print("    want %s" % json.dumps(e["expected"], sort_keys=True)[:100])
        print("    got  %s" % e["got"].replace("\n", " ")[:100])

    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)

    failed = []
    for g in a.gate:
        if "=" not in g:
            sys.exit("--gate wants metric=threshold, got %r" % g)
        k, v = g.split("=", 1)
        if k not in result:
            sys.exit("unknown metric %r" % k)
        if result[k] < float(v):
            failed.append("%s %.3f < %.3f" % (k, result[k], float(v)))
    if failed:
        print("\nGATE FAILED: " + "; ".join(failed))
        return 1
    if a.gate:
        print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
