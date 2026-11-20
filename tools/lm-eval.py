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
and every rate carries a 95% bootstrap interval, because a rate over twenty items is not a
rate over a corpus. The denominator is printed next to every number for the same reason.

Exit code is 1 when any --gate threshold is missed, so CI can use it.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np
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
    """Greedy, stopping when the JSON object closes.

    The scorer only reads the first balanced {...}, so generating past its closing brace
    cannot change the score. Stopping there is most of the runtime: a byte-level model
    otherwise runs the full 96 tokens for every pair, each one re-reading the whole prefix.
    """
    ids = [train.BOS] + tok.encode("IN: %s\nOUT: " % prompt)
    out, depth, opened = [], 0, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-seq:]])
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == train.EOS:
            break
        ids.append(nxt)
        out.append(nxt)
        ch = tok.decode([nxt])
        if ch == "{":
            depth += 1
            opened = True
        elif ch == "}" and opened:
            depth -= 1
            if depth <= 0:
                break
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


def rubric_index(path):
    """Every rubric id the training split mentions — what a returned id has to resolve against."""
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out = json.loads(line).get("output") or {}
            except ValueError:
                continue
            rid = out.get("rubric_id")
            if rid:
                ids.add(rid)
    return ids


def bootstrap(values, resamples, alpha=0.05):
    """Percentile bootstrap. Returns (lo, hi) for the mean."""
    if not values:
        return (0.0, 0.0)
    v = np.array(values, dtype=np.float64)
    idx = np.random.randint(0, len(v), size=(resamples, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True, help="corpus name, e.g. act")
    ap.add_argument("--limit", type=int, default=0, help="0 = every held-out pair")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--gate", action="append", default=[],
                    help="metric=threshold, repeatable, e.g. json_valid=0.9")
    ap.add_argument("--task", default="",
                    help="a task-specific scorer: rep checks the rubric id, and whether it resolves")
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
            if a.limit and len(rows) >= a.limit:
                break
            rows.append(json.loads(line))

    valid = exact = 0
    fields = []
    valid_flags = []
    examples = []
    for i, r in enumerate(rows):
        if i and i % 10 == 0:
            print("  ... %d/%d" % (i, len(rows)), flush=True)
        text = generate_one(model, tok, r["input"], cfg["seq"])
        got = parse_json(text)
        ok_json = got is not None
        valid += ok_json
        valid_flags.append(1.0 if ok_json else 0.0)
        f = field_score(r["output"], got) if ok_json else 0.0
        fields.append(f)
        exact += (f == 1.0)
        r["_got"] = got
        if len(examples) < 5:
            examples.append({"input": r["input"][:80], "expected": r["output"],
                             "got": text[:120], "field_score": round(f, 3)})

    # task-specific scoring, on the same generated text
    task_result = {}
    if a.task in ("book", "materia", "forge"):
        # one key decides each task: which book, which remedy, whether the spec is usable
        field = {"book": "book", "materia": "remedy", "forge": "id"}[a.task]
        right = present = 0
        for r in rows:
            want = (r.get("output") or {}).get(field)
            got = (r.get("_got") or {}).get(field)
            right += (want is not None and got == want)
            present += (got is not None)
        task_result = {field + "_exact": round(right / max(len(rows), 1), 4),
                       field + "_present": round(present / max(len(rows), 1), 4)}
        if a.task == "forge":
            # a weaker check than compiling the spec: the three keys a KILN spec cannot omit
            usable = sum(1 for r in rows if isinstance(r.get("_got"), dict)
                         and {"id", "family", "spec_keys"} <= set(r["_got"]))
            task_result["spec_usable"] = round(usable / max(len(rows), 1), 4)

    if a.task == "rep":
        index = rubric_index(os.path.join(a.corpus, "%s.train.jsonl" % a.model))
        hits = resolved = 0
        for r in rows:
            want = (r.get("output") or {}).get("rubric_id")
            got = (r.get("_got") or {}).get("rubric_id") if r.get("_got") else None
            hits += (want is not None and got == want)
            resolved += (got in index) if got else 0
        task_result = {"id_exact": round(hits / max(len(rows), 1), 4),
                       "id_resolvable": round(resolved / max(len(rows), 1), 4),
                       "index_size": len(index)}

    n = max(len(rows), 1)
    total = sum(1 for _ in open(path, encoding="utf-8"))
    result = {
        "run": a.run, "model": a.model, "n": len(rows), "held_out": total,
        "json_valid": round(valid / n, 4),
        "exact": round(exact / n, 4),
        "field_match": round(sum(fields) / n, 4),
        "examples": examples,
    }
    lo, hi = bootstrap(valid_flags, a.boot)
    flo, fhi = bootstrap(fields, a.boot)
    result["json_valid_ci"] = [round(lo, 3), round(hi, 3)]
    result["field_match_ci"] = [round(flo, 3), round(fhi, 3)]
    print("%-14s %d of %d held-out pairs" % (a.model, len(rows), total))
    print("  json_valid   %.3f   95%% CI %.3f-%.3f" % (result["json_valid"], lo, hi))
    print("  exact        %.3f" % result["exact"])
    print("  field_match  %.3f   95%% CI %.3f-%.3f" % (result["field_match"], flo, fhi))
    if task_result:
        result.update(task_result)
        for k, v in task_result.items():
            if k.endswith("_size"):
                continue
            print("  %-14s %.3f" % (k, v))
    if len(rows) < total:
        print("  note: a sample, not the split — run without --limit before quoting it")
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
