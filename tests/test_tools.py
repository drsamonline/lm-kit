#!/usr/bin/env python3
"""Smoke test — runs with no repositories, no network and no GPU.

    python3 tests/test_tools.py

Checks the parts of the pipeline that can be checked in a second: determinism,
the quality filter, the tokenizer round trip, and the extractors on a fixture.
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harvest = load("harvest", os.path.join(HERE, "..", "tools", "lm-corpus-harvest.py"))
bpe = load("bpe", os.path.join(HERE, "..", "tools", "lm-bpe.py"))

FAILED = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  " + detail) if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


print("determinism")
a = {"input": "x", "output": {"b": 2, "a": 1}}
b = {"input": "x", "output": {"a": 1, "b": 2}}
check("stable_key ignores dict order", harvest.stable_key(a) == harvest.stable_key(b))
check("stable_key separates content",
      harvest.stable_key(a) != harvest.stable_key({"input": "y", "output": {"a": 1, "b": 2}}))

print("quality filter")
prose = ("The remedy is indicated in complaints of women, especially those predisposed to "
         "haemorrhages, and it has been used where bleeding is copious and persistent.")
index = "Delirium, Acon. ant. am. Ars. Bell. Bry. calc. camph. canth. carb-v. Cham. Chin. cic."
check("keeps prose", harvest.good_prose(prose))
check("rejects a remedy index", not harvest.good_prose(index),
      "abbrev density %.2f" % harvest.abbrev_density(index))
check("rejects a fragment", not harvest.good_prose("too short"))

print("licence gate")
check("US 2028 is refused",
      harvest.pd_ok({"copyright_jurisdictions": {"US": 2028}})[0] is False)
check("US 1897 is allowed",
      harvest.pd_ok({"copyright_jurisdictions": {"US": 1897}})[0] is True)
check("no jurisdictions is allowed", harvest.pd_ok({})[0] is True)

print("tokenizer")
spec = bpe.train(["the quick brown fox jumps over the lazy dog"] * 20, 300, min_count=1, verbose=False)
tok = bpe.BPE(spec)
for text in ["the quick brown fox", "{\"api\": \"Trakt\", \"auth\": \"none\"}"]:
    ids = tok.encode(text)
    back = tok.decode(ids)
    norm = lambda t: " ".join(t.split())
    check("round trip: " + text[:24], norm(back) == norm(text))
    check("compresses: " + text[:24], len(ids) < len(text))

print("extractors on a fixture")
with tempfile.TemporaryDirectory() as tmp:
    root = os.path.join(tmp, "src")
    os.makedirs(os.path.join(root, "TOOLWORKS", "data"))
    json.dump({"count": 2, "tools": [
        {"name": "Shrink", "desc": "resize an image down to a width you choose",
         "category": "Images", "group": "media", "source": "all4one", "dangerous": False},
        {"name": "x", "desc": "", "category": "Other"},
    ]}, open(os.path.join(root, "TOOLWORKS", "data", "catalogue.json"), "w"))
    skip = []
    pairs = list(harvest.x_act(root, skip))
    check("yields one pair per described tool", len(pairs) == 1, "got %d" % len(pairs))
    check("carries the danger flag", pairs and pairs[0]["output"]["dangerous"] is False)
    check("records its source", pairs and pairs[0]["source"].startswith("TOOLWORKS/"))

print()
if FAILED:
    print("%d failed: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
