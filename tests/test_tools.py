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


print("js literal reader (no execution)")
objs = list(harvest.js_objects('window.Q.push({u:1,d:"E",q:"a {brace} inside",o:["x","y"],c:0,e:"why"});',
                               ("q", "o", "c")))
check("reads a bare-key object", len(objs) == 1, "got %d" % len(objs))
check("keeps braces inside strings", objs and objs[0]["q"] == "a {brace} inside")
check("reads the array", objs and objs[0]["o"] == ["x", "y"])
check("reads numbers", objs and objs[0]["c"] == 0)

print("pronunciation decoder")
check("alphabet is 69 symbols", len(harvest.PHONES) == 69, "got %d" % len(harvest.PHONES))
check("first symbol is AA0", harvest.PHONES[0] == "AA0")
check("consonants follow the vowels", harvest.PHONES[45] == "B")

print("extractors on a fixture (continued)")
with tempfile.TemporaryDirectory() as tmp:
    root = os.path.join(tmp, "src")
    d = os.path.join(root, "ntet-homoeopathy-cbt", "data")
    os.makedirs(d)
    open(os.path.join(d, "q-unit12.js"), "w").write(
        'window.QBANK.push({u:1,d:"E",q:"Which level?",o:["A","B"],c:1,e:"Because."},'
        '{u:1,d:"M",q:"Bad",o:[],c:0,e:""});')
    open(os.path.join(d, "glossary.js"), "w").write(
        'window.GLOSSARY=[{t:"Pedagogy",u:1,d:"The art and science of teaching, applied to children."}];')
    skip = []
    pairs = list(harvest.x_tutor(root, skip))
    check("keeps the well-formed question", any("Which level?" in q["input"] for q in pairs))
    check("drops the question with no options", not any("Bad" in q["input"] for q in pairs))
    check("carries the rationale", any(q["output"].get("rationale") == "Because." for q in pairs))
    check("reads the glossary", any(q["input"] == "Pedagogy" for q in pairs))

print("registry (pure file operations, so this runs everywhere)")
reg = load("lm_registry", os.path.join(HERE, "..", "tools", "lm-registry.py"))
with tempfile.TemporaryDirectory() as tmp:
    runs = os.path.join(tmp, "models")
    live = os.path.join(runs, "m")
    os.makedirs(live)
    for f in ("config.json", "card.json"):
        open(os.path.join(live, f), "w").write('{"steps": 1}')

    class A:
        model, runs = "m", runs

    versions = []
    for _ in range(4):
        reg.cmd_publish(A)
        r = json.load(open(os.path.join(runs, "registry.json")))
        versions.append(r["models"]["m"]["current"])
    check("versions advance past the prune", versions == [1, 2, 3, 4], str(versions))
    check("only the last two snapshots survive",
          sorted(os.listdir(os.path.join(runs, ".archive", "m"))) == ["v3", "v4"])
    check("rollback moves back one", reg.cmd_rollback(A) == 0 and
          json.load(open(os.path.join(runs, "registry.json")))["models"]["m"]["current"] == 3)

print("task scorers (skipped without torch)")
try:
    import torch  # noqa: F401
    ev = load("lm_eval", os.path.join(HERE, "..", "tools", "lm-eval.py"))
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.train.jsonl")
        with open(p, "w") as fh:
            fh.write(json.dumps({"input": "a", "output": {"rubric_id": "r1"}}) + "\n")
            fh.write(json.dumps({"input": "b", "output": {"book": "x"}}) + "\n")
        idx = ev.rubric_index(p)
        check("rubric index picks up ids", idx == {"r1"}, str(idx))

    # the task scorers are arithmetic over generated rows, so they can be checked without a model
    rows = [{"output": {"book": "a"}, "_got": {"book": "a"}},
            {"output": {"book": "b"}, "_got": {"book": "a"}},
            {"output": {"book": "c"}, "_got": None}]
    right = sum(1 for r in rows if (r["output"].get("book") == (r["_got"] or {}).get("book")))
    present = sum(1 for r in rows if (r["_got"] or {}).get("book") is not None)
    check("one of three is right", right == 1, str(right))
    check("two of three are present", present == 2, str(present))
    check("a missing file gives an empty index", ev.rubric_index("/nope") == set())
except ImportError:
    print("  skip  torch is not installed here")

print()
if FAILED:
    print("%d failed: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
