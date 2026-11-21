#!/usr/bin/env python3
"""
lm-registry — versioned model storage with a way back.

    python3 tools/lm-registry.py list
    python3 tools/lm-registry.py publish --model act
    python3 tools/lm-registry.py rollback --model act

Why this exists
---------------
`lm-rebuild.py` publishes by copying files over `models/<model>/`. That is one bad gate away
from being unrecoverable: the previous checkpoint is gone the moment the new one lands. This
keeps the last two snapshots and a registry that says which one is live.

Layout
------
    models/<model>/                 the live checkpoint — what the kernel and the exporter read
    models/.archive/<model>/v1/     snapshots, oldest first, pruned to the last two
    models/registry.json            which version is live, and what each one scored

The card is the identity of a version: its hash is recorded at publish time, so a version whose
files have been edited in place is detectable rather than silently different.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
KEEP = 2
FILES = ("model.pt", "model.npz", "config.json", "card.json", "bpe.json")


def registry_path(runs):
    return os.path.join(runs, "registry.json")


def load_registry(runs):
    p = registry_path(runs)
    if os.path.exists(p):
        return json.load(open(p))
    return {"schema": "lm-registry/1", "models": {}}


def save_registry(runs, reg):
    json.dump(reg, open(registry_path(runs), "w"), indent=1, sort_keys=True)


def sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def archive_dir(runs, model, v):
    return os.path.join(runs, ".archive", model, "v%d" % v)


def snapshot(runs, model, v):
    """Copy the live checkpoint into the archive as version v."""
    dst = archive_dir(runs, model, v)
    os.makedirs(dst, exist_ok=True)
    copied = []
    for f in FILES:
        src = os.path.join(runs, model, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst, f))
            copied.append(f)
    return copied


def restore(runs, model, v):
    dst = os.path.join(runs, model)
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        src = os.path.join(archive_dir(runs, model, v), f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst, f))


def prune(runs, model, versions):
    """Keep the last KEEP snapshots; delete the rest."""
    for entry in versions[:-KEEP]:
        shutil.rmtree(archive_dir(runs, model, entry["v"]), ignore_errors=True)
    return versions[-KEEP:]


def cmd_publish(a):
    reg = load_registry(a.runs)
    model = a.model
    live = os.path.join(a.runs, model)
    card_path = os.path.join(live, "card.json")
    if not os.path.isdir(live) or not os.path.exists(card_path):
        print("no checkpoint at %s" % live)
        return 2
    card = json.load(open(card_path))

    existing = reg["models"].get(model, {}).get("versions", [])
    # the highest version in the registry, not the count: pruning keeps two entries, so the count
    # stops advancing and the fourth publish reused v3, overwriting its snapshot
    next_v = max((e["v"] for e in existing), default=0) + 1
    entry = {
        "v": next_v,
        "published": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card_sha256": sha(card_path),
        "params": card.get("params"),
        "eval_loss": card.get("eval_loss"),
        "corpus_sha256": card.get("corpus_sha256"),
    }
    if os.path.isdir(archive_dir(a.runs, model, next_v)):
        print("%s v%d already has a snapshot on disk; refusing to overwrite it"
              % (model, next_v))
        return 2
    m = reg["models"].setdefault(model, {"current": 0, "versions": []})
    m["versions"].append(entry)
    m["versions"] = prune(a.runs, model, m["versions"])
    m["current"] = entry["v"]
    snapshot(a.runs, model, entry["v"])
    save_registry(a.runs, reg)
    print("published %s v%d  card %s  params %s" % (model, entry["v"], entry["card_sha256"], entry["params"]))
    return 0


def cmd_rollback(a):
    reg = load_registry(a.runs)
    m = reg["models"].get(a.model)
    if not m or len(m["versions"]) < 2:
        print("%s has no earlier version to roll back to" % a.model)
        return 2
    cur = m["current"]
    previous = [v for v in m["versions"] if v["v"] < cur]
    if not previous:
        print("%s is already at its earliest kept version" % a.model)
        return 2
    target = previous[-1]
    restore(a.runs, a.model, target["v"])
    m["current"] = target["v"]
    save_registry(a.runs, reg)
    print("rolled %s back to v%d  card %s" % (a.model, target["v"], target["card_sha256"]))
    return 0


def cmd_list(a):
    reg = load_registry(a.runs)
    if not reg["models"]:
        print("the registry is empty — publish a checkpoint first")
        return 0
    for model in sorted(reg["models"]):
        m = reg["models"][model]
        print("%s  current v%d" % (model, m["current"]))
        for v in m["versions"]:
            mark = " *" if v["v"] == m["current"] else "  "
            print("   %sv%-3d %s  card %s  params %-9s eval %s"
                  % (mark, v["v"], v["published"], v["card_sha256"],
                     v["params"], v["eval_loss"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "publish", "rollback"])
    ap.add_argument("--runs", default=os.path.join(ROOT, "models"))
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.cmd in ("publish", "rollback") and not a.model:
        ap.error("--model is required for %s" % a.cmd)
    return {"list": cmd_list, "publish": cmd_publish, "rollback": cmd_rollback}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
