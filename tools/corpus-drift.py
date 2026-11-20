#!/usr/bin/env python3
"""
corpus-drift — has the corpus moved since the last published manifest?

    python3 tools/corpus-drift.py --root ../src
    python3 tools/corpus-drift.py --root ../src --models rep,act
    python3 tools/corpus-drift.py --corpus ../corpus        # compare an existing harvest

The models are only as current as the corpus they were trained on, and that corpus is rebuilt from
repositories that change every day. This is the check that says whether a rebuild is due: it
re-harvests, then compares each model's SHA-256 against the one in `models/corpus-manifest.json`.

Exit codes
----------
0  nothing drifted
1  at least one model drifted — a rebuild is due
2  the harvest could not run (missing checkouts, missing manifest)
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MANIFEST = os.path.join(ROOT, "models", "corpus-manifest.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes(corpus, models):
    out = {}
    for m in models:
        p = os.path.join(corpus, "%s.train.jsonl" % m)
        if os.path.exists(p):
            out[m] = sha(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="directory holding the repo checkouts, to re-harvest")
    ap.add_argument("--corpus", help="an existing harvest to compare instead")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--models", default="",
                    help="comma-separated; default is every model in the manifest")
    ap.add_argument("--keep", action="store_true", help="leave the re-harvest on disk")
    a = ap.parse_args()

    if not os.path.exists(a.manifest):
        print("no manifest at %s" % a.manifest)
        return 2
    published = json.load(open(a.manifest))["models"]
    models = [m.strip() for m in a.models.split(",") if m.strip()] or sorted(published)

    corpus = a.corpus
    if not corpus:
        if not a.root:
            ap.error("one of --root or --corpus is required")
        # /tmp is a small tmpfs on some hosts and the repertory split is 134 MB, so the
        # scratch directory goes beside the repository unless you say otherwise
        corpus = tempfile.mkdtemp(prefix=".corpus-drift-", dir=ROOT)
        print("harvesting %s into %s" % (a.root, corpus))
        rc = subprocess.run([sys.executable, os.path.join(HERE, "lm-corpus-harvest.py"),
                             "--root", a.root, "--out", corpus,
                             "--models", ",".join(models)]).returncode
        if rc != 0:
            print("the harvest failed; a source checkout is probably missing")
            return 2

    now = hashes(corpus, models)
    drifted, same, new = [], [], []
    for m in models:
        if m not in now:
            new.append(m)
            continue
        old = published.get(m, {}).get("splits", {}).get("train", {}).get("sha256")
        if old is None:
            new.append(m)
        elif old == now[m]:
            same.append(m)
        else:
            drifted.append(m)

    if not a.keep and not a.corpus:
        shutil.rmtree(corpus, ignore_errors=True)

    for m in same:
        print("  same     %-8s %s" % (m, now[m][:12]))
    for m in drifted:
        print("  DRIFTED  %-8s %s -> %s" % (m, published[m]["splits"]["train"]["sha256"][:12],
                                            now[m][:12]))
    for m in new:
        print("  new      %-8s %s" % (m, now.get(m, "not harvested")[:12]))

    if drifted or new:
        print("\n%d drifted, %d new — rebuild and republish the manifest" % (len(drifted), len(new)))
        return 1
    print("\nnothing drifted across %d models" % len(same))
    return 0


if __name__ == "__main__":
    sys.exit(main())
