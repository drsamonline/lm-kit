#!/usr/bin/env python3
"""
lm-rebuild — close the loop: drift in, gated checkpoint out.

    python3 tools/lm-rebuild.py --root ../src --corpus ../corpus --runs ../models \\
            --gate json_valid=0.30 --steps 60

What it does, per model
-----------------------
1. Is the corpus still the one the checkpoint was trained on? If not, do nothing. A rebuild that
   fires every run is not a trigger, it is a cron.
2. Resume from the existing checkpoint if there is one, or start fresh if there is not.
3. Score the result with lm-eval against the gate.
4. Publish the card only when the gate passes. A failed rebuild keeps the old checkpoint and says
   why, which is the point: a model that got worse should not ship because a script ran.

This is the middle of the update loop. The harvester and the drift check are the front of it; a
registry with two-version rollback is the back of it and is not written yet.

Exit codes
----------
0  every model either did not drift or passed its gate
1  a rebuild ran and failed its gate — the previous checkpoint was kept
2  a rebuild could not run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def drifted_models(corpus, models):
    args = [sys.executable, os.path.join(HERE, "corpus-drift.py"), "--corpus", corpus]
    if models:
        args += ["--models", ",".join(models)]
    out = run(args)
    changed = [l.split()[1] for l in out.stdout.splitlines() if l.strip().startswith("DRIFTED")]
    new = [l.split()[1] for l in out.stdout.splitlines() if l.strip().startswith("new")]
    return changed + new, out.stdout


def rebuild_one(model, corpus, runs, steps, batch, gate, eval_limit, prune=0.0, recover=0):
    """Train, evaluate, publish only on a pass. Returns (status, detail)."""
    target = os.path.join(runs, model)
    staging = os.path.join(runs, "." + model + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    cfg_path = os.path.join(target, "config.json")
    cmd = [sys.executable, os.path.join(HERE, "lm-train.py"),
           "--corpus", corpus, "--model", model, "--out", staging,
           "--steps", str(steps), "--batch", str(batch)]
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cmd += ["--dim", str(cfg["dim"]), "--layers", str(cfg["layers"]),
                "--heads", str(cfg["heads"]), "--ff", str(cfg["ff"]), "--seq", str(cfg["seq"])]
        if cfg.get("tokenizer") == "bpe" and os.path.exists(os.path.join(target, "bpe.json")):
            shutil.copy(os.path.join(target, "bpe.json"), os.path.join(staging, "bpe.json"))
            cmd += ["--tokenizer", "bpe", "--bpe", os.path.join(staging, "bpe.json")]
        # warm start: copy the old weights in and tell the trainer to resume from them.
        # Copying alone did nothing — lm-train ignored a model.pt that was already there.
        shutil.copy(os.path.join(target, "model.pt"), os.path.join(staging, "model.pt"))
        cmd += ["--resume"]

    trained = run(cmd)
    if trained.returncode != 0:
        tail = trained.stderr.strip().splitlines()
        return "error", tail[-1] if tail else "train failed"

    # optional structured pruning, then a short recovery, then the gate runs on the pruned model:
    # a smaller model that fails its gate must not be published just because it is smaller
    if prune and prune < 1.0:
        pruned = staging + "-pruned"
        shutil.rmtree(pruned, ignore_errors=True)
        made = run([sys.executable, os.path.join(HERE, "lm-prune.py"),
                    "--run", staging, "--keep", str(prune), "--out", pruned])
        if made.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            return "error", "prune failed"
        new_ff = json.load(open(os.path.join(pruned, "config.json")))["ff"]
        if recover:
            run([sys.executable, os.path.join(HERE, "lm-train.py"),
                 "--corpus", corpus, "--model", model, "--out", pruned,
                 "--steps", str(recover), "--batch", str(batch), "--resume", "--ff", str(new_ff)])
        shutil.rmtree(staging, ignore_errors=True)
        os.rename(pruned, staging)

    ev = [sys.executable, os.path.join(HERE, "lm-eval.py"),
          "--run", staging, "--corpus", corpus, "--model", model,
          "--limit", str(eval_limit), "--boot", "400"]
    for g in gate:
        ev += ["--gate", g]
    scored = run(ev)

    if scored.returncode == 0:
        os.makedirs(target, exist_ok=True)
        for f in ("model.pt", "model.npz", "config.json", "card.json", "bpe.json"):
            src = os.path.join(staging, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target, f))
        shutil.rmtree(staging, ignore_errors=True)
        # record the version after the files land, so v1 exists on a first publish and every
        # later publish has a predecessor to roll back to
        run([sys.executable, os.path.join(HERE, "lm-registry.py"), "publish",
             "--model", model, "--runs", runs])
        return "published", "gate passed"
    shutil.rmtree(staging, ignore_errors=True)
    tail = [l.strip() for l in scored.stdout.splitlines() if "GATE" in l]
    return "blocked", "; ".join(tail) or "gate failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="source checkouts, to re-harvest first")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", default=os.path.join(ROOT, "models"))
    ap.add_argument("--models", default="")
    # append with a non-empty default accumulates: passing --gate a=1 would leave a=0.30 in
    # the list too, and the stricter one would silently win
    ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-limit", type=int, default=12)
    ap.add_argument("--prune", type=float, default=0.0,
                    help="keep this fraction of feed-forward channels, 0 disables")
    ap.add_argument("--recover", type=int, default=0,
                    help="steps to retrain after pruning, before the gate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    if not a.gate:
        a.gate = ["json_valid=0.30"]
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    if a.root:
        args = [sys.executable, os.path.join(HERE, "lm-corpus-harvest.py"),
                "--root", a.root, "--out", a.corpus]
        if models:
            args += ["--models", ",".join(models)]
        if run(args).returncode != 0:
            print("harvest failed; a source checkout is probably missing")
            return 2

    changed, drift_text = drifted_models(a.corpus, models)
    if not changed:
        print(drift_text.strip())
        print("\nnothing to rebuild")
        return 0

    print("drifted: %s" % ", ".join(changed))
    if a.dry_run:
        print("dry run: would rebuild %d model(s)" % len(changed))
        return 0

    report = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "gate": a.gate, "models": {}}
    worst = 0
    for m in changed:
        t0 = time.time()
        status, detail = rebuild_one(m, a.corpus, a.runs, a.steps, a.batch, a.gate,
                                     a.eval_limit, a.prune, a.recover)
        report["models"][m] = {"status": status, "detail": detail,
                               "seconds": round(time.time() - t0, 1)}
        print("  %-8s %-9s %s" % (m, status, detail))
        if status == "blocked":
            worst = max(worst, 1)
        elif status == "error":
            worst = 2

    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    published = sum(1 for v in report["models"].values() if v["status"] == "published")
    print("\n%d rebuilt, %d published" % (len(changed), published))
    return worst


if __name__ == "__main__":
    sys.exit(main())
