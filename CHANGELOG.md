# Changelog

Newest first. Every entry says what changed and what it cost.

Versions 0.3.0 to 0.6.0 are recorded here and were never tagged; the first two were. v0.7.0 is the
first tag since v0.2.0 and it covers everything in between.

## 0.7.0 — 21 November 2026

**The scorer was punishing the model.** `lm-eval.py` capped generation at 96 tokens; a better-trained
model writes longer values, never reaches its closing brace, and scored `json_valid` 0.000 for it.
The budget is 192. Re-measured on the same 20 pairs:

| Steps | json_valid | field_match |
|---|---|---|
| 180 | 0.600 | 0.067 |
| ~600 | **1.000** | **0.350** |

More training helped and the cap stopped truncating. "The format is learned, the vocabulary is not"
was too pessimistic, for a reason that had nothing to do with the model.

Continued to ~840 steps, the loss kept falling and field accuracy did not: 0.350 at 600, 0.313 at
840, overlapping intervals. That is the shape of a capacity limit, and it points at the 15 M model
rather than more CPU hours.

## 0.6.0 — 20 November 2026

**The evaluator scores tasks, not just JSON.** `lm-eval.py --task rep` adds `id_exact` (was the
rubric id right) and `id_resolvable` (does that id exist in the index the app will search — 315,240
of them). `id_resolvable` is the safety metric: a fabricated id scores zero.

Scorers for `book`, `materia` and `forge` follow the same shape, each on the one field that decides
its task. Measured floor across four tasks: zero, on models trained for 30–80 steps.

Also: `lm-train.py --eval-limit`, because the repertory eval split is 6,211 blocks and scoring all
of it turned a 30-step training run into a ten-minute one.

## 0.5.0 — 20 November 2026

**Pruning works where quantisation did not.** `lm-prune.py` removes whole feed-forward channels and
shrinks `ff`. At equal recovery steps, 36% of the parameters go for +2.3% eval across three seeds
(1.3365 against 1.3068, t = 1.17) — real enough to keep, inside seed noise at n=3. Q4 removed 50%
of the bits and destroyed the model.

Also: `lm-train.py --seed`, because a single run is not a measurement, and `lm-rebuild.py --prune`
with `--recover`, so a rebuild can produce a *smaller* model and not only a newer one. The gate runs
on the pruned checkpoint, not the full one — a smaller model that fails its gate must not ship
because it is smaller.

Also in this release: `lm-export-c.py --adapter` and `lm --delta`, so an adapter ships as its own
file and the kernel applies it at load — verified byte-identical to PyTorch.

## 0.4.0 — 20 November 2026

**Two claims tested, one of them withdrawn.**

- `lm-delta.py` — measures how much a continuation differs from its base. A real rebuild differs
  by 3.4% in norm, but only 27% of entries are dead, so the delta is dense and storing it costs
  what the model costs. **Base + deltas does not save space as a difference of two fine-tunes.**
  It needs LoRA, which is not implemented.
- `--q4` — 0.58 MB against int8's 1.1 MB, and it breaks the model at 1.08 M parameters. int8 is
  the floor at this size. Measured, in `docs/EVAL.md`.
- `lm-train.py --resume` — the trainer can now continue from a checkpoint. Before this it silently
  started fresh, which means `lm-rebuild`'s warm start was a copy that changed nothing.
- `lm-train.py --lora R` — low-rank adapters. Rank 8 on the demo model is 86,016 trainable
  parameters (7.9%), reaches eval 1.7138 against the full fine-tune's 1.6276, and stores as 0.09 MB.
  Eight deltas plus one base is 1.77 MB against 8.67 MB for eight models, so the family's
  shared-base claim holds once the delta is low-rank rather than a difference of two fine-tunes.
- `lm-export-c.py --adapter` and `lm --delta` — the adapter ships as its own file and the kernel
  applies it at load. A 1.1 MB int8 base plus a 338 KB adapter reproduces the PyTorch LoRA output
  byte for byte, across 40 adapters.

## 0.3.0 — 20 November 2026

**The update loop closes.** `tools/corpus-drift.py` says whether the corpus has moved;
`tools/lm-rebuild.py` retrains a drifted model and publishes it **only if it passes its gate**.

- `lm-rebuild.py` — drift in, gated checkpoint out. A failed gate keeps the previous checkpoint and
  says why, so a model that got worse cannot ship because a script ran.
- `corpus-drift.py` — re-harvests and compares each model's SHA-256 against the published manifest.
- `lm-registry.py` — the last two versions of every checkpoint, a registry naming which is live,
  and a rollback that restores the previous one. Verified end to end: two rebuilds produce v1 and
  v2, rollback restores v1 and the live card names it.
- `.github/workflows/corpus.yml` — weekly and on demand; warns and stops without `CORPUS_TOKEN`
  rather than failing on a missing directory. The rebuild step is opt-in.
- Two argparse traps fixed on the way: `--gate` accumulated its default, so the stricter threshold
  silently won, and `tempfile` defaulted to `/tmp` with a 134 MB corpus.

## 0.2.0 — 20 November 2026

**The kernel runs without Python, and the numbers are honest.**

- `kernel/lm.c` — a 400-line inference kernel, libc and libm only, 56 KB native and 44 KB as
  WebAssembly. Verified against PyTorch: three prompts per tokenizer, byte-identical.
- BPE tokenisation in C, with the merges carried inside the `.lmk`.
- `tools/lm-export-c.py` — checkpoint to `.lmk`, int8 or fp32, with the shape in the file so the
  kernel cannot guess it.
- `tools/lm-eval.py` — scores the output, not the loss, with 95% bootstrap intervals and a gate
  that can fail.
- `tools/gen-api.py` — `docs/API.md` generated from the source; CI fails when it is stale.

**Corrections.** Throughput was overstated 1.7× because the benchmark divided a fixed 200 by the
time of a run that emitted ~110 tokens (527 → 319, 1,016 → 533, 157 → 82). The eval rate was
published from a 20-item sample and is now measured on all 56 pairs (0.550 → 0.411, 0.286–0.536).

**Fixes.** Two heap overflows in the kernel — an unbounded KV cache index and an unbounded id
buffer. A trainer that died with `MemoryError` on the 359 k-pair corpus. A corpus hash that read
268 MB into memory.

## 0.1.0 — 20 November 2026

**The pipeline works.**

- `tools/lm-corpus-harvest.py` — nine extractors, 503,159 pairs, deterministic across runs, with a
  licence gate that refuses a volume not yet public domain in the US.
- `tools/lm-bpe.py` — a 4,096-token BPE, 3.29 characters per token.
- `tools/lm-train.py` — byte and BPE, masked loss, int8 export, model card naming the corpus hash.
- Two demo checkpoints and their cards.
- `DISCLAIMER.md` and `NOTICE.md` — medical and AI scope, third-party attribution.
