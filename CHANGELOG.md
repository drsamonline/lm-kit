# Changelog

Newest first. Every entry says what changed and what it cost.

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
