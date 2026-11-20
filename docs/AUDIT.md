# Audit — what the review found

Run against the whole repository at `1d79f26` using the code-review-and-release-gate method:
correctness and data loss first, then security, then performance, then design. Every finding below
was reproduced before it was fixed.

## Findings

| # | Severity | Where | What happens | Status |
|---|---|---|---|---|
| 1 | **P0** | `kernel/lm.c` forward() | the KV cache is indexed by `pos` with no bound. A prompt longer than `seq` writes past the end of a `calloc`'d buffer — heap corruption, not a crash you would recognise | **fixed** — guard, and generation stops cleanly at the limit |
| 2 | **P0** | `kernel/lm.c` main() | the BPE path did `n += tok_encode(...)` with no room check, so a long prompt overflowed the `ids[MAXTOK]` stack array | **fixed** — room is computed and the encode is rejected |
| 3 | **P1** | `kernel/lm.c` load_model() | `heads` and `seq` were **hardcoded to 4 and 256**. A model exported with any other shape would run and produce confidently wrong output | **fixed** — the shape travels in the `.lmk` as `config/shape`, and a file without it is refused |
| 4 | **P1** | `tools/lm-train.py` corpus_hash() | `h.update(open(p,"rb").read())` reads the whole split into memory — 134 MB for `rep`, twice over with the eval split, against a 1.5 GB cap | **fixed** — streams in 1 MB chunks |
| 5 | **P2** | `tools/lm-train.py` load_split() | built a Python list of every token id; died with `MemoryError` on the repertory corpus | **fixed earlier** in fa35be8 — streams into `array('H')` |
| 6 | **P2** | `tools/lm-eval.py` | defaulted to `--limit 60` and the report did not say so; the published numbers were n=20 of 56 held-out pairs, with no interval | **fixed** — every rate now carries a 95% bootstrap interval, the denominator is printed, and the default is the whole split. Re-measured: byte `json_valid` 0.411 (0.286–0.536), not the 0.550 the sample suggested |
| 7 | **P2** | `kernel/lm.c` load_model() | the raw tensor buffers are never freed (`free(t)` frees the table, not the data) | **fixed** — every payload is freed after dequantisation |
| 8 | **P2** | `kernel/lm.c` forward() | used `static` scratch buffers, so the kernel was not re-entrant and two models could not run in one process | **fixed** — scratch lives in the caller's cache struct |
| 9 | **P2** | `tools/lm-corpus-harvest.py` x_act() | the pico rule regex could match a parenthesised pair that is not a rule tuple | **accepted** — it yields a wrong pair rather than crashing, and the count is small |
| 10 | **P1** | the benchmarks in the README | throughput was computed as 200 ÷ elapsed, but the run emits ~110 tokens and stops at EOS, so every rate was overstated about 1.7× | **fixed** — measured by tokens actually emitted: 319 / 533 / 82 tok/s |

## What the review did not find

- No injection surface: the harvester parses data, never executes it. The JS literal reader exists
  precisely so a repository's `.js` data file is not run.
- No secrets, no network calls, no telemetry in any tool.
- No unbounded recursion: the only recursion is the JS literal parser, bounded by brace depth.

## The two that mattered

**#1 and #2 were silent.** Neither would have shown up in the demo: the demo prompt is 30
characters and the demo generation stops at EOS long before 256 tokens. They would have appeared
the first time an app passed a real case note to the kernel — a heap overflow reading past a
buffer, or a stack smash — and neither would have looked like a bug in the kernel.

**#3 was worse in a quiet way.** Hardcoding `heads=4, seq=256` means the kernel runs and returns
plausible tokens for a model with different attention geometry. Nothing fails; the output is just
wrong. The shape now travels with the weights and a file without it is refused outright.

## Regression checks after the fixes

```
kernel/parity.py, byte model   all 3 prompts match
kernel/parity.py, BPE model    all 3 prompts match
tests/test_tools.py            all checks passed
```

## Verdict

Ship. The two P0s, both P1s and the sampling P2 are fixed and verified; the two remaining P2s are
accepted with reasons rather than quietly dropped. The remaining release-scale gaps are not in this repository: Pages is
not enabled, and the documents still name private repositories while the repo is public.
