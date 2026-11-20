# API reference

**Generated from the source by `tools/gen-api.py`.** Every option and function below is
read out of the tool file, so this page cannot describe a flag that does not exist. CI
runs `gen-api.py --check` and fails when the two disagree.

| tool | what it does |
|---|---|
| [`lm-corpus-harvest.py`](../tools/lm-corpus-harvest.py) | lm-corpus-harvest — turn the drsamonline repositories into training pairs. |
| [`lm-bpe.py`](../tools/lm-bpe.py) | lm-bpe — a byte-level BPE trainer and codec, standard library only. |
| [`lm-train.py`](../tools/lm-train.py) | lm-train — train one small model from a harvested corpus. |
| [`lm-eval.py`](../tools/lm-eval.py) | lm-eval — score a checkpoint against its held-out split, and fail when it is not good enough. |
| [`lm-export-c.py`](../tools/lm-export-c.py) | lm-export-c — turn a trained checkpoint into one portable file the C kernel can load. |
| [`corpus-drift.py`](../tools/corpus-drift.py) | corpus-drift — has the corpus moved since the last published manifest? |
| [`lm-rebuild.py`](../tools/lm-rebuild.py) | lm-rebuild — close the loop: drift in, gated checkpoint out. |
| [`lm-registry.py`](../tools/lm-registry.py) | lm-registry — versioned model storage with a way back. |
| [`lm-delta.py`](../tools/lm-delta.py) | lm-delta — how big is the difference between a base model and its continuation? |
| [`lm-prune.py`](../tools/lm-prune.py) | lm-prune — structured pruning of the feed-forward, measured before it is trusted. |
| [`gen-api.py`](../tools/gen-api.py) | gen-api — write docs/API.md from the source, so the reference cannot describe a flag that is gone. |

Usage, worked examples and the measured numbers are in the [README](../README.md).


---

## `lm-corpus-harvest.py`

lm-corpus-harvest — turn the drsamonline repositories into training pairs.

| option | takes | what it does |
|---|---|---|
| `--root` | required | directory holding the repo checkouts |
| `--out` | required | directory to write into |
| `--models` | default `"all"` | comma-separated, or all |
| `--limit` | `int` default `0` | cap pairs per model (0 = no cap) |

| function | arguments | returns |
|---|---|---|
| `sha256_bytes` | `b` |  |
| `stable_key` | `pair` | Order pairs by content, never by filesystem order. |
| `is_eval` | `pair, every=50` | A deterministic 1-in-N eval split, chosen by content hash. |
| `clean` | `text, limit=None` |  |
| `chunks` | `text, size=900, min_len=120` | Split prose on paragraph boundaries into ~size-character chunks. |
| `abbrev_density` | `text` | Fraction of tokens that look like a remedy abbreviation (Bell., nux-v., ARS.). |
| `good_prose` | `text, min_len=60, max_abbrev=0.30` | Reject OCR noise, remedy indices and fragments. |
| `load_json` | `path` |  |
| `pd_ok` | `book` | True when the volume is public domain in the US today. |
| `x_rep` | `root, skip` | Repertory: a rubric's text in, its identifier and graded remedies out. |
| `x_materia` | `root, skip` | Materia medica: a proving paragraph in, the remedy it belongs to out. |
| `x_book` | `root, skip` | Treatises and the public-domain shelf: prose in, structure out. |
| `x_act` | `root, skip` | Automation: a request in, one command or one tool out. |
| `x_audit` | `root, skip` | Review: a failure in, its cause and the guard that now catches it out. |
| `x_forge` | `root, skip` | App generation: a capability in, its contract out — and specs as targets. |
| `x_tutor` | `root, skip` | NTET: a question in, the key and its rationale out; a term in, its definition. |
| `x_verse` | `root, skip` | Poets Studio: a word in, its ARPAbet pronunciation out. |
| `x_file` | `root, skip` | Tools that act on files: a request in, the tool and its category out. |
| `main` | `` |  |


---

## `lm-bpe.py`

lm-bpe — a byte-level BPE trainer and codec, standard library only.

| option | takes | what it does |
|---|---|---|
| `--corpus` | flag |  |
| `--models` | default `"act"` |  |
| `--vocab` | `int` default `4096` |  |
| `--limit` | `int` default `0` |  |
| `--bpe` | flag |  |
| `--text` | flag |  |

| function | arguments | returns |
|---|---|---|
| `read_texts` | `corpus, models, limit=0` |  |
| `train` | `texts, vocab_size, min_count=2, verbose=True` |  |
| `main` | `` |  |


---

## `lm-train.py`

lm-train — train one small model from a harvested corpus.

| option | takes | what it does |
|---|---|---|
| `--corpus` | flag |  |
| `--model` | default `"act"` |  |
| `--eval-limit` | `int` default `200` | blocks of the eval split to score; the repertory split is 6,211 |
| `--seed` | `int` default `0` | seed for batch order; vary it to see how much of a result is noise |
| `--lora` | `int` default `0` | train low-rank adapters of this rank instead of the whole model |
| `--resume` | store_true | continue from model.pt already in --out, instead of starting fresh |
| `--out` | flag |  |
| `--dim` | `int` default `128` |  |
| `--layers` | `int` default `4` |  |
| `--heads` | `int` default `4` |  |
| `--ff` | `int` default `512` |  |
| `--seq` | `int` default `256` |  |
| `--batch` | `int` default `8` |  |
| `--steps` | `int` default `400` |  |
| `--lr` | `float` default `3e-3` |  |
| `--limit` | `int` default `0` |  |
| `--tokenizer` | default `"bytes"` |  |
| `--bpe` | default `None` |  |
| `--run` | flag |  |
| `--generate` | default `None` |  |

| function | arguments | returns |
|---|---|---|
| `make_tokenizer` | `args` | bytes (default) or a trained BPE. This is the seam the proposal calls out. |
| `render` | `pair` | One pair into (prompt_text, answer_text). |
| `load_split` | `path, tok, seq, limit=0` |  |
| `batch` | `ids, mask, seq, idx` | Cut `idx` blocks out of the stream. X is the block, Y is it shifted by one. |
| `rope_cache` | `seq, dim, base=10000.0` |  |
| `apply_rope` | `x, cos, sin` |  |
| `add_lora` | `model, r, alpha=16` | Wrap every attention and MLP projection. Embeddings and norms stay as they are. |
| `lr_at` | `step, total, base, warm=50` |  |
| `train` | `args` |  |
| `corpus_hash` | `corpus, model` |  |
| `export_int8` | `model, path` | Per-output-channel int8 quantisation. The first step of the size ladder. |
| `generate` | `run, prompt, n=60, temp=0.8, topk=20` |  |


---

## `lm-eval.py`

lm-eval — score a checkpoint against its held-out split, and fail when it is not good enough.

| option | takes | what it does |
|---|---|---|
| `--run` | required |  |
| `--corpus` | required |  |
| `--model` | required | corpus name, e.g. act |
| `--limit` | `int` default `0` | 0 = every held-out pair |
| `--boot` | `int` default `2000` | bootstrap resamples |
| `--gate` | append default `[]` | metric=threshold, repeatable, e.g. json_valid=0.9 |
| `--task` | default `""` | a task-specific scorer: rep checks the rubric id, and whether it resolves |
| `--out` | default `None` | write the result json here |

| function | arguments | returns |
|---|---|---|
| `load` | `name, path` |  |
| `generate_one` | `model, tok, prompt, seq, max_new=96` | Greedy, stopping when the JSON object closes. |
| `parse_json` | `text` | Take the first balanced {...} and parse it, or return None. |
| `field_score` | `expected, got` |  |
| `rubric_index` | `path` | Every rubric id the training split mentions — what a returned id has to resolve against. |
| `bootstrap` | `values, resamples, alpha=0.05` | Percentile bootstrap. Returns (lo, hi) for the mean. |
| `main` | `` |  |


---

## `lm-export-c.py`

lm-export-c — turn a trained checkpoint into one portable file the C kernel can load.

| option | takes | what it does |
|---|---|---|
| `--run` | required |  |
| `--out` | required |  |
| `--adapter` | store_true | export only the low-rank adapters, to be applied over a base at load |
| `--q4` | store_true | 4-bit with a per-row scale: half the file of int8, two values a byte |
| `--fp32` | store_true | export float32 from model.pt instead of int8 — for a parity test |

| function | arguments | returns |
|---|---|---|
| `write_tensor` | `fh, name, arr, dtype` |  |
| `main` | `` |  |


---

## `corpus-drift.py`

corpus-drift — has the corpus moved since the last published manifest?

| option | takes | what it does |
|---|---|---|
| `--root` | flag | directory holding the repo checkouts, to re-harvest |
| `--corpus` | flag | an existing harvest to compare instead |
| `--manifest` | default `MANIFEST` |  |
| `--models` | default `""` | comma-separated; default is every model in the manifest |
| `--keep` | store_true | leave the re-harvest on disk |

| function | arguments | returns |
|---|---|---|
| `sha` | `path` |  |
| `hashes` | `corpus, models` |  |
| `main` | `` |  |


---

## `lm-rebuild.py`

lm-rebuild — close the loop: drift in, gated checkpoint out.

| option | takes | what it does |
|---|---|---|
| `--root` | flag | source checkouts, to re-harvest first |
| `--corpus` | required |  |
| `--runs` | default `os.path.join(ROOT` |  |
| `--models` | default `""` |  |
| `--gate` | append default `[]` |  |
| `--steps` | `int` default `60` |  |
| `--batch` | `int` default `4` |  |
| `--eval-limit` | `int` default `12` |  |
| `--prune` | `float` default `0.0` | keep this fraction of feed-forward channels, 0 disables |
| `--recover` | `int` default `0` | steps to retrain after pruning, before the gate |
| `--dry-run` | store_true |  |
| `--report` | default `""` |  |

| function | arguments | returns |
|---|---|---|
| `run` | `cmd, **kw` |  |
| `drifted_models` | `corpus, models` |  |
| `rebuild_one` | `model, corpus, runs, steps, batch, gate, eval_limit, prune=0.0, recover=0` | Train, evaluate, publish only on a pass. Returns (status, detail). |
| `main` | `` |  |


---

## `lm-registry.py`

lm-registry — versioned model storage with a way back.

| option | takes | what it does |
|---|---|---|
| `--runs` | default `os.path.join(ROOT` |  |
| `--model` | flag |  |

| function | arguments | returns |
|---|---|---|
| `registry_path` | `runs` |  |
| `load_registry` | `runs` |  |
| `save_registry` | `runs, reg` |  |
| `sha` | `path` |  |
| `archive_dir` | `runs, model, v` |  |
| `snapshot` | `runs, model, v` | Copy the live checkpoint into the archive as version v. |
| `restore` | `runs, model, v` |  |
| `prune` | `runs, model, versions` | Keep the last KEEP snapshots; delete the rest. |
| `cmd_publish` | `a` |  |
| `cmd_rollback` | `a` |  |
| `cmd_list` | `a` |  |
| `main` | `` |  |


---

## `lm-delta.py`

lm-delta — how big is the difference between a base model and its continuation?

| option | takes | what it does |
|---|---|---|
| `--base` | required |  |
| `--tuned` | required |  |
| `--out` | default `""` |  |
| `--verbose` | store_true |  |

| function | arguments | returns |
|---|---|---|
| `load` | `run` |  |
| `main` | `` |  |


---

## `lm-prune.py`

lm-prune — structured pruning of the feed-forward, measured before it is trusted.

| option | takes | what it does |
|---|---|---|
| `--run` | required |  |
| `--keep` | `float` default `0.75` | fraction of ff channels to keep |
| `--out` | required |  |
| `--dry-run` | store_true |  |

| function | arguments | returns |
|---|---|---|
| `load` | `run` |  |
| `main` | `` |  |


---

## `gen-api.py`

gen-api — write docs/API.md from the source, so the reference cannot describe a flag that is gone.

| option | takes | what it does |
|---|---|---|
| `--check` | store_true | fail when docs/API.md is stale |

| function | arguments | returns |
|---|---|---|
| `summary` | `src` | First paragraph of the module docstring, on one line. |
| `functions` | `src` |  |
| `options` | `src` |  |
| `build` | `` |  |
| `main` | `` |  |
