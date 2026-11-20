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
| `--out` | default `None` | write the result json here |

| function | arguments | returns |
|---|---|---|
| `load` | `name, path` |  |
| `generate_one` | `model, tok, prompt, seq, max_new=96` | Greedy, stopping when the JSON object closes. |
| `parse_json` | `text` | Take the first balanced {...} and parse it, or return None. |
| `field_score` | `expected, got` |  |
| `bootstrap` | `values, resamples, alpha=0.05` | Percentile bootstrap. Returns (lo, hi) for the mean. |
| `main` | `` |  |


---

## `lm-export-c.py`

lm-export-c — turn a trained checkpoint into one portable file the C kernel can load.

| option | takes | what it does |
|---|---|---|
| `--run` | required |  |
| `--out` | required |  |
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
