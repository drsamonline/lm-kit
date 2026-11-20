<div align="center">

# lm-kit

![Licence](https://img.shields.io/badge/licence-PolyForm%20Noncommercial%201.0.0-2e7d32)
![Python](https://img.shields.io/badge/python-3.8%2B-3776ab)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![Network](https://img.shields.io/badge/network-not%20required-brightgreen)
![Status](https://img.shields.io/badge/status-pipeline%20built%2C%20models%20not%20trained-orange)

**Small, schema-locked language models for the drsamonline fleet.**

Harvest the corpus from the repositories · train a tiny transformer · export int8 · ship it
beside the app. One file per stage, standard library only, no network, no API key.

</div>

---

## The boundary — read this first

This kit builds models that turn a sentence into **the JSON an app already knows how to use**.
It does not build chat. A model here is a function with weights: text in, a schema-valid object
out, and a refusal when the request does not fit.

**Nothing in this repository is a diagnostic or a clinical engine.** Several apps that will use
these models are clinical. Mapping case text to rubric identifiers is a retrieval aid. It is not a
clinical judgement and must never be presented as one.

The fleet already has both ends of this. `ml-kit` is light AI — classic algorithms, no model.
The user's own key through `sui` or `ai-bridge` is heavy AI. This is the missing middle.

## Status

| | |
|---|---|
| Analysis | **complete** — 157 repositories, 73,426 files, 2,042,289 lines |
| Model family | **decided** — eight core models, one shared base with eight deltas |
| Corpus harvester | **working** — 503,159 pairs across nine models, deterministic |
| Tokenizer | **working** — 4,096-token BPE, 3.29 characters per token |
| Trainer | **working** — byte and BPE, masked loss, int8 export, model card |
| Production models | **not trained** — one GPU-week per model |

## Quickstart

```bash
# 1. harvest — point it at a directory holding the repo checkouts
python3 tools/lm-corpus-harvest.py --root ../src --out ../corpus

# 2. tokenizer
python3 tools/lm-bpe.py train --corpus ../corpus --models act --vocab 4096 --bpe bpe-act.json
python3 tools/lm-bpe.py stats --bpe bpe-act.json --corpus ../corpus --model act

# 3. train (CPU is enough to prove the pipeline; a GPU is needed to ship)
python3 tools/lm-train.py --corpus ../corpus --model act --out ../run \
        --tokenizer bpe --bpe bpe-act.json --steps 180 --batch 4

# 4. sample
python3 tools/lm-train.py --run ../run --generate "Job postings from 185 sources"
```

## What it produced

```
rep       359432 pairs  rubric text → id, book, section, graded remedies
verse     133309 pairs  word → ARPAbet pronunciation, from the CMU dictionary
book        5502 pairs  prose → book, author, section
act         2626 pairs  description → tool or API, with constraints
materia     1709 pairs  proving paragraph → remedy
tutor        392 pairs  question + options → key and rationale
forge         98 pairs  capability or pitch → KILN spec
file          72 pairs  description → the file tool that does it
audit         19 pairs  lesson or issue → guard, severity, status
```

Two runs of the same checkouts produce byte-identical files and identical SHA-256s. The
licence gate skipped one book — Bach's *Free Thyself* (1932), which enters the US public
domain on 1 January 2028 — and recorded why.

## The demo run, honestly

Same corpus, same shape, same 180 steps, same CPU. Only the tokenizer changes.

| | byte-level | BPE 4,096 |
|---|---|---|
| chars per token | 1.00 | 3.29 |
| parameters | 1,083,520 | 1,514,240 |
| eval loss per token | 1.0083 | 3.2267 |
| **bits per character** | 1.4547 | **1.4149** |

The per-token row is a trap: divide by characters per token and BPE is 2.7% better. What it
generates is the point — the byte model produces `"cat": "Bungorincdinaly"`, the BPE model
produces `"cat": "Games & Comics"`. The format is learned; the vocabulary is not. That is the
correct result for 1.5 M parameters and three CPU minutes, and it is what the 22 M model with a
GPU-week closes.

## The family

| Model | Job | Params | Q4 |
|---|---|---|---|
| lm-book | read a book, return its structure | 24 M | 13 MB |
| lm-materia | a proving, as structured remedy knowledge | 30 M | 16 MB |
| lm-rep | case text → rubric IDs, never a remedy | 48 M | 26 MB |
| lm-act | one sentence → one command | 15 M | 8 MB |
| lm-tutor | question, answer, rationale | 34 M | 18 MB |
| lm-forge | idea → spec, or spec → patch | 60 M | 33 MB |
| lm-audit | diff or log → findings and a verdict | 24 M | 13 MB |
| lm-fleet | route and compose across the seven | 90 M | 48 MB |

Eight group models share one 3 M base plus a small delta each — nine specialities for about 4 MB.

## Repository layout

```
tools/     the three stages — harvest, tokenize, train
docs/      the dossier and the five proposals that led to it
models/    the corpus manifest and the two demo checkpoints with their cards
tests/     a smoke test that runs without the repositories or the network
```

## Documentation

Start with `docs/LM-KIT-project-dossier.docx` — eight pages covering the whole arc: how the fleet
was read, what it contains, which models are worth building, what the corpora measure, what was
built and run, what the external research changed, and what is left.

The five earlier documents are kept as the detailed per-topic versions.

## Licence

PolyForm Noncommercial 1.0.0 — free for personal use, commercial use requires a licence.
