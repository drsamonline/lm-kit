# Evaluation — the first honest numbers

Two checkpoints, same corpus (`act`, 56 held-out pairs), **20 sampled for speed** — the tool defaults
to `--limit 60` and the report never said so. Twenty of fifty-six is a small sample: treat the
figures as an indication, not a confidence interval. Greedy decoding.
The metric is on the **output**, not the loss: does it parse, does it match, which fields survive.

| Metric | byte-level | BPE 4,096 |
|---|---|---|
| `json_valid` — the text parses as a JSON object | **0.550** | 0.000 |
| `exact` — the object equals the expected one | 0.000 | 0.000 |
| `field_match` — mean fraction of fields correct | 0.058 | 0.000 |

## What this says

**Neither model can be trusted to emit valid JSON.** That is the finding, and it is the point.

The byte model gets the punctuation right half the time because a comma is one byte and it sees
thousands of them. The BPE model gets the *words* right — `Games & Comics`, `unknown`, `yes` are
real values from the corpus — and then breaks on the delimiter, because a comma is usually fused
into a neighbouring token and 180 steps is not enough to learn when it separates.

```
expected  {"api": "Forza", "auth": "none", "cat": "Games & Comics", "cors": "unknown", ...}
byte      {"api": "Scater", "auth": "none", "cat": "Developper.com", "https": "yes"}     ← parses
BPE       {"api": "Sa "auth": "none", "cat": "Development", "cors": "unknown", ...      ← missing comma
```

## Why this vindicates the design rather than undermining it

The proposal calls for **grammar-masked decoding**: the logits are masked so that only tokens
keeping the output valid against the schema can be emitted. With that in place, `json_valid` is 1.0
by construction and the model spends its capacity on the *values* — which is the part it can
actually learn at this size.

An unconstrained 1.5 M model at 180 steps is not expected to hold JSON syntax and a 2,600-entry
vocabulary at the same time. The gate exists to say so out loud instead of shipping it.

## Reproduce

```bash
python3 tools/lm-eval.py --run ../run-act      --corpus ../corpus --model act --limit 20
python3 tools/lm-eval.py --run ../run-act-bpe  --corpus ../corpus --model act --limit 20
```

With a gate that can stop a ship:

```bash
python3 tools/lm-eval.py --run ../run-act-bpe --corpus ../corpus --model act \
        --gate json_valid=0.90 --gate field_match=0.70
# GATE FAILED: json_valid 0.000 < 0.900; field_match 0.000 < 0.700
```

## Quantisation at 1 M parameters

The size ladder says Q4 halves int8. At this size it also breaks the model, and the same weights
in PyTorch produce the same break, so this is the quantiser and not the kernel.

| Weights | Size | Output |
|---|---|---|
| fp32 | 4.3 MB | `{"api": "Deanmert", "auth": "none", "cat` |
| int8, per-row scale | 1.1 MB | `{"api": "Deatang", "auth": "none", "cat` |
| **q4, per-row scale** | **0.58 MB** | `{"api": "Dainininmintinininininininmanti` |
| q4, 32-element groups | 0.58 MB | `{"api": "Deanminmi", "auth": "none", "ca` |
| q4, 16-element groups | 0.58 MB | `{"api": "Seanmininininenmens", "auth": ` |

The int8 model is barely degraded. The 4-bit one collapses into a repeating token, and finer
grouping only moves it from collapsed to nearly collapsed.

The reason is arithmetic: 1.08 M parameters with a 264-token embedding table leaves about 4,000
parameters per token, so a 7% relative error per weight is not absorbed anywhere. An 8 B model has
six thousand times the parameters to absorb it, which is why the ternary and 4-bit results in the
literature are reported at 2 B and above.

**int8 is the floor at this size.** Q4 belongs on the 20 M models, and it is unmeasured there.

## One base, several deltas

The family design says a shared base plus a small delta per speciality. Measured three ways on the
`act` model, all on the same held-out split:

| Method | Trainable | Eval loss | Delta size |
|---|---|---|---|
| base only | — | 2.0861 | — |
| full fine-tune | 1,083,520 | 1.6276 | 1.08 MB, dense |
| **LoRA rank 8** | **86,016 (7.9%)** | **1.7138** | **0.09 MB** |

A dense difference is 3.4% of the base in norm but only 27% of its entries are dead, so storing it
costs what storing the model costs. Low-rank training produces the thing the design assumed:

```
eight deltas   0.69 MB  +  one base 1.08 MB  =  1.77 MB
eight models                                  =  8.67 MB
```

LoRA reaches 1.7138 against the full fine-tune's 1.6276, so it gives up about 5% of the improvement
for 8% of the storage. That is the trade the family depends on, and it holds.

## The gates each production model must clear

| Model | Metric | Threshold |
|---|---|---|
| lm-rep | rubric in top 5 for a free-text symptom | 0.90 |
| lm-rep | identifiers returned that do not resolve | 0 |
| lm-act | command matches an existing rule | 0.85 |
| lm-act | off-catalogue requests refused | 0.95 |
| lm-forge | emitted spec compiles and passes all gates | 0.85 |
| lm-book | section boundary placed correctly | 0.80 |
| lm-materia | remedy named correctly from a proving paragraph | 0.85 |

None of these is met today, because none of these models has been trained at size. The harness
exists so that when they are, the claim is measured rather than asserted.
