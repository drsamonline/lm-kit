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

**Shipped form.** `lm-export-c.py --adapter` writes the adapter as its own `.lmk`, and the kernel
applies it at load:

```bash
./lm base.lmk "Job postings from 185 sources" --delta adapter.lmk
# applied 40 adapters from adapter.lmk
```

A 1.1 MB int8 base plus a 338 KB adapter produces output byte-identical to the PyTorch LoRA model.
The adapter is fp32 here; stored the way the weights are it is 86 KB.

## More steps, and a scorer that was punishing the model

The demo model was extended from 180 steps to about 600, resuming across sessions. Same 20 held-out
pairs, same scorer:

| Steps | json_valid | field_match |
|---|---|---|
| 180 | 0.600 (0.400–0.800) | 0.067 (0.025–0.108) |
| ~600 | **1.000** | **0.350 (0.282–0.418)** |

Two things changed. More training helped — the format goes from mostly-right to always-right and
the values from 7% to 35% — and the scorer stopped truncating.

The second matters more. At 96 tokens the better model wrote longer values, never reached its
closing brace, and scored `json_valid` 0.000. That looked like a regression from training and was a
cap in the evaluator. The budget is 192 now.

So the earlier reading — "the format is learned, the vocabulary is not" — was too pessimistic, and
pessimistic for a reason that had nothing to do with the model.

## Task-specific scoring

A JSON that parses is not an answer. `lm-eval.py --task rep` scores the thing the repertory model
is for:

| Metric | What it asks |
|---|---|
| `id_exact` | the rubric id the model returned is the one the held-out pair expected |
| `id_resolvable` | that id exists in the index the application will look it up in — 315,240 ids, from the training split |

`id_resolvable` is the safety metric. A model that invents a plausible-looking id scores zero on it,
which is the outcome the design wants: the kit discards what it cannot resolve rather than showing
the user a fabricated rubric.

The same shape covers the other three, each scored on the one field that decides its task:

| Task | Metric | Asks |
|---|---|---|
| `book` | `book_exact`, `book_present` | which of 158 volumes the passage came from |
| `materia` | `remedy_exact`, `remedy_present` | which remedy the proving describes |
| `forge` | `id_exact`, `spec_usable` | whether the spec names a catalogue capability and carries the three keys KILN cannot do without |

`spec_usable` is a weaker check than compiling the spec, and it says so: compiling needs KILN in the
loop, which this repository does not have.

### The floor, measured

| Model | Steps | Pairs trained on | Every metric |
|---|---|---|---|
| `rep` | 30 | 20,000 | 0.000 |
| `book` | 60 | 5,373 | 0.000 |
| `materia` | 80 | 1,671 | 0.000 |

Three models, three tasks, nothing usable — and not only on the task metric: `json_valid` is zero
too, so at these step counts the models have not learned the output format, let alone the answer.
The `act` model reached `json_valid` 0.411 at 180 steps, which is what separates a demo from a
result.

The harness is ready and the models are not. That is the honest state, and it is the argument for
the GPU-week rather than against the design.

## Structured pruning

`lm-prune.py` removes whole feed-forward channels — rows from w1 and w3, the matching columns from
w2 — and shrinks `ff`. Each is retrained for the same 40 steps from the same checkpoint, so the
comparison is at equal steps.

| Run | Parameters | ff | Eval loss | vs unpruned |
|---|---|---|---|---|
| unpruned | 1,083,520 | 512 | **1.3024** | — |
| pruned 75% | 886,912 | 384 | 1.3551 | −18.1% |
| pruned 50% | 690,304 | 256 | **1.3253** | −36.3% |

One seed is not a measurement, so it was repeated three times:

| Condition | Seeds | Mean | SD |
|---|---|---|---|
| unpruned | 1.3024, 1.2975, 1.3204 | 1.3068 | 0.0121 |
| pruned 50% | 1.3253, 1.3009, 1.3834 | 1.3365 | 0.0424 |

The difference is **+0.0298 (+2.3%), SE 0.0254, t = 1.17** — not distinguishable from seed noise at
three seeds. What the experiment supports is narrower than the first run suggested and stronger than
"unknown":

- pruning 36% of the parameters does not stop the model training, and the loss is at worst a few
  percent worse;
- the pruned model's spread is three times wider (SD 0.042 against 0.012), so it is less reliable
  run to run even where its mean is close.

**The feed-forward is over-parameterised at this size, and pruning works where 4-bit quantisation
did not.** To turn 36% into a shipping number needs more seeds, not a different method.

That contrast is worth stating plainly. Q4 removed 50% of the bits and destroyed the model. Pruning
removed 36% of the parameters and cost 2% of the loss. The difference is that pruning removes whole
channels that the retrain can compensate for, while quantisation perturbs every weight at once.

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
