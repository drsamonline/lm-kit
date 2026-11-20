# lm-train

**Trains one small model from a harvested corpus.** Decoder-only transformer, CPU-capable, PyTorch.

```bash
python3 lm-train.py --corpus ../corpus --model act --out ../run-act --steps 180 --batch 4
python3 lm-train.py --run ../run-act --generate "Job postings from 185 sources"
```

## The demo run, honestly

The `act` corpus — 2,570 training pairs of description → tool or API — trained for 180 steps on a
1.08 M-parameter model, in 167 seconds on this CPU.

```
step    0  loss 121.5439
step   36  loss   3.3843
step   90  loss   1.3213
step  179  loss   1.3886
eval loss 1.0083  (perplexity 2.7)
```

And here is what it generates, unedited:

```
IN: Job postings from 185 sources
→ {"api": "BBdicegi", "auth": "none", "cat": "Bungorincdinaly"

IN: Movie and TV data
→ {"api": "BBananal.aa", "auth": "none", "cat": "Trine", "cors
```

The shape is right and the content is wrong. It learned that an answer is a JSON object with
`api`, `auth`, `cat`, `cors` and `https`, in that order, quoted the way JSON quotes — and it has not
learned a single API name. That is exactly what 1.08 M parameters and three CPU minutes buy, and it
is worth stating plainly rather than showing a cherry-picked sample: **the format is learned, the
vocabulary is not.** The 20 M model with a 10,240-token BPE and a GPU-week is what closes that gap,
and this run is the evidence that everything around the gap already works.

## What is actually correct here

**The loss is masked to the answer.** Each pair renders as `IN: <input>` then `OUT: <json>`, and the
cross-entropy is computed only on the OUT half. Without that the model spends most of its capacity
learning to reproduce the input, which is the most common way a small instruction model is wasted.

**Pairs are packed, not padded.** The stream of all pairs is concatenated and cut into blocks of
`seq+1`. Padding each short pair to 256 tokens would have thrown away 99% of the compute — the first
version of this script produced 6 training blocks from 2,570 pairs for exactly that reason.

**The split is by hash, not by file order.** `eval` blocks come from the 1-in-50 that
lm-corpus-harvest marked, so the held-out set is the same set every time, on every machine.

**The card records the data.** `card.json` carries the corpus SHA-256 alongside the configuration
and the eval loss. A model that cannot name the data it saw cannot be audited, and this fleet audits.

## The tokenizer, measured

Same corpus, same 1.08 M-parameter shape, same 180 steps, same CPU. The only change is the
tokenizer: bytes (264 symbols) versus a 4,096-token BPE trained on the same corpus in 40 seconds.

| | byte-level | BPE 4,096 |
|---|---|---|
| chars per token | 1.00 | 3.29 |
| parameters | 1,083,520 | 1,514,240 |
| train time | 167 s | 221 s |
| eval loss (per token) | **1.0083** | 3.2267 |
| perplexity | 2.7 | 25.2 |
| **bits per character** | **1.4547** | **1.4149** |

Read the per-token row and BPE looks three times worse. It is not: a token now carries 3.29
characters instead of one, so the two numbers are not the same quantity. Divided through by
characters per token, BPE is 2.7% better — and the generated text shows why that matters more than
the number:

```
byte-level   → {"api": "BBdicegi", "auth": "none", "cat": "Bungorincdinaly"
BPE          → {"api": "SDD", "auth": "none", "cat": "Games & Comics", "cors": "unknown",
                "host": "cems..io", "https": "yes"}
```

`Games & Comics` and `Sports & Fitness` are real categories from the corpus; `none`, `unknown` and
`yes` are real values. The host is still wrong, but it is wrong in the shape of a host. A model that
must spell every word one byte at a time cannot produce a vocabulary in three CPU minutes; one that
can emit `Video` as a single token can.

That is the whole argument for the 10,240-token tri-lingual vocabulary in the proposal, and this run
is the evidence for it.

## The tokenizer seam

Byte-level: 256 symbols plus three specials. No training step, no dependency, and deliberately the
weakest part of the pipeline — a byte-level model spends its capacity on spelling. Production swaps
in the 10,240-token BPE over English, Hindi and Gujarati, and `Tokenizer` is the only class that
changes:

```python
class Tokenizer:
    def encode(self, text): ...
    def decode(self, ids): ...
    def vocab_size(self): ...
```

## Export

| file | what it is |
|---|---|
| `model.pt` | the torch state dict, fp32 |
| `model.npz` | int8 weights with per-output-channel scales — 1.1 MB for 1.08 M params |
| `config.json` | the shape, so the model can be rebuilt without the training script |
| `card.json` | corpus hash, steps, params, train and eval loss, date |

`export_int8` is the first rung of the size ladder: fp32 → int8 at roughly 4× smaller, with the
scales kept so the weights can be dequantised exactly. Q4, pruning and the delta scheme come later;
this proves the seam exists.

## What a real run needs

The configuration in the proposal — 10 layers, 384 wide, 6 heads, 10,240 vocabulary — is 22 M
parameters. At this script's throughput (about 1 step per second at batch 4, seq 256, on four CPU
cores) that is roughly 40 days. On one GPU it is about a week. Nothing in the script changes except
`--dim`, `--layers`, `--heads`, `--ff` and the tokenizer.
