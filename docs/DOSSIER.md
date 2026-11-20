# lm-kit — project dossier

**Everything, in one place.** The fleet read, the model family decided, the corpora measured, the
pipeline built and run, the external research assessed, the mistakes made and corrected, and what
remains.

Version 1.0 · 20 November 2026 · Dr Sohil Momin

> This is the Markdown edition, for reading on GitHub and on a phone. The `.docx` files in this
> folder are the same material formatted for circulation. Where the two disagree, this one is
> generated from the repository and the `.docx` is older.

---

## 1. What this is

The single record of the work. It replaces five separate documents with one that carries the whole
arc: how the fleet was read, what it turned out to contain, which models are worth building and why,
what the corpora actually measure, what has been built and run, what the outside research changed,
and what is still open.

| If you want | Read | Because |
|---|---|---|
| the conclusion in one page | §4 and §6 | the finding and the family |
| the evidence | §2, §3, §9 | counts, corpora, provenance |
| what exists right now | §11, §16 | measured runs and file names |
| what to do next | §17 | ordered, with reasons |
| what went wrong | §14 | kept on purpose |

**Status in one line.** The analysis is complete, the family is decided, the harvester and the
trainer both work and have been run end to end, and the kernel runs the model without Python.
Nothing has been trained at production size, because that needs a GPU-week per model.

## 2. How the fleet was read

Every repository was inventoried by file tree, then read. It took three passes, because the first
two were not enough.

| Pass | What it covered | What it produced |
|---|---|---|
| 1 | 5 repositories, read by hand | a proposal built on five data points. Wrong in three factual ways, and blind to most of the fleet |
| 2 | 157 file trees and 157 READMEs | 73,426 files, 7,504.8 MB, every repository classified by role. Still only description-level |
| 3 | 112 cloned and profiled, 45 read through trees and key files | 2,042,289 lines scanned, every API surface and data schema read |

The third pass changed the design. Reading code rather than descriptions is what revealed that the
kit layer is 15,000 lines rather than a framework, that `Sorta` already classifies and explains in
Rust, and that `pico`'s automation contract was already written and already running a 2 GB model.

## 3. The fleet in numbers

| Measure | Value | Note |
|---|---|---|
| Repositories | 157 | 146 public, 11 private |
| Files | 73,426 | |
| Total size | 7,504.8 MB | 634.9 MB code, 401.6 MB data |
| Lines scanned | 2,042,289 | excluding vendored third-party trees |
| Workflow files | 383 across 141 repos | 1.28 MB of CI YAML |
| Machine-readable text | ~400 MB | on the order of 50–60 M tokens |

| Role | Count | Examples |
|---|---|---|
| Corpus — feeds a model | 35 | the homoeopathy JSON library, corpus-extendum, TOOLWORKS, forge, audit, skills |
| Consumer — would call one | 58 | repertory-reader, HMS, SmartNotes, pico, CLDR, All4ONE |
| Kit — the substrate | 33 | ml-kit, search-kit, repertory-kit, web-kit, form-kit |
| Runtime — executes | 16 | NEO, volumen, nodus, sorex, The_Browser, weft8-kit |
| Delivery — ships | 5 | app-builder, frame-kit, helioprism, BLINKTRANSFER, blinkcode-ai |
| Infrastructure | 10 | forge, audit, registry, knowledge, control-panel, tessera |

Thirty repositories feed no model at all. That is why the family is small: the games, the lists and
the demos consume the platform rather than feeding it.

## 4. The finding

The fleet has two AI paths and a hole between them. **Light AI** is `ml-kit`: thirteen classic
algorithms in 300 lines, no model, no weights, no network. **Heavy AI** is the user's own key
through `sui` or `ai-bridge`. Nothing exists in the middle, and the middle is where nearly every app
sits.

More usefully: **not one app in the fleet needs free-form generation.** Every one needs messy human
text in, and a schema the app already owns out.

| Repository | What the user types | What the app needs back |
|---|---|---|
| SmartNotes | "call the lab about the repertory reprint before friday, high" | `{category, priority, due}` |
| KILN | "a tea log with tags and search" | `{archetype, fields[], capabilities[]}` |
| All4ONE / TTB | "shrink these photos and strip the metadata" | `{tool, args[]}` |
| pico / CLDR | "how many lines in notes.txt" | one command matching a rule |
| repertory-reader | "the rubric about grief with silent weeping" | `{rubric_ids[]}` |
| remedy-finder | "worse from cold air, better from warmth" | `{symptom_ids[], remedies[]}` |
| Cobble | "a stone about the smell of rain on the bus" | `{title, tags, body}` |

That is classification, slot-filling and routing. All three are solved at tens of millions of
parameters. Conversation is not. So the model is built as the first thing and never as the second.

## 5. Four tiers, and the boundary

| Tier | What it is | Size | Network | Status |
|---|---|---|---|---|
| 0 Light | ml-kit — tf-idf, Naive Bayes, k-NN, k-means, PCA, ID3, Markov, grammar | 0 KB | never | built |
| 1a Nano | 3.3 M params, for Weft-8 and microcontrollers | ~2 MB | never | proposed |
| 1b Local | 15–90 M params, schema-locked | 8–48 MB | never | proposed |
| 2 Heavy | the user's own key — sui, ai-bridge, any frontier model | gigabytes | yes | built |

The rule that makes both new tiers safe: **lm-kit must refuse to be tier 2.** When a request does
not fit a schema it returns a handoff, not a guess. That refusal is what lets a 13 MB file sit
inside a clinical app without pretending to be a clinician.

## 6. The model family

| Model | Job | Params | Q4 | Consumers |
|---|---|---|---|---|
| lm-book | read a book, return its structure | 24 M | 13 MB | corpus-extendum, organon-reader, book-kit, volumen |
| lm-materia | a proving, as structured remedy knowledge | 30 M | 16 MB | open-materia-medica, app-thermals, mnemosyne |
| lm-rep | case text → rubric IDs, never a remedy | 48 M | 26 MB | repertory-reader, UniRep, remedy-finder, HMS, case-kit |
| lm-act | one sentence → one command | 15 M | 8 MB | pico, CLDR, All4ONE, TTB, kitbox |
| lm-tutor | question, answer, rationale | 34 M | 18 MB | ntet-homoeopathy-cbt, skills, Matneo |
| lm-forge | idea → spec, or spec → patch | 60 M | 33 MB | KILN, app-builder, forge |
| lm-audit | diff or log → findings and a verdict | 24 M | 13 MB | audit, forge, control-panel, every CI |
| lm-fleet | route and compose across the seven | 90 M | 48 MB | every app, as the single entry point |
| 8 group deltas | one base, eight specialities | 3–6 M | 1.7–3.3 MB | lm-tag, lm-act-n, lm-clinic, lm-kid, lm-sky, lm-verse, lm-tale, lm-file |

Installed together the family is about 100 MB after compression. Installed as intended, an app
carries one of them. They share one tokenizer, one kernel and one schema pack, so a second model on
a device costs the weights alone.

**Why seven specialists rather than one model.**

- The output vocabularies do not resemble each other: pico's target is one of 114 regexes,
  lm-rep's is an identifier from 20,786 remedy keys, lm-forge's is nested JSON with 49 capability
  names.
- The consumers do not overlap: an app that reads a book never routes a shell command.
- They fail differently, and one model would have to satisfy every refusal at once.

## 7. Why those sizes

Capability is roughly parameters × tokens × data quality × narrowness. The first three cost money;
the fourth is free and is the one the design controls.

TinyStories is the proof: 3.5 M parameters producing coherent multi-sentence stories, because the
corpus has a closed vocabulary and a repeating grammar. Every operation in the family has that
shape — a closed output vocabulary (your field names, tool names, rubric IDs) and a repeating
grammar (your JSON schema).

The second reason the models stay small: **the facts do not belong in the weights.** 357,762
rubrics, 1,435,488 graded links, 344 materia medica portraits and 969 tool descriptions would need a
model a thousand times larger, would be stale on ship day, and could not be audited. `search-kit`
already retrieves; `repertory-kit` already scores. The model translates, the kits remember.

That division is also the safety story: a model that cannot recall a rubric cannot invent one. It
emits an ID, the kit resolves it, and an ID that is not in the index is discarded.

## 8. The size ladder

| Params | fp16 | int8 | Q4 | Q2 | Ternary |
|---|---|---|---|---|---|
| 1.5 M | 3.0 MB | 1.5 MB | 0.8 MB | 0.4 MB | 0.33 MB |
| 3 M | 6.0 MB | 3.0 MB | 1.7 MB | 0.8 MB | 0.66 MB |
| 6 M | 12 MB | 6.0 MB | 3.3 MB | 1.7 MB | 1.3 MB |
| 20 M | 40 MB | 20 MB | 11 MB | 5.6 MB | 4.4 MB |
| 48 M | 96 MB | 48 MB | 26 MB | 13 MB | 10.5 MB |
| 60 M | 120 MB | 60 MB | 33 MB | 17 MB | 13 MB |

| Technique | Realistic gain | Where it applies |
|---|---|---|
| Structured pruning + retrain | **36% measured** | 690,304 params from 1,083,520. Three seeds: +2.3% eval, t = 1.17 — real, but inside seed noise |
| Low-rank factorisation of MLPs | 10–25% | small quality cost |
| Vocabulary pruning | up to 45% | only on an inherited base |
| Entropy coding | 3–8% | the final archive |
| Ternary weights | 2.5× | proven at 2 B (BitNet 2B4T: 0.4 GB, 30–45 tok/s on a laptop CPU); unproven at 3 M |

**Measured exception.** Q4 breaks the 1.08 M demo model: per-row scales collapse the output to a
repeating token, and 32-element groups only get it back to nearly collapsed. int8 survives with
almost no visible degradation. The Q4 column above applies to the 20 M models and up; at 1 M,
int8 is the floor. Details in [`EVAL.md`](EVAL.md).

| Tier | Budget | Runtime | What fits |
|---|---|---|---|
| S script | 0.8–2 MB | plain JavaScript, no WASM | 3 M params at Q4 as base64 in one file |
| W wasm | 5–16 MB | the C kernel compiled to WebAssembly | 8–30 M params |
| N native | 16–50 MB | the same kernel, native | 30–90 M params |

## 9. The corpora

| Source | What it yields | Scale |
|---|---|---|
| homoeopathic JSON library | rubric text → id, section, graded remedies | 72 books · 357,762 rubrics · 1,435,488 links |
| materia medica volumes | proving paragraph → remedy | 15 volumes · 39,770 symptoms · 344 portraits |
| treatises | prose → book, section | 41 volumes · 5,731 sections |
| corpus-extendum | 79 books + 79 full texts | 158 volumes, India and US verified |
| TOOLWORKS | description → tool, category, danger flag | 969 tools, 5 flagged dangerous |
| app-endpoints | description → API, auth, CORS | 2,214 entries |
| pico | command template → rule, handler | 114 rules across 12 packs |
| KILN | pitch → spec; capability → contract | 49 capabilities · 37 specs · 599 tests |
| forge | 154 specs · 184 master-list entries · 20 prompts · 18 lessons | the app-making procedure, written down |
| audit | 193 dated entries with before-and-after evidence | narrative, not labelled pairs |
| ntet | question bank with rationales | 280 public, 400 private |
| skills | 11,692 skill folders | 408,741 lines — the largest corpus in the fleet |

The honest constraint: **50–60 M tokens of your own text is a fine-tuning corpus, not a pretraining
one.** Every model therefore assumes a public-text pretraining stage or an initialisation from an
existing small base, then distillation onto your data.

## 10. The pipeline

Four stages, plus a kernel.

**Harvest.** `tools/lm-corpus-harvest.py` walks the checkouts, reads the data files already in
them, and writes one JSONL per model plus a manifest. Nine extractors. It refuses three things: a
book not yet public domain in the US, OCR noise, and a split chosen by file order.

```
rep       359432 pairs  rubric text → id, book, section, graded remedies
verse     133309 pairs  word → ARPAbet pronunciation
book        5502 pairs  prose → book, author, section
act         2626 pairs  description → tool or API, with constraints
materia     1709 pairs  proving paragraph → remedy
tutor        392 pairs  question + options → key and rationale
forge         98 pairs  capability or pitch → KILN spec
file          72 pairs  description → the file tool that does it
audit         19 pairs  lesson or issue → guard, severity, status
```

**Tokenize.** `tools/lm-bpe.py` trains a byte-level BPE. Incremental merge counting, so 4,096
merges finish in about forty seconds.

**Train.** `tools/lm-train.py` trains a decoder-only transformer with RMSNorm, SwiGLU, RoPE, tied
embeddings and causal attention. The loss is masked to the answer half of each pair.

**Evaluate.** `tools/lm-eval.py` scores the output, not the loss, with a gate that can fail.

**Run.** `kernel/lm.c` runs the model without Python.

## 11. Measured results

**Training.** Same corpus, same shape, same 180 steps, same CPU. Only the tokenizer changes.

| | byte-level | BPE 4,096 |
|---|---|---|
| chars per token | 1.00 | 3.29 |
| parameters | 1,083,520 | 1,514,240 |
| train time | 167 s | 221 s |
| eval loss per token | 1.0083 | 3.2267 |
| **bits per character** | 1.4547 | **1.4149** |

Read the per-token row and BPE looks three times worse. Divide by characters per token and it is
2.7% better. The generated text settles it:

```
byte-level  → {"api": "BBdicegi", "auth": "none", "cat": "Bungorincdinaly"
BPE         → {"api": "SDD", "auth": "none", "cat": "Games & Comics",
               "cors": "unknown", "host": "cems..io", "https": "yes"
```

**Evaluation.** All 56 held-out pairs, greedy, 95% bootstrap interval over 2,000 resamples.

| Metric | byte-level | BPE 4,096 |
|---|---|---|
| `json_valid` | **0.411** (0.286–0.536) | 0.000 |
| `exact` | 0.000 | 0.000 |
| `field_match` | 0.054 (0.030–0.083) | 0.000 |

Re-measured at a 192-token budget: the byte model emits valid JSON **1.000** of the time after
about 600 steps, with **0.350** of fields correct. The earlier figures were truncated by a 96-token
cap in the scorer. That is the measured case for
grammar-masked decoding, which makes `json_valid` 1.0 by construction and gives the capacity back
to the values.

**Kernel.** Verified against PyTorch: three prompts on each tokenizer, byte-identical.

| | size | throughput, one core |
|---|---|---|
| native, `-O2`, byte model | 56 KB | 319 tok/s |
| native, `-O2`, BPE model | 56 KB | **533 tok/s** |
| wasm32-wasi, `-Os` | **44 KB** | 82 tok/s |

The first published figures were 527, 1,016 and 157 — wrong, because the benchmark divided a fixed
200 by the time of a run that emitted about 110 tokens and stopped at EOS. The denominator was the
error, and it is the same class of mistake as quoting a rate over twenty items.

**Scale.** The repertory corpus streams to 305,920 blocks: 78 M tokens in 157 MB, loaded in 33 s.

## 12. What the external research changed

Sixty-one links on CPU-only and small-model work. Thirty-four were read in full.

| Source | What it says | What it changes |
|---|---|---|
| awesome-ai-cpu | ~40 CPU engines, including deepseek.cpp (~2k LOC), gpt-oss.java (~1,000 lines), MNN (~800 KB core) | the C kernel is a solved problem to copy, not invent |
| reame | ternary-first server; 8B in 2.2 GB; 2.125-bit interleaved format | the packing scheme for the nano tier |
| on-prem-llm-cpu | 168 runs on a 2016 Xeon: decode 9–22 tok/s, cold prefill 5–12 minutes at 16k, warm ~0.1 s | prefill and decode must be gated separately |
| cpubrrr | quad-interleaved layout; a 7.5× claim that collapsed to 1.17× once the baseline was tuned | a baseline is a configuration, not a binary |
| web-llm | JSON-mode structured generation in the WASM layer | confirms grammar masking |
| BitNet / MINI-LLM / LLM Compressor | ternary at 2 B; structured pruning; 2:4 sparsity at 98.4% recovery | the compression ladder has precedent |

The gap worth naming: none of the sixty-one is about building a 20 M-parameter task model. They
serve the kernel and the size ladder; they say nothing about distilling 359,432 rubric pairs into
26 MB, which is the part that is novel here.

## 13. Decisions

| Decision | Chosen | Rejected, and why |
|---|---|---|
| What the model does | text → schema, fixed operations | chat and summarisation: 10× the size, breaks every guarantee |
| One model or seven | seven specialists plus a router | one 22 M model: three vocabularies in one head |
| Where facts live | search-kit and repertory-kit | memorisation: 1,000× the size, stale, unauditable |
| How the app maker works | spec mode and patch mode, compiled by KILN | emitting whole files: 15,000 tokens with no second chance |
| The combiner | a 90 M router first | merged generalist: its failure mode is a plausible wrong answer |
| Runtime | a ~600-line C kernel, three tiers | llama.cpp: 300 KB of runtime on a 50 KB app |
| Distribution | one shared asset per device | per-app bundles: fifty copies |
| Evaluation | frozen split by content hash | file-order splits: not reproducible |

## 14. What went wrong

Kept deliberately, because each one changed a number or a design.

| Claim | Corrected to | Consequence |
|---|---|---|
| 74,480 repertory rubrics | 357,762 rubrics, 1,435,488 links | lm-rep is the only pretraining-scale corpus |
| 403 All4ONE tools | 969 tools in TOOLWORKS, plus 493 in kitbox | lm-act's vocabulary is 2.4× larger |
| a C kernel is plausible | weft8-hardware already runs a VM in C on an Arduino | the microcontroller tier became real |
| 6.5 GB of source in dumps | 2,148 + 1,463 MB of duplicate tarballs | not an extra corpus; stopped counting it |
| a repair model on knowledge | three issues, zero logs | withdrawn; lm-audit uses forge's 18 lessons |
| 400 NTET questions | 280 public, 400 private | the eval set must use 280 |
| NEO lowers to C via a backend crate | emitted from compiler/src/backend/mod.rs | the binding plan was adjusted |
| one 22 M model, seven operations | seven specialists | the family changed shape |
| `json_valid` 0.550 | 0.411 (0.286–0.536) on the full split | n=20 was sampling noise, not a measurement |
| 527 / 1,016 / 157 tok/s | 319 / 533 / 82 tok/s | the benchmark divided by a token count the run never emitted |
| `load_split` was fine | it died on the 359 k-pair corpus | streamed into `array('H')` |

## 15. Refusals and risks

| Model | Must refuse |
|---|---|
| lm-book | answering questions about the text; summarising beyond a section |
| lm-materia | prescribing, potency, dose; any clinical claim |
| lm-rep | naming a remedy or a diagnosis; any ID it cannot resolve |
| lm-act | anything outside the catalogue; any dangerous tool without confirmation |
| lm-tutor | medical advice; any citation that does not resolve |
| lm-forge | any capability outside the catalogue; any app not traceable to a master-list entry |
| lm-audit | passing a build it did not measure |
| lm-fleet | composing across more than one specialist in a single pass |

| Risk | Mitigation |
|---|---|
| not enough tokens | pretrain on public text, or initialise from SmolLM2-135 M and distil down |
| licence drift | the harvester gates on `copyright_jurisdictions.US` and records the skip |
| Hindi and Gujarati are thin | ship English first; add HI and GU when the data exists |
| clinical overreach | IDs only; repertory-kit scores; the boundary travels with the weights |
| automation blast radius | 5 of 969 tools are flagged dangerous and blocked without confirmation |
| model rot | the update loop, with a red gate that stops a ship |
| compute | one GPU-week per core model; each earns its week |

## 16. What exists

| Artifact | What it is |
|---|---|
| `tools/lm-corpus-harvest.py` | nine extractors, deterministic, licence-gated |
| `tools/lm-bpe.py` | BPE trainer and codec |
| `tools/lm-train.py` | the trainer, masked loss, int8 export, model card |
| `tools/lm-eval.py` | scores the output with bootstrap intervals and a gate |
| `tools/lm-export-c.py` | checkpoint → `.lmk` |
| `tools/gen-api.py` | generates `docs/API.md` from the source |
| `kernel/lm.c` | the inference kernel, verified against PyTorch |
| `kernel/parity.py` | the proof |
| `models/` | the corpus manifest and two demo checkpoints with cards |
| `docs/API.md` | generated reference, checked in CI |
| `docs/AUDIT.md` | nine findings from the release-gate review |
| `docs/EVAL.md` | measured numbers with intervals |

## 17. What is left

| Order | Work | Blocked by | Effort |
|---|---|---|---|
| 1 | lm-rep at production size | one GPU-week | 1 week |
| 2 | the integer kernel and quad-interleaved layout | nothing; 1,016 tok/s is the number to beat | 2 weeks |
| 3 | two more extractors (note, sky) | no labelled data exists for either | unknown |
| 4 | ship lm-act | nothing; replaces qwen2.5:3b in production | 2 weeks |
| 5 | 10,240 tri-lingual BPE | Hindi and Gujarati parallel text does not exist yet | 1 week + data |
| 6 | Q4, pruning and delta export — all three built and measured; pruning needs more seeds | done | — |
| 7 | CI update loop — drift, gated rebuild and two-version rollback are built; only the push trigger is missing | the harvester is deterministic, so the hash is the trigger | 1 day |
| 8 | spec normalisation for lm-forge | 154 specs, 0 canonical | 1 week |
| 9 | delivery: EXE, APK, air-gap, seal | app-builder, frame-kit, lockhtml exist | 2 weeks |

---

**In one sentence.** The fleet does not need a small chatbot; it needs eight small functions that
turn a sentence into the JSON an app already knows how to use, trained on the 357,762 rubrics and
969 tools you already wrote, and honest enough to hand off everything else.
