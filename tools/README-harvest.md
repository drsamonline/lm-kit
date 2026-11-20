# lm-corpus-harvest

**Turns the drsamonline repositories into training pairs.** One file, Python 3.8+, standard library
only, no network, no model, no dependency.

```bash
python3 lm-corpus-harvest.py --root ../src --out ../corpus
```

```
rep       359432 pairs  train 352287 / eval 7145  from 1 sources
materia     1709 pairs  train   1671 / eval   38  from 1 sources
book        5502 pairs  train   5373 / eval  129  from 2 sources
act         2626 pairs  train   2570 / eval   56  from 3 sources
audit         19 pairs  train     17 / eval    2  from 2 sources
forge         98 pairs  train     89 / eval    9  from 1 sources

369386 pairs across 6 models -> ../corpus
```

## What it does

Walks the checkouts you point it at, reads the data files that already exist in them, and writes one
JSONL file per model plus a manifest. It does not invent data and it does not paraphrase: every pair
is a real record from a real repository, carrying the path it came from.

Every pair has the same shape:

```json
{"input": "LUMP in abdomen, sensation of",
 "output": {"rubric_id": "kent-repertory-1897-r22839", "book": "kent-repertory-1897",
            "section": "Abdomen", "remedies": [{"abbrev": "Ant-t", "grade": null}]},
 "source": "homoeopathic-materia-medica-repertory-json/repertory/kent-1897/abdomen.json",
 "meta": {"task": "rubric_lookup", "parent": null}}
```

## The six models it feeds

| model | input | output | source |
|---|---|---|---|
| `rep` | rubric text | rubric id, book, section, graded remedies | 171 repertory volumes |
| `materia` | a proving paragraph | remedy, abbreviation, book, year | 15 materia medica volumes |
| `book` | a passage of prose | book, author, section number and title | 41 treatises + the 79-book shelf |
| `act` | a description | tool, or API, with its constraints | TOOLWORKS, app-endpoints, pico |
| `audit` | a lesson or an issue | occurrences, guard, severity, status | forge/LEARNINGS.md, knowledge |
| `forge` | a capability, or an app pitch | spec keys, or a KILN spec | KILN catalogue and specs |

## Three things it refuses to do

**It will not train on a book that is not yet public domain in the US.** A volume whose
`copyright_jurisdictions.US` year is still ahead is skipped, and the skip is recorded in the manifest
with its reason. Today that catches one treatise — Bach's *Free Thyself* (1932), which enters the US
public domain on 1 January 2028. One skip, recorded, rather than one silent inclusion.

**It will not emit OCR noise.** `good_prose()` rejects a chunk shorter than 60 characters, one that
is more than 30% abbreviation tokens, or one that is under 70% letters and spaces. On the treatise
corpus that removes 293 chunks out of 5,795 — the remedy indices and the page-number fragments —
and leaves the prose.

**It will not guess a split.** `eval` is a deterministic 1-in-50 by content hash (1-in-10 for models
under a thousand pairs). The same checkout always produces the same bytes, so the same hash, so a
model can always name the exact dataset it was trained on.

## The manifest

```json
{"schema": "lm-corpus/1",
 "models": {"rep": {"pairs": 359432,
                    "splits": {"train": {"pairs": 352287, "sha256": "f91b4e39cd19..."}},
                    "sources": {"homoeopathic-materia-medica-repertory-json": 359432},
                    "licence_skips": 0, "skip_reasons": []}}}
```

The hash covers the JSONL bytes only — not the date, not the path — so two machines harvesting the
same commits agree. That is the value the model card records and the value tessera keeps.

## What it found that is worth knowing

The repertory corpus is far larger than the rest: 359,432 pairs against 5,502 for books and 1,709 for
materia medica. Any plan that trains all six models equally is wrong; `rep` is the one with data
enough to pretrain on, and the others are fine-tuning sets.

`audit` yields 19 pairs, not 193. The audit repository holds 193 narrative reports; the only
*labelled* lessons in the fleet are forge's 18 entries in LEARNINGS.md, each carrying an occurrence
count and a guard, plus three issues in `knowledge`. That is the honest size of lm-audit's corpus
today, and it is why that model waits.

`forge` yields 98, not the 154 specs I quoted earlier, because a capability entry with an empty note
is not a training pair. The 154 specs are the target language; 98 of them carry a pitch that can be
learned from.

## Running it in CI

One job per data-bearing repository, on push to a path that matters:

```yaml
- name: harvest
  run: python3 lm-corpus-harvest.py --root . --out out --models rep
- name: record
  run: cat out/manifest.json >> $GITHUB_STEP_SUMMARY
```

Then a weekly job that harvests everything, compares the `sha256` per model against the last
published value, and starts a delta fine-tune only when the hash has moved.
