# Notice — third-party content and attribution

Version: 1.0
Effective: 20 November 2026

This repository contains no third-party source code. It does contain **text and data** that came
from elsewhere, and this file records each one, its terms, and where it is used.

## Bundled or harvested text

| Source | Terms | Used for | Note |
|---|---|---|---|
| CMU Pronouncing Dictionary (cmusphinx/cmudict) | BSD-2-Clause | the `verse` extractor — word to ARPAbet pronunciation | redistributed inside `app-poets-studio` as a 124,926-word subset; this repository reads it, it does not vendor it |
| Classical homoeopathic repertories, materia medicas and treatises, 1840–1906 | public domain | the `rep`, `materia` and `book` extractors | each volume's public-domain basis is recorded in the source catalogue, including its US status |
| Public-domain shelf of 79 books | public domain in India and the US | the `book` extractor | title, author and licence basis per record |
| NTET question bank and glossary | this account's own work | the `tutor` extractor | 280 questions with rationales, 112 glossary terms |
| Tool and API catalogues | this account's own work | the `act` and `file` extractors | 969 tools, 2,214 APIs |
| KILN capability catalogue and specifications | this account's own work | the `forge` extractor | 49 capabilities, 37 specs |
| forge `LEARNINGS.md` | this account's own work | the `audit` extractor | 18 lessons, each with an occurrence count and a guard |

## What this means in practice

- **Nothing is vendored.** The harvester reads from checkouts you point it at; the repository itself
  contains no third-party text beyond this notice.
- **A model trained on this corpus inherits its provenance.** If you distribute weights, carry this
  file with them. The model card records the corpus hash so the exact inputs can be recovered.
- **The BSD-2-Clause notice** requires the copyright notice and disclaimer to travel with any
  redistribution of the dictionary data. The dictionary's own header in
  `app-poets-studio/docs/play/dict.js` carries it; do not strip it.

## Attribution in applications

An application that ships a model trained on the classical texts should say so in its own about
screen, naming the source library, and should not present the text as its own composition.

## Correction

If anything above is wrong, or a source is missing, email sohilmomin2000@gmail.com and it will be
corrected in the next commit.
