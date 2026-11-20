# Disclaimer

Version: 1.0
Effective: 20 November 2026
Owner: Dr Sohil Momin — sohilmomin2000@gmail.com
Changes: Initial version.

## Read this before you use anything this repository produces

This repository is a **toolchain**. It harvests training pairs from source repositories, trains
small language models, and exports them. It is not a product, it is not a medical device, and it
makes no clinical claim of any kind.

## What the models are

A model built with this kit is a **lookup and formatting function**. It takes a sentence and returns
a JSON object that matches a schema the calling application already defines — a rubric identifier, a
tool name, a category, a command. It answers nothing else. It has no knowledge of medicine, no
memory of any patient, and no ability to reason about a case.

The models are trained on **public-domain historical texts** and on the published catalogues of this
account's own software. They are not trained on patient records, and the pipeline has no path by
which patient data could enter it.

## What they must never be used for

| Not this | Why |
|---|---|
| Diagnosis, prognosis, or case analysis | a rubric identifier is a pointer into a book, not a clinical judgement |
| Prescribing, potency, dose, or repetition | outside the scope of the software entirely |
| Replacing a qualified practitioner | the models refuse; the application must not override that |
| Advertising or promoting a remedy for a listed condition | the Drugs and Magic Remedies (Objectionable Advertisements) Act, 1954 restricts such advertisements in India |
| Any decision about a named individual | the models take no personal data and produce no personal advice |

## On homoeopathy specifically

Several source corpora are classical homoeopathic texts — repertories, materia medicas and
treatises — that are in the public domain in India and, where recorded, in the United States. Their
presence here is **bibliographic**, in the same sense as a library holding a scanned 1897 volume.
Inclusion is not an endorsement, and the software makes no claim about the therapeutic value of any
substance named in those texts.

Every volume carries its public-domain basis in the source catalogue, and a volume whose US
public-domain year is still in the future is excluded from training by the harvester and recorded in
the manifest.

## On AI output

Text and JSON produced by a model from this kit may be **wrong**. It is generated, not retrieved.
The design mitigates this in three ways, and none of them makes the output authoritative:

- output is constrained to a schema, so it is well-formed;
- identifiers are resolved against a local index, so an invented one is discarded;
- every result carries a confidence score, and below a threshold the kit returns candidates rather
  than an answer.

**Do not rely on generated output for anything consequential without a human checking it.**

## Liability

The software is provided as-is under the PolyForm Noncommercial License 1.0.0, which disclaims
warranty and limits liability. Nothing in this document creates a duty of care towards any user of
any application built on it.

## What a lawyer should confirm before this ships in a clinical product

- whether the intended use crosses into a regulated medical device under the Medical Devices Rules,
  2017, as amended;
- whether any application screen would constitute an advertisement under the 1954 Act;
- whether the Digital Personal Data Protection Act, 2023 and the rules made under it impose
  obligations once an application stores case notes, even on the user's own device;
- whether the public-domain basis recorded for each volume is sufficient for the intended
  jurisdiction of distribution.

This document is a review-ready draft, not legal advice.
