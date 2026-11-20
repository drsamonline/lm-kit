# Security

## Reporting

Email sohilmomin2000@gmail.com. Please do not open a public issue for a vulnerability.

## What this pipeline touches

- **Reads** files from a directory you name (`--root`). It does not follow symlinks out of it.
- **Writes** JSONL, weights and manifests to a directory you name (`--out`).
- **Never** opens a socket. There is no download step, no model fetch and no telemetry.

## Deliberate limits

| Limit | Why |
|---|---|
| No network at any stage | the corpus must be reproducible from the checkouts alone |
| No execution of repository code | the harvester reads data files, it does not run them |
| No credentials | nothing here needs a token |
| Licence gate on every volume | a book not yet public domain in the US is skipped and recorded |

## If you run the trainer

`lm-train.py` loads JSONL you generated. Treat a corpus from an untrusted source the way you would
treat any other untrusted input: it is data, not code, but it will be memorised if you train on it.
