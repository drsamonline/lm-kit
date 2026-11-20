# Privacy

Version: 1.0
Effective: 20 November 2026
Owner: Dr Sohil Momin — sohilmomin2000@gmail.com
Changes: Initial version.

## What this is

A pipeline that turns the repositories of this account into training pairs, and a trainer that
produces small language models. It runs on your own machine.

## What it collects

Nothing. There is no account, no server, no telemetry and no network call at any stage. The
harvester reads files from a directory you name and writes JSONL to a directory you name. The
trainer reads that JSONL and writes weights.

## What ends up in the models

Only text that was already in the repositories, plus the public-domain books they carry. A volume
whose US public-domain year is still in the future is skipped and the skip is recorded in the
manifest with its reason.

## What the models must not do

The models answer with schema-valid JSON and refuse everything else. They hold no personal data,
make no clinical claim, and are never given the network.
