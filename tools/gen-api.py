#!/usr/bin/env python3
"""
gen-api — write docs/API.md from the source, so the reference cannot describe a flag that is gone.

    python3 tools/gen-api.py            # write docs/API.md
    python3 tools/gen-api.py --check    # exit 1 if the committed file is out of date

It reads the tools as text: no import, no execution, nothing that can differ between machines.
The module docstring supplies the summary, `def` lines supply the function table, and
`add_argument` calls supply the options. CI runs --check, so a flag added without regenerating
the reference fails the build rather than quietly rotting.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "docs", "API.md")

TOOLS = [
    ("lm-corpus-harvest.py", "Harvest the repositories into training pairs."),
    ("lm-bpe.py", "Train and apply a byte-level BPE tokenizer."),
    ("lm-train.py", "Train a small transformer on a harvested corpus."),
    ("lm-eval.py", "Score a checkpoint on its held-out split."),
    ("lm-export-c.py", "Export a checkpoint to the .lmk the C kernel loads."),
    ("corpus-drift.py", "Report whether the corpus has moved since the published manifest."),
    ("lm-rebuild.py", "Retrain a drifted model and publish it only if it passes its gate."),
    ("gen-api.py", "Write this page from the source."),
]

FUNC = re.compile(r"^def ([a-zA-Z_][\w]*)\(([^)]*)\):", re.M)
ARG = re.compile(r'add_argument\(\s*"(--?[\w-]+)"(.*?)\)\s*$', re.M | re.S)
KIND = re.compile(r"action=\"(\w+)\"")
HELP = re.compile(r'help="([^"]*)"')
TYPE = re.compile(r"type=([\w.]+)")
DEFAULT = re.compile(r"default=([^,)]+)")


def summary(src):
    """First paragraph of the module docstring, on one line."""
    m = re.search(r'"""(.+?)"""', src, re.S)
    if not m:
        return ""
    body = m.group(1).strip().split("\n\n")[0]
    return " ".join(line.strip() for line in body.splitlines() if line.strip())


def functions(src):
    out = []
    for m in FUNC.finditer(src):
        name, args = m.group(1), " ".join(m.group(2).split())
        if name.startswith("_"):
            continue
        tail = src[m.end():m.end() + 400]
        doc = re.search(r'^\s*"""(.+?)"""', tail, re.S)
        first = ""
        if doc:
            first = doc.group(1).strip().split("\n")[0].strip()
        out.append((name, args, first))
    return out


def options(src):
    out = []
    for m in ARG.finditer(src):
        flag, rest = m.group(1), m.group(2)
        help_ = HELP.search(rest)
        typ = TYPE.search(rest)
        default = DEFAULT.search(rest)
        kind = KIND.search(rest)
        bits = []
        if typ:
            bits.append("`%s`" % typ.group(1))
        elif kind:
            bits.append(kind.group(1))
        if "required=True" in rest:
            bits.append("required")
        if default:
            bits.append("default `%s`" % default.group(1).strip())
        out.append((flag, " ".join(bits) or "flag",
                    help_.group(1) if help_ else ""))
    return out


def build():
    lines = [
        "# API reference",
        "",
        "**Generated from the source by `tools/gen-api.py`.** Every option and function below is",
        "read out of the tool file, so this page cannot describe a flag that does not exist. CI",
        "runs `gen-api.py --check` and fails when the two disagree.",
        "",
        "| tool | what it does |",
        "|---|---|",
    ]
    bodies = []
    for name, purpose in TOOLS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        lines.append("| [`%s`](../tools/%s) | %s |" % (name, name, summary(src) or purpose))

        body = ["", "---", "", "## `%s`" % name, "", summary(src) or purpose, ""]
        opts = options(src)
        if opts:
            body += ["| option | takes | what it does |", "|---|---|---|"]
            for flag, takes, help_ in opts:
                body.append("| `%s` | %s | %s |" % (flag, takes, help_))
            body.append("")
        fns = functions(src)
        if fns:
            body += ["| function | arguments | returns |", "|---|---|---|"]
            for fn, args, doc in fns:
                body.append("| `%s` | `%s` | %s |" % (fn, args or "", doc))
            body.append("")
        bodies.append(body)

    lines += ["", "Usage, worked examples and the measured numbers are in the [README](../README.md).", ""]
    for body in bodies:
        lines += body
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail when docs/API.md is stale")
    a = ap.parse_args()
    new = build()
    if a.check:
        if not os.path.exists(DOC):
            print("docs/API.md is missing — run tools/gen-api.py")
            return 1
        if open(DOC, encoding="utf-8").read() != new:
            print("docs/API.md is stale — run tools/gen-api.py")
            return 1
        print("docs/API.md is current")
        return 0
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("wrote docs/API.md (%d lines)" % new.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
