#!/usr/bin/env python3
"""
lm-bpe — a byte-level BPE trainer and codec, standard library only.

    python3 lm-bpe.py train --corpus ../corpus --models act,book,rep --vocab 4096 --out bpe.json
    python3 lm-bpe.py stats --bpe bpe.json --corpus ../corpus --model act
    python3 lm-bpe.py encode --bpe bpe.json --text "Movie and TV data"

Why this exists
---------------
The training harness shipped with a byte tokenizer: 256 symbols, no training step,
no dependency, and deliberately the weakest part of the pipeline. A byte-level model
spends its capacity learning to spell, and on this corpus — JSON keys, rubric phrases,
remedy abbreviations — the same strings recur constantly. BPE turns those repetitions
into single tokens.

The algorithm is the ordinary one: start from bytes, repeatedly merge the most
frequent adjacent pair, weight every count by the frequency of the word it appears
in. It is incremental — after a merge only the words that contained that pair are
recounted — which is what makes 4,096 merges finish in seconds instead of hours.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

WORD = re.compile(r"\S+")
BASE = 256


def read_texts(corpus, models, limit=0):
    out = []
    for m in models:
        for split in ("train", "eval"):
            p = os.path.join(corpus, "%s.%s.jsonl" % (m, split))
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    if limit and n >= limit:
                        break
                    try:
                        pair = json.loads(line)
                    except ValueError:
                        continue
                    out.append(str(pair.get("input", "")))
                    out.append(json.dumps(pair.get("output", {}), sort_keys=True, ensure_ascii=False))
    return out


def train(texts, vocab_size, min_count=2, verbose=True):
    if not texts:
        # an empty corpus produces a 256-token vocabulary that trains nothing and looks like
        # success. Fail instead of writing it.
        raise SystemExit("no text to train on — check --corpus and --models")
    freq = Counter()
    for t in texts:
        for w in WORD.findall(t):
            freq[w] += 1

    words = {w: list(w.encode("utf-8")) for w in freq}
    weight = dict(freq)
    merges = []

    pair_count = Counter()
    pair_words = defaultdict(set)
    for w, syms in words.items():
        c = weight[w]
        for i in range(len(syms) - 1):
            p = (syms[i], syms[i + 1])
            pair_count[p] += c
            pair_words[p].add(w)

    target = vocab_size - BASE
    while len(merges) < target:
        if not pair_count:
            break
        best, n = pair_count.most_common(1)[0]
        if n < min_count:
            break
        new_id = BASE + len(merges)
        merges.append([best[0], best[1]])

        for w in list(pair_words[best]):
            syms = words[w]
            c = weight[w]
            # remove this word's old contributions
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_count[p] -= c
                if pair_count[p] <= 0:
                    pair_count.pop(p, None)
                s = pair_words.get(p)
                if s is not None:
                    s.discard(w)
                    if not s:
                        pair_words.pop(p, None)
            # merge
            merged, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            words[w] = merged
            # add the new contributions
            for i in range(len(merged) - 1):
                p = (merged[i], merged[i + 1])
                pair_count[p] += c
                pair_words[p].add(w)

        if verbose and (len(merges) % 512 == 0 or len(merges) == target):
            print("  merge %5d  pair %s  count %d" % (len(merges), best, n))

    return {"version": 1, "base": BASE, "vocab_size": BASE + len(merges), "merges": merges}


class BPE:
    def __init__(self, spec):
        self.merges = [tuple(m) for m in spec["merges"]]
        self.rank = {p: i for i, p in enumerate(self.merges)}
        self.vocab_size = spec["vocab_size"]
        # resolve every merge to its bytes once, so decode is a concat
        self.expand = {}
        for i, (a, b) in enumerate(self.merges):
            ea = self.expand.get(a, bytes([a]) if a < BASE else b"")
            eb = self.expand.get(b, bytes([b]) if b < BASE else b"")
            self.expand[BASE + i] = ea + eb

    def encode(self, text):
        out = []
        for w in WORD.findall(text):
            syms = list(w.encode("utf-8"))
            while len(syms) > 1:
                best, best_i = None, -1
                for i in range(len(syms) - 1):
                    r = self.rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best):
                        best, best_i = r, i
                if best is None:
                    break
                syms[best_i:best_i + 2] = [BASE + best]
            out.extend(syms)
            out.append(32)  # keep a space between words
        return out[:-1] if out else out

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if i < BASE:
                buf.append(i)
            elif i in self.expand:
                buf.extend(self.expand[i])
        return buf.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "stats", "encode"])
    ap.add_argument("--corpus")
    ap.add_argument("--models", default="act")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bpe")
    ap.add_argument("--text")
    a = ap.parse_args()

    if a.cmd == "train":
        models = a.models.split(",")
        texts = read_texts(a.corpus, models, a.limit)
        chars = sum(len(t) for t in texts)
        print("training on %d strings, %d characters" % (len(texts), chars))
        spec = train(texts, a.vocab)
        json.dump(spec, open(a.bpe, "w"))
        print("vocab %d  merges %d  -> %s (%.1f KB)"
              % (spec["vocab_size"], len(spec["merges"]), a.bpe,
                 os.path.getsize(a.bpe) / 1024))

    elif a.cmd == "stats":
        spec = json.load(open(a.bpe))
        tok = BPE(spec)
        texts = read_texts(a.corpus, a.models.split(","))
        chars = sum(len(t) for t in texts)
        ids = sum(len(tok.encode(t)) for t in texts)
        print("byte tokens %d   bpe tokens %d   chars %d" % (chars, ids, chars))
        print("compression %.2f chars/token (byte-level is 1.00)" % (chars / max(ids, 1)))
        print("sequence length falls %.1fx" % (chars / max(ids, 1)))

    else:
        spec = json.load(open(a.bpe))
        tok = BPE(spec)
        ids = tok.encode(a.text)
        print("tokens:", ids)
        back = tok.decode(ids)
        norm = lambda t: " ".join(t.split())
        print("round trip:", norm(back) == norm(a.text), repr(back))


if __name__ == "__main__":
    main()
