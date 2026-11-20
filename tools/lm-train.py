#!/usr/bin/env python3
"""
lm-train — train one small model from a harvested corpus.

Reads the JSONL that lm-corpus-harvest.py wrote and trains a decoder-only
transformer on it. CPU is enough to prove the pipeline; a GPU is needed to
train anything worth shipping.

    python3 lm-train.py --corpus ../corpus --model act --out ../run
    python3 lm-train.py --corpus ../corpus --model act --out ../run --steps 400
    python3 lm-train.py --run ../run --generate "Movie and TV data"

What it is
----------
A 2020s-shaped transformer at toy scale: RMSNorm, SwiGLU, rotary position
embeddings, tied input/output embeddings, causal attention. Nothing exotic,
because the point is the pipeline, not the architecture.

The loss is masked to the answer
--------------------------------
Every pair is rendered as `IN: <input>` then `OUT: <json>`; the loss is computed
only on the OUT half. That is what makes it an instruction model rather than a
text continuer, and it costs nothing to do properly.

The tokenizer seam
------------------
Byte-level here: 256 symbols plus three specials, no training step, no
dependency. That is deliberately the weakest part — production swaps in the
10,240-token BPE over English, Hindi and Gujarati described in the proposal.
`Tokenizer` is the only class that has to change.

Export
------
`model.pt` (torch state dict), `model.npz` (int8 weights plus per-row scales),
`config.json`, and `card.json` — the card records the corpus hash, the exact
configuration, the eval loss and the parameter count, because a model that
cannot name its data cannot be audited.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import re
from array import array
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

PAD, BOS, EOS = 256, 257, 258
VOCAB = 264  # 259 real symbols, padded to a multiple of 8


# ---------------------------------------------------------------- tokenizer
class Tokenizer:
    """Byte-level. The production build replaces this class and nothing else."""

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

    def vocab_size(self):
        return VOCAB


def make_tokenizer(args):
    """bytes (default) or a trained BPE. This is the seam the proposal calls out."""
    if getattr(args, "tokenizer", "bytes") == "bpe":
        if not args.bpe or not os.path.exists(args.bpe):
            sys.exit("--tokenizer bpe needs --bpe <file>")
        spec = json.load(open(args.bpe))

        class BPETok:
            def __init__(self, spec):
                self.merges = [tuple(m) for m in spec["merges"]]
                self.rank = {p: i for i, p in enumerate(self.merges)}
                self.vs = spec["vocab_size"]
                self.expand = {}
                for i, (a, b) in enumerate(self.merges):
                    ea = self.expand.get(a, bytes([a]) if a < 256 else b"")
                    eb = self.expand.get(b, bytes([b]) if b < 256 else b"")
                    self.expand[256 + i] = ea + eb

            def encode(self, text):
                out = []
                for w in re.findall(r"\S+", text):
                    syms = list(w.encode("utf-8"))
                    while len(syms) > 1:
                        best, bi = None, -1
                        for i in range(len(syms) - 1):
                            r = self.rank.get((syms[i], syms[i + 1]))
                            if r is not None and (best is None or r < best):
                                best, bi = r, i
                        if best is None:
                            break
                        syms[bi:bi + 2] = [256 + best]
                    out.extend(syms)
                    out.append(32)
                return out[:-1] if out else out

            def decode(self, ids):
                buf = bytearray()
                for i in ids:
                    if i < 256:
                        buf.append(i)
                    elif i in self.expand:
                        buf.extend(self.expand[i])
                return buf.decode("utf-8", "replace")

            def vocab_size(self):
                return self.vs

        return BPETok(spec)
    return Tokenizer()


# ---------------------------------------------------------------- data
def render(pair):
    """One pair into (prompt_text, answer_text)."""
    out = json.dumps(pair["output"], sort_keys=True, ensure_ascii=False)
    return "IN: %s\nOUT: " % pair["input"], out


def load_split(path, tok, seq, limit=0):
    """Stream a split into two compact arrays, then cut it into blocks.

    The first version built a Python list of every token id. On the repertory corpus
    (134 MB of JSONL) that is tens of millions of ints at 28 bytes each and it dies with
    MemoryError before training starts. Streaming into array('i') is 4 bytes a token, and
    the blocks are sliced out of the array at batch time rather than materialised.

    Returns (ids, mask, nblocks) or (None, None, 0).
    """
    # uint16 is enough for any vocabulary under 65,536 and halves the footprint of a
    # long corpus: the repertory split is ~80 M tokens, which is 320 MB as int32 and
    # 160 MB as uint16. The container here caps at 1.5 GB.
    ids = array("H" if tok.vocab_size() < 65536 else "I")
    mask = array("b")
    started = time.time()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                break
            if n and n % 50000 == 0:
                print("  ... %d pairs, %.0f M tokens, %.0fs" % (n, len(ids) / 1e6, time.time() - started),
                      flush=True)
            try:
                pair = json.loads(line)
            except ValueError:
                continue
            prompt, answer = render(pair)
            p_ids = [BOS] + tok.encode(prompt)
            a_ids = tok.encode(answer) + [EOS]
            ids.extend(p_ids)
            ids.extend(a_ids)
            mask.extend(bytes(len(p_ids)))
            mask.extend(b"\x01" * len(a_ids))
    if len(ids) < seq + 2:
        return None, None, 0
    ids = np.frombuffer(ids, dtype=np.uint16 if ids.typecode == "H" else np.uint32)
    mask = np.frombuffer(mask, dtype=np.int8)
    nblocks = (len(ids) - 1) // seq
    return ids, mask, nblocks


def batch(ids, mask, seq, idx):
    """Cut `idx` blocks out of the stream. X is the block, Y is it shifted by one."""
    x = np.stack([ids[i * seq:i * seq + seq] for i in idx]).astype(np.int64)
    y = np.stack([ids[i * seq + 1:i * seq + seq + 1] for i in idx]).astype(np.int64)
    m = np.stack([mask[i * seq + 1:i * seq + seq + 1] for i in idx]).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


# ---------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq, dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, :x.shape[2], :]
    s = sin[None, None, :x.shape[2], :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = self.w2(F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x)))
        return x + h


class TinyLM(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, ff=512, seq=256):
        super().__init__()
        self.seq = seq
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied
        cos, sin = rope_cache(seq, d // heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- train
def lr_at(step, total, base, warm=50):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def train(args):
    tok = make_tokenizer(args)
    V = tok.vocab_size()
    t0 = time.time()
    ids, mask, ntr = load_split(os.path.join(args.corpus, args.model + ".train.jsonl"),
                                tok, args.seq, args.limit)
    if ids is None:
        sys.exit("no training blocks in %s" % args.corpus)
    eids, emask, nev = load_split(os.path.join(args.corpus, args.model + ".eval.jsonl"), tok, args.seq)
    print("train blocks %d  eval blocks %d  tokens/block %d  loaded in %.1fs (%.0f MB)"
          % (ntr, nev or 0, args.seq, time.time() - t0, ids.nbytes / 1e6))

    model = TinyLM(V, args.dim, args.layers, args.heads, args.ff, args.seq)
    n = model.n_params()
    print("parameters {:,}  ({:.2f} MB at int8)".format(n, n / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        idx = np.random.randint(0, ntr, args.batch)
        xb, yb, mb = batch(ids, mask, args.seq, idx)
        logits = model(xb)
        loss_tok = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1), reduction="none")
        denom = mb.reshape(-1).sum().clamp(min=1)
        loss = (loss_tok * mb.reshape(-1)).sum() / denom
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print("  step %4d  loss %.4f  lr %.2e  %.0fs" %
                  (step, loss.item(), opt.param_groups[0]["lr"], time.time() - t0))

    eval_loss = None
    if eids is not None and nev:
        model.eval()
        with torch.no_grad():
            tot, den = 0.0, 0.0
            for i in range(0, nev, args.batch):
                idx = np.arange(i, min(i + args.batch, nev))
                ex, ey, em = batch(eids, emask, args.seq, idx)
                lg = model(ex)
                l = F.cross_entropy(lg.reshape(-1, V), ey.reshape(-1), reduction="none")
                tot += (l * em.reshape(-1)).sum().item()
                den += em.reshape(-1).sum().item()
            eval_loss = tot / max(den, 1)
        print("  eval loss %.4f  (perplexity %.1f)" % (eval_loss, math.exp(min(eval_loss, 20))))

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    export_int8(model, os.path.join(args.out, "model.npz"))
    cfg = {"vocab": V, "tokenizer": getattr(args, "tokenizer", "bytes"), "dim": args.dim, "layers": args.layers, "heads": args.heads,
           "ff": args.ff, "seq": args.seq, "params": n}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    card = {"model": "lm-" + args.model, "corpus": args.model, "steps": args.steps,
            "batch": args.batch, "lr": args.lr, "params": n,
            "final_train_loss": round(float(np.mean(losses[-20:])), 4),
            "eval_loss": None if eval_loss is None else round(eval_loss, 4),
            "corpus_sha256": corpus_hash(args.corpus, args.model),
            "generated": time.strftime("%Y-%m-%d")}
    json.dump(card, open(os.path.join(args.out, "card.json"), "w"), indent=1)
    print("\nwrote %s  (%.1f MB int8)" % (args.out, os.path.getsize(os.path.join(args.out, "model.npz")) / 1e6))
    return card


def corpus_hash(corpus, model):
    h = hashlib.sha256()
    for split in ("train", "eval"):
        p = os.path.join(corpus, "%s.%s.jsonl" % (model, split))
        if os.path.exists(p):
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]


def export_int8(model, path):
    """Per-output-channel int8 quantisation. The first step of the size ladder."""
    out = {}
    for name, p in model.state_dict().items():
        if p.dtype != torch.float32 or p.dim() < 2:
            out[name.replace(".", "/")] = p.numpy().astype(np.float32)
            continue
        w = p.detach()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        out[name.replace(".", "/")] = q.numpy()
        out[name.replace(".", "/") + ".scale"] = scale.squeeze(1).numpy().astype(np.float32)
    np.savez_compressed(path, **out)


# ---------------------------------------------------------------- sample
def generate(run, prompt, n=60, temp=0.8, topk=20):
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = TinyLM(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["seq"])
    model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu"))
    model.eval()
    tok = make_tokenizer(argparse.Namespace(tokenizer=cfg.get("tokenizer", "bytes"),
                                            bpe=os.path.join(run, "bpe.json")))
    ids = [BOS] + tok.encode("IN: %s\nOUT: " % prompt)
    with torch.no_grad():
        for _ in range(n):
            ctx = torch.tensor([ids[-cfg["seq"]:]])
            logits = model(ctx)[0, -1] / temp
            v, i = torch.topk(logits, topk)
            probs = torch.softmax(v, -1)
            nxt = i[torch.multinomial(probs, 1)].item()
            if nxt == EOS:
                break
            ids.append(nxt)
    return tok.decode(ids[len(tok.encode("IN: %s\nOUT: " % prompt)) + 1:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--model", default="act")
    ap.add_argument("--out")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokenizer", default="bytes", choices=["bytes", "bpe"])
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--run")
    ap.add_argument("--generate", default=None)
    a = ap.parse_args()
    if a.run and a.generate:
        print(generate(a.run, a.generate))
    elif a.corpus and a.out:
        train(a)
    else:
        ap.print_help()
