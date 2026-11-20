#!/usr/bin/env python3
"""
lm-corpus-harvest — turn the drsamonline repositories into training pairs.

One file. Python 3.8+ standard library only. No network, no model, no dependency.
Deterministic: the same checkouts always produce the same bytes and the same hash.

    python3 lm-corpus-harvest.py --root ../src --out ../corpus

What it does
------------
Walks the repositories you point it at, reads the data files that already exist in
them, and writes one JSONL file per model plus a manifest. It does not invent data
and it does not paraphrase: every pair is a real record from a real repository,
carrying the path it came from.

    corpus/
      rep.train.jsonl      rep.eval.jsonl
      materia.train.jsonl  materia.eval.jsonl
      book.*  act.*  audit.*  forge.*
      manifest.json        <- counts, sources, hashes, licence skips

The pair shape is the same for every model:

    {"input": "...", "output": {...}, "source": "repo/path", "meta": {...}}

The licence gate
----------------
A book whose `copyright_jurisdictions.US` year is still in the future is skipped,
and the skip is recorded in the manifest with its reason. That is not caution for
its own sake: one treatise in the library (Bach, 1932) enters the US public domain
on 1 January 2028, and a training set built today must not contain it.

Exit codes
----------
0  everything harvested
1  a requested model had no source (usually a missing checkout)
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date

SCHEMA = "lm-corpus/1"
THIS_YEAR = date.today().year

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def stable_key(pair):
    """Order pairs by content, never by filesystem order."""
    return sha256_bytes((pair["input"] + json.dumps(pair["output"], sort_keys=True)).encode())


def is_eval(pair, every=50):
    """A deterministic 1-in-N eval split, chosen by content hash."""
    return int(stable_key(pair)[:8], 16) % every == 0


def clean(text, limit=None):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if limit and len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "\u2026"
    return t


def chunks(text, size=900, min_len=120):
    """Split prose on paragraph boundaries into ~size-character chunks."""
    parts = re.split(r"(?<=[.!?])\s{2,}|\n{2,}", text or "")
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not buf:
            buf = p
        elif len(buf) + len(p) < size:
            buf += " " + p
        else:
            if len(buf) >= min_len:
                out.append(buf)
            buf = p
    if len(buf) >= min_len:
        out.append(buf)
    return out



def abbrev_density(text):
    """Fraction of tokens that look like a remedy abbreviation (Bell., nux-v., ARS.)."""
    toks = re.findall(r"\S+", text or "")
    if len(toks) < 6:
        return 0.0
    ABBREV = re.compile(r"^(?:[A-Z][A-Za-z\-]{0,6}\.|[A-Z]{2,6}\.?|[a-z]{2,4}\.)$")
    hits = sum(1 for t in toks if ABBREV.match(t))
    return hits / len(toks)


def good_prose(text, min_len=60, max_abbrev=0.30):
    """Reject OCR noise, remedy indices and fragments."""
    t = (text or "").strip()
    if len(t) < min_len:
        return False
    if abbrev_density(t) > max_abbrev:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in t)
    if letters / max(len(t), 1) < 0.70:
        return False
    return True

def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def pd_ok(book):
    """True when the volume is public domain in the US today."""
    jur = (book or {}).get("copyright_jurisdictions") or {}
    us = jur.get("US")
    if isinstance(us, int) and us > THIS_YEAR:
        return False, "US public domain from %d" % us
    return True, ""


# --------------------------------------------------------------------------
# extractors — each yields (pair, None) or (None, skip_reason)
# --------------------------------------------------------------------------

def x_rep(root, skip):
    """Repertory: a rubric's text in, its identifier and graded remedies out."""
    base = os.path.join(root, "homoeopathic-materia-medica-repertory-json")
    if not os.path.isdir(base):
        return
    for dirpath, _, files in os.walk(os.path.join(base, "repertory")):
        for name in sorted(files):
            if not name.endswith(".json") or name == "index.json":
                continue
            path = os.path.join(dirpath, name)
            doc = load_json(path)
            if not doc or doc.get("type") != "repertory":
                continue
            book = doc.get("book", {})
            ok, why = pd_ok(book)
            if not ok:
                skip.append((os.path.relpath(path, root), why))
                continue
            bid = book.get("id", name[:-5])
            secs = doc.get("sections")
            if not secs and isinstance(doc.get("section"), dict):
                secs = [doc["section"]]          # split volumes carry one section per file
            for section in (secs or []):
                stitle = clean(section.get("title", ""), 80)
                for rub in section.get("rubrics", []):
                    text = clean(rub.get("text", ""))
                    if len(text) < 4:
                        continue
                    remedies = sorted(
                        (r for r in rub.get("remedies", []) if r.get("abbrev")),
                        key=lambda r: (-(r.get("grade") or 0), r.get("abbrev") or ""),
                    )[:6]
                    yield {
                        "input": text,
                        "output": {
                            "rubric_id": rub.get("id"),
                            "book": bid,
                            "section": stitle,
                            "remedies": [
                                {"abbrev": r.get("abbrev"), "grade": r.get("grade")}
                                for r in remedies
                            ],
                        },
                        "source": os.path.relpath(path, root),
                        "meta": {"task": "rubric_lookup", "parent": rub.get("parent")},
                    }


def x_materia(root, skip):
    """Materia medica: a proving paragraph in, the remedy it belongs to out."""
    base = os.path.join(root, "homoeopathic-materia-medica-repertory-json")
    if not os.path.isdir(base):
        return
    for name in sorted(os.listdir(os.path.join(base, "materia_medica"))):
        if not name.endswith(".json"):
            continue
        path = os.path.join(base, "materia_medica", name)
        doc = load_json(path)
        if not doc or doc.get("type") != "materia_medica":
            continue
        book = doc.get("book", {})
        ok, why = pd_ok(book)
        if not ok:
            skip.append((os.path.relpath(path, root), why))
            continue
        for rem in doc.get("remedies", []):
            sections = rem.get("sections") or {}
            if isinstance(sections, dict):
                body = "\n\n".join(v for v in sections.values() if isinstance(v, str))
            else:
                body = str(sections)
            for piece in chunks(body):
                if not good_prose(piece):
                    continue
                yield {
                    "input": clean(piece),
                    "output": {
                        "remedy": rem.get("name") or rem.get("canonical_name"),
                        "abbrev": rem.get("abbrev"),
                        "book": book.get("id"),
                        "year": book.get("year"),
                    },
                    "source": os.path.relpath(path, root),
                    "meta": {"task": "attribute_proving",
                             "kingdom": rem.get("kingdom"),
                             "clinical": (rem.get("clinical_indications") or [])[:5]},
                }


def x_book(root, skip):
    """Treatises and the public-domain shelf: prose in, structure out."""
    base = os.path.join(root, "homoeopathic-materia-medica-repertory-json")
    if os.path.isdir(base):
        for name in sorted(os.listdir(os.path.join(base, "treatise"))):
            if not name.endswith(".json"):
                continue
            path = os.path.join(base, "treatise", name)
            doc = load_json(path)
            if not doc or doc.get("type") != "treatise":
                continue
            book = doc.get("book", {})
            ok, why = pd_ok(book)
            if not ok:
                skip.append((os.path.relpath(path, root), why))
                continue
            for sec in doc.get("sections", []):
                for piece in chunks(sec.get("text", "")):
                    if not good_prose(piece):
                        continue
                    yield {
                        "input": clean(piece),
                        "output": {
                            "book": book.get("id"),
                            "title": clean(book.get("title"), 120),
                            "author": clean(book.get("author"), 80),
                            "section": sec.get("n"),
                            "section_title": clean(sec.get("title"), 120),
                        },
                        "source": os.path.relpath(path, root),
                        "meta": {"task": "place_in_book"},
                    }
    # the public-domain shelf catalogue
    cat = os.path.join(root, "corpus-extendum", "data", "catalog.json")
    doc = load_json(cat)
    if isinstance(doc, list):
        for b in doc:
            if not isinstance(b, dict):
                continue
            blurb = clean(b.get("blurb", ""))
            if len(blurb) < 40:
                continue
            yield {
                "input": blurb,
                "output": {
                    "title": b.get("title"),
                    "author": (b.get("author") or {}).get("name"),
                    "year": b.get("year"),
                    "category": b.get("category"),
                    "slug": b.get("slug"),
                },
                "source": "corpus-extendum/data/catalog.json",
                "meta": {"task": "identify_book"},
            }


def x_act(root, skip):
    """Automation: a request in, one command or one tool out."""
    # pico rules: the regex is the canonical command template
    pico = os.path.join(root, "pico")
    if os.path.isdir(pico):
        for dirpath, _, files in os.walk(os.path.join(pico, "packs")):
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                for m in re.finditer(r'\(\s*r?"([^"]+)"\s*,\s*([A-Za-z_][\w]*)\s*\)', text):
                    pattern, handler = m.group(1), m.group(2)
                    template = re.sub(r"\(\?P<(\w+)>[^)]*\)", r"{\1}", pattern)
                    template = template.lstrip("^").rstrip("$").replace("\\", "")
                    if "{" not in template:
                        continue
                    yield {
                        "input": "command: " + template.replace("{", "<").replace("}", ">"),
                        "output": {"rule": pattern, "handler": handler,
                                   "pack": os.path.basename(name)[:-3],
                                   "command": template},
                        "source": os.path.relpath(path, root),
                        "meta": {"task": "rule_match"},
                    }
    # TOOLWORKS: 969 tools, each with a description and a danger flag
    tw = os.path.join(root, "TOOLWORKS", "data", "catalogue.json")
    doc = load_json(tw)
    if isinstance(doc, dict):
        for t in doc.get("tools", []):
            desc = clean(t.get("desc", ""))
            if len(desc) < 12:
                continue
            yield {
                "input": desc,
                "output": {"tool": t.get("name"), "category": t.get("category"),
                           "group": t.get("group"), "source": t.get("source"),
                           "dangerous": bool(t.get("dangerous"))},
                "source": "TOOLWORKS/data/catalogue.json",
                "meta": {"task": "route_to_tool"},
            }
    # endpoints: 2,214 APIs with their constraints
    ep = os.path.join(root, "app-endpoints", "data", "apis.json")
    doc = load_json(ep)
    if isinstance(doc, dict):
        for a in doc.get("apis", []):
            desc = clean(a.get("desc", ""))
            if len(desc) < 12:
                continue
            yield {
                "input": desc,
                "output": {"api": a.get("name"), "host": a.get("host"),
                           "auth": a.get("auth"), "cors": a.get("cors"),
                           "https": a.get("https"), "cat": a.get("cat")},
                "source": "app-endpoints/data/apis.json",
                "meta": {"task": "route_to_api"},
            }


def x_audit(root, skip):
    """Review: a failure in, its cause and the guard that now catches it out."""
    learn = os.path.join(root, "forge", "LEARNINGS.md")
    if os.path.isfile(learn):
        text = open(learn, encoding="utf-8", errors="ignore").read()
        blocks = re.split(r"\n(?=#{2,4}\s)", text)
        for blk in blocks:
            if len(blk) < 160:
                continue
            if not re.search(r"×\s*\d+|\bx\s*\d+\b|Guard", blk):
                continue
            count = re.search(r"×\s*(\d+)", blk)
            guard = re.search(r"\*?\*?Guard\*?\*?\s*[:—-]\s*(.+)", blk)
            body = clean(re.sub(r"[#*]", "", blk), 1200)
            if len(body) < 120:
                continue
            yield {
                "input": body,
                "output": {"occurrences": int(count.group(1)) if count else None,
                           "guard": clean(guard.group(1), 200) if guard else None,
                           "closed": bool(guard)},
                "source": "forge/LEARNINGS.md",
                "meta": {"task": "diagnose_failure"},
            }
    idx = os.path.join(root, "knowledge", "index.json")
    doc = load_json(idx)
    if isinstance(doc, dict):
        for it in doc.get("issues", []):
            title = clean(it.get("title", ""))
            if not title:
                continue
            yield {
                "input": title,
                "output": {"repo": it.get("repo"), "severity": it.get("severity"),
                           "status": it.get("status"), "tags": it.get("tags", []),
                           "path": it.get("path")},
                "source": "knowledge/index.json",
                "meta": {"task": "classify_issue"},
            }


def x_forge(root, skip):
    """App generation: a capability in, its contract out — and specs as targets."""
    cat = os.path.join(root, "KILN", "catalogue")
    if os.path.isdir(cat):
        for dirpath, _, files in os.walk(cat):
            for name in sorted(files):
                if not name.endswith(".json") or name in ("capabilities.json", "categories.json", "kits.json"):
                    continue
                path = os.path.join(dirpath, name)
                d = load_json(path)
                if not isinstance(d, dict) or "id" not in d:
                    continue
                note = clean(d.get("note", ""))
                if len(note) < 20:
                    continue
                yield {
                    "input": "%s — %s" % (d.get("name", ""), note),
                    "output": {"id": d.get("id"), "family": d.get("family"),
                               "spec_keys": d.get("spec_keys", []),
                               "status": d.get("status")},
                    "source": os.path.relpath(path, root),
                    "meta": {"task": "capability_contract"},
                }
    # forge specs: the machine block is the target
    spec_dir = os.path.join(root, "KILN", "specs")
    if os.path.isdir(spec_dir):
        for dirpath, _, files in os.walk(spec_dir):
            for name in sorted(files):
                if not name.endswith(".md"):
                    continue
                path = os.path.join(dirpath, name)
                text = open(path, encoding="utf-8", errors="ignore").read()
                m = re.search(r"```json\s+kiln\s*(\{.*?\})\s*```", text, re.S)
                if not m:
                    continue
                try:
                    spec = json.loads(m.group(1))
                except ValueError:
                    continue
                pitch = clean(spec.get("pitch", ""))
                if not pitch:
                    continue
                yield {
                    "input": pitch,
                    "output": {k: spec.get(k) for k in
                               ("id", "name", "category", "features", "entities", "storage", "not")},
                    "source": os.path.relpath(path, root),
                    "meta": {"task": "idea_to_spec", "tier": spec.get("tier")},
                }


MODELS = {
    "rep": x_rep,
    "materia": x_materia,
    "book": x_book,
    "act": x_act,
    "audit": x_audit,
    "forge": x_forge,
}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Harvest the fleet into training pairs.")
    ap.add_argument("--root", required=True, help="directory holding the repo checkouts")
    ap.add_argument("--out", required=True, help="directory to write into")
    ap.add_argument("--models", default="all", help="comma-separated, or all")
    ap.add_argument("--limit", type=int, default=0, help="cap pairs per model (0 = no cap)")
    args = ap.parse_args()

    wanted = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    os.makedirs(args.out, exist_ok=True)
    manifest = {"schema": SCHEMA, "generated": date.today().isoformat(),
                "root": os.path.basename(os.path.abspath(args.root)), "models": {}}
    missing = []

    for model in wanted:
        fn = MODELS.get(model)
        if not fn:
            print("!! unknown model %r" % model, file=sys.stderr)
            continue
        skip, pairs, sources = [], [], {}
        for pair in fn(args.root, skip):
            pairs.append(pair)
            src = pair["source"].split("/")[0]
            sources[src] = sources.get(src, 0) + 1
            if args.limit and len(pairs) >= args.limit:
                break
        if not pairs:
            missing.append(model)
            print("!! %s: no source found under %s" % (model, args.root), file=sys.stderr)
            continue

        pairs.sort(key=stable_key)
        every = 50 if len(pairs) >= 1000 else 10
        train = [p for p in pairs if not is_eval(p, every)]
        ev = [p for p in pairs if is_eval(p, every)]
        digests = {}
        for split, rows in (("train", train), ("eval", ev)):
            path = os.path.join(args.out, "%s.%s.jsonl" % (model, split))
            blob = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(blob)
            digests[split] = {"pairs": len(rows), "bytes": len(blob.encode()),
                              "sha256": sha256_bytes(blob.encode())}
        manifest["models"][model] = {
            "pairs": len(pairs),
            "splits": digests,
            "sources": dict(sorted(sources.items())),
            "licence_skips": len(skip),
            "skip_reasons": sorted({s[1] for s in skip}),
            "skip_examples": [s[0] for s in skip[:5]],
        }
        print("%-8s %7d pairs  train %d / eval %d  from %d sources"
              % (model, len(pairs), len(train), len(ev), len(sources)))

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    total = sum(m["pairs"] for m in manifest["models"].values())
    print("\n%d pairs across %d models -> %s" % (total, len(manifest["models"]), args.out))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
