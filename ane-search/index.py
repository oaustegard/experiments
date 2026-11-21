#!/usr/bin/env python3
"""Index write-ups under ~/Projects: chunk by heading, embed each chunk on the Neural Engine, store them in a remax_kb knowledge base
(index/projects.kbi + projects.kbc/, see kb.py). Re-running only embeds chunks whose text changed."""
import os, re, sys, json, time, hashlib, math, collections, numpy as np, warnings
warnings.filterwarnings("ignore")
from common import *
BUDGET = 215          # body tokens per chunk; the rest of the 256 is the path/heading line and specials

def files():
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.endswith(".mlpackage") and not d.startswith("."))
        for f in sorted(fn):
            if f.lower().endswith(EXTS) and not SKIP_FILES.match(f):
                p = os.path.join(dp, f)
                try: s = os.path.getsize(p)
                except OSError: continue
                if 200 < s < 2_000_000: yield p

def chunks(path, tok):
    """(heading trail, first line number, text) pieces of at most BUDGET tokens, split at headings then blank lines."""
    try: raw = open(path, errors="ignore").read().replace("\r\n", "\n")
    except OSError: return
    if not path.lower().endswith(".md") and not prose(raw): return
    lines = raw.split("\n")
    trail, blocks, cur, start, fence = [], [], [], 1, False
    def flush(n):
        nonlocal cur, start
        if any(l.strip() for l in cur): blocks.append((" > ".join(trail), start, "\n".join(cur).strip()))
        cur, start = [], n
    for n, l in enumerate(lines, 1):
        if l.lstrip().startswith("```"): fence = not fence
        m = None if fence else re.match(r"^(#{1,6})\s+(.*)", l)
        if m:
            flush(n); lvl = len(m.group(1)); trail[:] = trail[:lvl - 1] + [m.group(2).strip()[:80]]
        elif not l.strip() and not fence: cur.append(l); flush(n + 1) if sum(len(x) for x in cur) > 1200 else None
        else: cur.append(l)
    flush(len(lines) + 1)
    for head, ln, text in blocks:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids.ids) <= BUDGET:
            if len(text) >= 40: yield head, ln, text
            continue
        offs, i = ids.offsets, 0
        while i < len(offs):                                       # long block: windows of BUDGET tokens, cut at a line break when one is near
            j = min(i + BUDGET, len(offs)); a, b = offs[i][0], offs[j - 1][1]
            if j < len(offs):
                nl = text.rfind("\n", a + (b - a) // 2, b)
                if nl > 0:
                    b = nl
                    while j > i + 1 and offs[j - 1][1] > b: j -= 1
            yield head, ln + text.count("\n", 0, a), text[a:b].strip()
            i = j

def main():
    import kb, contextlib, io
    from remax_kb.pack import Chunk
    t0 = time.time(); os.makedirs(IDX, exist_ok=True)
    emb = Embedder(); leaf = kb.Leaf(emb)
    out, seen = [], set()
    fl = list(files())
    for p in fl:
        rel = os.path.relpath(p, ROOT)
        for head, ln, text in chunks(p, emb.tok):
            full = rel + (" — " + head if head else "") + "\n" + text      # what is embedded and keyword-indexed
            h = hashlib.sha1(text.encode()).hexdigest()[:16]
            if h in seen: continue                                  # same text in another file (copies, generated dirs)
            seen.add(h)
            out.append(Chunk(id=hashlib.sha1(full.encode()).hexdigest()[:16], text=full, meta={"path": rel, "head": head, "line": ln}))
    if "--full" in sys.argv:
        import shutil; shutil.rmtree(kb.KBC, ignore_errors=True); os.path.exists(kb.KBI) and os.remove(kb.KBI)
    w = kb.writer(leaf); d = w.sync(out)                            # content-addressed: only new or changed passages are embedded
    with contextlib.redirect_stderr(io.StringIO()):                 # bm25s progress bars
        if d.changed or not os.path.exists(kb.KBI): w.commit()
        if w.tombstone_ratio > 0.3: w.compact()                     # rebuilds from live rows; re-embeds everything (about 5 minutes)
    kb.add_sidecar()
    stats = {"files": len(fl), "chunks": w.live_count, "added": d.added + d.updated, "deleted": d.deleted, "embedded_now": leaf.n,
             "embed_ms_per_chunk": round(1000 * leaf.seconds / max(1, leaf.n), 2), "tombstones": w.tombstone_count, "seconds": round(time.time() - t0, 1),
             "kbi_mb": round(os.path.getsize(kb.KBI) / 1e6, 1), "built": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(stats, open(f"{IDX}/stats.json", "w")); print(json.dumps(stats))

if __name__ == "__main__": main()
