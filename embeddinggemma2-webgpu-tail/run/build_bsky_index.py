"""Embed the harvested Bluesky posts and write the 2-bit index the search page loads.

Usage: python3 -I run/build_bsky_index.py [--batch 32] [--max-tokens 128] [--embed-only] [--limit N]

--limit N indexes only the first N posts (a quick end-to-end test while the full embed runs);
--embed-only stops after the embeddings.

Input: out/bsky/posts.jsonl (run/harvest_bsky.py). Steps:
  1. embed every post with model_q4.onnx on onnxruntime's CPU provider (the graph the
     page runs on WebGPU; cosine 1.0 between the two in round 2), document prompt
     "title: none | text: ", batches sorted by token count, truncated to --max-tokens
     (128 covers the 95th percentile of 92 tokens; CPU throughput here is ~10 posts/s).
     Checkpoints out/bsky/emb/chunk_NNNNN.npy every 2,048 posts; a rerun resumes.
  2. remex Quantizer(768, bits=2, seed=42, mean=corpus mean): packed codes, scale, bias.
  3. writes web/bsky/ (served by Pages next to search.html):
       index.json   n, d, bits, mu, the author table, source and build info
       codes.u8.txt n x 192 bytes, remex byte order, doc-major
       aux.f32.txt  n x 2 float32 (scale, bias), little-endian
       posts.json   per post [author index, rkey]; no text (the page fetches it from Bluesky)
       parity.json  test queries with remex search_adc top-10 (indices and scores) and
                    the exact fp32 top-10, for the page's startup check
Writes the fp32 embeddings to out/bsky/emb.npy for reuse (embstore put).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import remex
from remex.packing import pack
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out" / "bsky"
EMB = SRC / "emb"
HF = ROOT / "out" / "hf"
WEB = ROOT / "web" / "bsky"
CHUNK = 2048
DOC = "title: none | text: "
QUERY = "task: search result | query: "
PARITY_QUERIES = [
    "scaling laws for large language models",
    "is AI going to take software engineering jobs",
    "interpretability research on neurons and features",
    "open weights models versus closed models",
    "AI and copyright for training data",
    "benchmarks are saturated and misleading",
    "how the brain learns compared to neural networks",
    "agents that write and run code",
]


def load_posts():
    rows, authors, seen = [], [], set()
    for line in (SRC / "posts.jsonl").open():
        m = json.loads(line)
        ai = None
        for p in m["posts"]:
            if p["uri"] in seen:
                continue
            seen.add(p["uri"])
            if ai is None:
                ai = len(authors)
                authors.append({"did": m["did"], "handle": m["handle"]})
            rows.append({"a": ai, "rkey": p["uri"].rsplit("/", 1)[1], "text": p["text"],
                         "created_at": p.get("created_at")})
    return rows, authors


def session():
    so = ort.SessionOptions()
    return ort.InferenceSession(str(HF / "model_q4.onnx"), so, providers=["CPUExecutionProvider"])


def special_strings(tok_path):
    """Every added special token's text. A post containing one (a real post has "<|image|>")
    would otherwise tokenize to the image token, and the graph's multimodal merge then fails
    looking for image features. Stripped from documents and queries alike; search.js does the same."""
    t = json.loads(Path(tok_path).read_text())
    return sorted((a["content"] for a in t["added_tokens"] if a.get("special")), key=len, reverse=True)


SPECIALS = None


def clean(text):
    global SPECIALS
    if SPECIALS is None:
        SPECIALS = special_strings(HF / "tokenizer.json")
    for sp in SPECIALS:
        if sp in text:
            text = text.replace(sp, " ")
    return text


def embed(sess, tok, texts, max_tokens):
    enc = [tok.encode(clean(t)).ids[:max_tokens] for t in texts]
    S = max(map(len, enc))
    ids = np.zeros((len(enc), S), np.int64)
    attn = np.zeros((len(enc), S), np.int64)
    for b, row in enumerate(enc):
        ids[b, : len(row)] = row
        attn[b, : len(row)] = 1
    empty = np.zeros((0, 512), np.float32)
    out = sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": attn,
                   "image_features": empty, "video_features": empty, "audio_features": empty})[0]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def embed_all(rows, batch, max_tokens):
    EMB.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    sess = session()
    n = len(rows)
    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        path = EMB / f"chunk_{c0 // CHUNK:05d}.npy"
        if path.exists():
            continue
        idx = list(range(c0, min(n, c0 + CHUNK)))
        lens = [len(rows[i]["text"]) for i in idx]
        order = [idx[j] for j in np.argsort(lens)]      # length-sorted batches pad less
        out = np.zeros((len(idx), 768), np.float32)
        for b in range(0, len(order), batch):
            sel = order[b: b + batch]
            e = embed(sess, tok, [DOC + rows[i]["text"] for i in sel], max_tokens)
            for i, v in zip(sel, e):
                out[i - c0] = v
        np.save(path, out)
        done = min(n, c0 + CHUNK)
        rate = done / max(1e-9, time.time() - t0)
        print(f"embedded {done}/{n} ({rate:.1f} posts/s this run)", flush=True)
    return np.concatenate([np.load(p) for p in sorted(glob.glob(str(EMB / "chunk_*.npy")))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--embed-only", action="store_true")
    a = ap.parse_args()
    rows, authors = load_posts()
    if a.limit:
        # whole chunks only, so the checkpoints stay valid for the full run (posts.jsonl only grows)
        if a.limit % CHUNK:
            raise SystemExit(f"--limit must be a multiple of {CHUNK}")
        rows = rows[: a.limit]
        used = sorted({r["a"] for r in rows})
        remap = {old: new for new, old in enumerate(used)}
        authors = [authors[i] for i in used]
        for r in rows:
            r["a"] = remap[r["a"]]
    print(f"{len(rows)} posts by {len(authors)} authors", flush=True)
    X = embed_all(rows, a.batch, a.max_tokens)
    if a.limit:
        X = X[: a.limit]
    assert len(X) == len(rows), (len(X), len(rows))
    np.save(SRC / "emb.npy", X)
    if a.embed_only:
        return 0

    mu = X.mean(0).astype(np.float32)
    qz = remex.Quantizer(768, bits=2, seed=42, mean=mu)
    cv = qz.encode(X)
    packed = pack(cv.indices.astype(np.uint8), 2).reshape(len(X), 192)
    aux = np.stack([qz._score_norms(cv), qz._row_bias(cv)], 1).astype("<f4")

    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    Q = embed(session(), tok, [QUERY + q for q in PARITY_QUERIES], a.max_tokens).astype(np.float32)
    parity = []
    for q, v in zip(PARITY_QUERIES, Q):
        idx, sc = qz.search_adc(cv, v, k=10)
        exact = np.argsort(-(X @ v))[:10]
        parity.append({"query": q, "remex_top10": [int(i) for i in idx],
                       "remex_scores": [float(s) for s in sc], "exact_top10": [int(i) for i in exact]})

    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "codes.u8.txt").write_bytes(np.ascontiguousarray(packed).tobytes())
    (WEB / "aux.f32.txt").write_bytes(aux.tobytes())
    (WEB / "posts.json").write_text(json.dumps([[r["a"], r["rkey"]] for r in rows], separators=(",", ":")))
    (WEB / "parity.json").write_text(json.dumps(parity, indent=1))
    members = json.loads((SRC / "members.json").read_text())
    dates = sorted(r["created_at"] for r in rows if r["created_at"])
    (WEB / "index.json").write_text(json.dumps({
        "n": len(rows), "d": 768, "bits": 2, "seed": 42, "rotation": "rht",
        "remex_version": remex.__version__, "model": "onnx-community/embeddinggemma-2-ONNX model_q4.onnx",
        "doc_prompt": DOC, "query_prompt": QUERY, "max_tokens": a.max_tokens,
        "source": {"list": members["list"], "list_name": members["name"],
                   "filter": "posts_and_author_threads, own posts, >= 20 chars, newest 200 per member",
                   "first_post": dates[0] if dates else None, "last_post": dates[-1] if dates else None},
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authors": authors,
        "mu": [float(v) for v in mu],
        "files": {"codes": "codes.u8.txt", "aux": "aux.f32.txt", "posts": "posts.json", "parity": "parity.json"},
    }, separators=(",", ":")))
    print(json.dumps({"n": len(rows), "codes_bytes": packed.nbytes, "aux_bytes": aux.nbytes}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
