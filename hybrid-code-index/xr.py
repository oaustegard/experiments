"""xr — cross-repo search over the account index, fast enough to reach for.

    xr "how do we avoid re-encoding unchanged content"
    xr -r experiments "concurrent LLM calls through a gateway"
    xr -k 20 --json "haar versus rht rotation"

Why this exists rather than `hcindex.load` + a few lines of numpy: loading the
published index that way measured **21 s**, of which 17.5 s was `load()`
rebuilding a 221,832-entry Python postings dict that the query then uses for a
handful of lookups. A tool nobody reaches for is not a tool, and 21 s is well
past the point where reaching stops.

Two changes get it to ~2.5 s cold, essentially all of it the ONNX session:

  * a prepared cache of plain .npy files, mmapped rather than decompressed. The
    dense matrix is stored **decoded and L2-normalized**, so the 2-bit codes are
    never decoded at query time at all. It is float32 and not float16: float16
    halves the 62 MB but numpy has no BLAS path for it and upcasts elementwise,
    which measured **8.7 s per query** against ~10 ms for the float32 matmul.
    Storage was the wrong thing to optimize.
  * top-k by `argpartition` over a bounded candidate pool rather than
    `argsort` over all 42,488 rows followed by a Python dedupe loop.
  * BM25 scored straight off the flat postings arrays. `bm_terms` is sorted, so
    a term resolves by `np.searchsorted` and its postings are an array slice.
    No dict is ever built, and the per-term inner loop vectorizes.

The cache is keyed by the index's `built_at`, so a rebuilt index re-prepares
once and every later query mmaps it.

`-r/--repo` is not a convenience filter. The account index buries findings that
a per-repo index surfaces immediately: "about to fan out concurrent LLM calls
through a Cloudflare gateway" returns nothing relevant in the top 20
account-wide, because claude-skills contributes 8,088 chunks dense with
API-invocation vocabulary. Scoped with `-r experiments` the same query puts
phase-a-bridges at #2 and te-bridges/RESULTS.md at #8. Scope when you know
where to look; go wide when you do not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "repo-index"))

REPO = os.environ.get("XR_INDEX_REPO", "oaustegard/claude-workspace")
TAG = os.environ.get("XR_INDEX_TAG", "account-index-v1")
CACHE = Path(os.environ.get("XR_CACHE", Path.home() / ".cache" / "xr"))
K1, B = 1.2, 0.75


# ── fetching ────────────────────────────────────────────────────────────────

def _api(url: str, accept: str = "application/vnd.github+json") -> bytes:
    tok = os.environ.get("GH_TOKEN", "")
    req = urllib.request.Request(url, headers={"Accept": accept,
                                               **({"Authorization": f"Bearer {tok}"} if tok else {})})
    return urllib.request.urlopen(req).read()


def fetch(dest: Path) -> Path:
    """Download the published index if the local copy is missing or stale."""
    rel = json.loads(_api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"))
    assets = {a["name"]: a for a in rel["assets"]}
    dest.mkdir(parents=True, exist_ok=True)
    npz = dest / "account-index.npz"
    stamp = dest / "asset.json"
    want = assets["account-index.npz"]["updated_at"]
    if npz.exists() and stamp.exists() and json.loads(stamp.read_text()).get("updated_at") == want:
        return npz
    print(f"fetching {TAG} ({assets['account-index.npz']['size']/2**20:.1f} MB) ...",
          file=sys.stderr)
    npz.write_bytes(_api(assets["account-index.npz"]["url"], "application/octet-stream"))
    stamp.write_text(json.dumps({"updated_at": want}))
    return npz


# ── cache preparation ───────────────────────────────────────────────────────

def prepare(npz: Path, out: Path) -> None:
    """Convert the published .npz into mmap-friendly plain arrays.

    Run once per published index. Everything here is the work the query path
    must not do: decompressing, decoding 2-bit codes, normalizing.
    """
    import remex
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))

    qz = remex.Quantizer(d=meta["dim"], bits=meta["bits"], seed=meta["seed"],
                         rotation=meta["rotation"])
    cv = remex.CompressedVectors(indices=z["codes"],
                                 norms=np.ones(len(z["files"]), np.float32),
                                 d=meta["dim"], bits=meta["bits"],
                                 rotation=meta["rotation"])
    x = qz.decode(cv)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    np.save(out / "dense.npy", np.ascontiguousarray(x, dtype=np.float32))

    for k in ("files", "lines", "bm_terms", "bm_offs", "bm_docs", "bm_tfs", "bm_lens"):
        np.save(out / f"{k}.npy", z[k])
    (out / "meta.json").write_text(json.dumps(meta))


def cache_for(npz: Path) -> Path:
    """Prepared-cache directory for this index, keyed by its build timestamp."""
    z = np.load(npz, allow_pickle=False)
    built = json.loads(str(z["meta"][0]))["built_at"].replace(":", "").replace("-", "")
    d = CACHE / f"prepared-{built}"
    if not (d / "meta.json").exists():
        print(f"preparing cache for {built} (once) ...", file=sys.stderr)
        prepare(npz, d)
    return d


# ── query ───────────────────────────────────────────────────────────────────

class Index:
    def __init__(self, cache: Path):
        # Read outright rather than mmap. The dense matmul touches every one of
        # the 62 MB, so mmap just converts one sequential read into 15k random
        # page faults: measured 7.1 s per query with mmap_mode='r' (6.0 s of it
        # system time) against ~0.4 s reading it up front. mmap pays off for
        # sparse access; this access pattern is total.
        self.dense = np.load(cache / "dense.npy")
        self.files = np.load(cache / "files.npy")
        self.lines = np.load(cache / "lines.npy")
        self.terms = np.load(cache / "bm_terms.npy")
        # postings are touched a few slices per query, so these do stay mmapped
        m = lambda n: np.load(cache / f"{n}.npy", mmap_mode="r")  # noqa: E731
        self.offs = m("bm_offs"); self.docs = m("bm_docs"); self.tfs = m("bm_tfs")
        self.lens = np.load(cache / "bm_lens.npy")
        self.meta = json.loads((cache / "meta.json").read_text())
        self.n = len(self.files)
        self.avg = float(self.lens.mean())

    def bm25(self, q: str, mask: np.ndarray | None = None) -> np.ndarray:
        """Score straight off the flat postings arrays -- no dict, vectorized."""
        import hcindex as H
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(H.tokens(q)):
            i = int(np.searchsorted(self.terms, t))
            if i >= len(self.terms) or self.terms[i] != t:
                continue
            lo, hi = int(self.offs[i]), int(self.offs[i + 1])
            d = np.asarray(self.docs[lo:hi], dtype=np.int64)
            f = np.asarray(self.tfs[lo:hi], dtype=np.float32)
            df = hi - lo
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            dl = self.lens[d].astype(np.float32)
            s[d] += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avg))
        if mask is not None:
            s[~mask] = 0.0
        return s

    def _top_files(self, scores: np.ndarray, k: int,
                   pool: int = 3000) -> list[tuple[str, int]]:
        """Best-chunk-per-file ranking over a bounded candidate pool.

        `hcindex.to_files` argsorts all 42,488 rows and then walks them in
        Python doing `f in best` against a list. Partitioning to the top `pool`
        first and deduping with a set gives the same answer for any sane k --
        no file can enter the top k on a chunk that is not in the top `pool`.
        """
        n = len(scores)
        m = min(pool, n)
        cand = np.argpartition(-scores, m - 1)[:m]
        cand = cand[np.argsort(-scores[cand])]
        best: list[tuple[str, int]] = []
        seen: set[str] = set()
        for i in cand:
            if not np.isfinite(scores[i]) or scores[i] <= 0:
                continue
            f = str(self.files[i])
            if f in seen:
                continue
            seen.add(f)
            # the line of the chunk that actually won, not the file's first
            best.append((f, int(self.lines[i])))
            if len(best) >= k:
                break
        return best

    def search(self, q: str, k: int = 8, repo: str | None = None,
               pool: int = 25) -> list[tuple[str, int]]:
        from ask import Encoder
        if not hasattr(self, "_enc"):
            self._enc = Encoder()
        qv = np.asarray(self._enc([q])[0], dtype=np.float32)

        mask = None
        if repo:
            pre = np.char.startswith(self.files, repo + "/")
            if not pre.any():
                # ValueError, not SystemExit: this runs inside the server, and
                # SystemExit derives from BaseException, so `except Exception`
                # would not catch it -- one bad `-r` took the whole server down
                # and the next client died on the connection reset.
                raise ValueError(
                    f"no chunks from repo {repo!r}; indexed repos include: "
                    + ", ".join(sorted(self.meta["repos"])[:8]) + " ...")
            mask = pre

        dense = self.dense @ qv
        if mask is not None:
            dense = np.where(mask, dense, -np.inf)
        lex = self.bm25(q, mask)

        import hcindex as H
        d = self._top_files(dense, pool)
        b = self._top_files(lex, pool)
        # keep each arm's winning line, preferring the dense arm's when both hit
        line = {f: l for f, l in b} | {f: l for f, l in d}
        out = H.rrf([[f for f, _ in d], [f for f, _ in b]])[:k]
        return [(f, line.get(f, 0)) for f in out]


# ── resident server ─────────────────────────────────────────────────────────
#
# Scoring a query costs 32 ms: 14 ms dense matmul, 17 ms BM25, 1 ms top-k. Every
# other millisecond of a fresh `xr` invocation is reloading state -- 62 MB of
# dense matrix and a 124 MB ONNX model, plus numpy and onnxruntime imports.
# Measured end-to-end for repeated one-shot runs: 11.0 s, 7.6 s, 6.1 s.
#
# So the tool is not slow, the process is. A resident server loads once and
# answers on a unix socket in ~40 ms, and the first query starts it
# automatically -- pay 8 s once per session, then it is a reflex. That is the
# difference between a thing you reach for and a thing you remember exists.

SOCK = CACHE / "xr.sock"
IDLE_EXIT = 3600.0


def serve(idx: "Index") -> None:
    import socket
    SOCK.parent.mkdir(parents=True, exist_ok=True)
    if SOCK.exists():
        SOCK.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK)); srv.listen(8); srv.settimeout(IDLE_EXIT)
    idx.search("warm the encoder", k=1)      # pay ONNX init before serving
    sys.stderr.write("xr: ready\n"); sys.stderr.flush()
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break                             # idle timeout -- exit, next query restarts
        with conn:
            try:
                req = json.loads(conn.recv(65536).decode())
                hits = idx.search(req["q"], k=req.get("k", 8), repo=req.get("repo"))
                body = {"results": [{"file": f, "line": l} for f, l in hits],
                        "meta": {"n_chunks": idx.meta["n_chunks"],
                                 "repos": len(idx.meta["repos"]),
                                 "built_at": idx.meta["built_at"]}}
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:        # noqa: BLE001 -- never take the server down
                body = {"error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall(json.dumps(body).encode())
            except OSError:
                pass                          # client hung up; keep serving
    SOCK.unlink(missing_ok=True)


def ask_server(q: str, k: int, repo: str | None, autostart: bool = True) -> dict | None:
    import socket
    import subprocess
    for attempt in range(2):
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(30)
            c.connect(str(SOCK))
            c.sendall(json.dumps({"q": q, "k": k, "repo": repo}).encode())
            buf = b""
            while chunk := c.recv(65536):
                buf += chunk
            if not buf:
                raise ConnectionResetError("empty response")
            return json.loads(buf.decode())
        except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError,
                json.JSONDecodeError, socket.timeout):
            SOCK.unlink(missing_ok=True)
            if not autostart or attempt:
                return None
            # detached so it outlives this shell
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            for _ in range(600):              # up to 60 s for the first load
                if SOCK.exists():
                    break
                time.sleep(0.1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="xr", description="cross-repo search over the account code index")
    ap.add_argument("query", nargs="*")
    ap.add_argument("--serve", action="store_true",
                    help="run the resident server in the foreground")
    ap.add_argument("--no-server", action="store_true",
                    help="score in-process (slow; for debugging the server)")
    ap.add_argument("--stop", action="store_true", help="stop the resident server")
    ap.add_argument("-r", "--repo", help="scope to one repo (strongly recommended "
                                         "when you know where to look)")
    ap.add_argument("-k", type=int, default=8, help="results (default 8)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-check the release")
    ap.add_argument("--stats", action="store_true", help="print index + timing info")
    a = ap.parse_args()

    if a.stop:
        import socket
        if SOCK.exists():
            SOCK.unlink()
            print("xr: server socket removed; it exits on its next idle timeout")
        else:
            print("xr: no server running")
        return

    def open_index() -> "Index":
        raw = CACHE / "raw"
        if a.refresh and (raw / "asset.json").exists():
            (raw / "asset.json").unlink()
        return Index(cache_for(fetch(raw)))

    if a.serve:
        serve(open_index())
        return
    if not a.query:
        ap.error("a query is required")

    q = " ".join(a.query)
    t0 = time.time()
    resp = None if a.no_server else ask_server(q, a.k, a.repo)
    if resp is None:                       # server unavailable -- score here
        idx = open_index()
        resp = {"results": [{"file": f, "line": l}
                            for f, l in idx.search(q, k=a.k, repo=a.repo)],
                "meta": {"n_chunks": idx.meta["n_chunks"],
                         "repos": len(idx.meta["repos"]),
                         "built_at": idx.meta["built_at"]}}
    dt = time.time() - t0
    if "error" in resp:
        raise SystemExit(f"xr: {resp['error']}")

    if a.json:
        print(json.dumps({"query": q, "repo": a.repo, **resp}, indent=1))
    else:
        for r in resp["results"]:
            print(f"{r['file']}:{r['line']}")
    if a.stats:
        m = resp["meta"]
        print(f"\n{m['n_chunks']:,} chunks / {m['repos']} repos "
              f"· built {m['built_at']}\n{dt*1000:.0f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
