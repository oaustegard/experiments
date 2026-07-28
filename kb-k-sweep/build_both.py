#!/usr/bin/env python3
"""Build two real v2 .kbi artifacts from the SAME float embeddings:
   baseline dim=256/k=8 (shipped) and candidate dim=512/k=4.

Feeds the production KBWriter a cached-embedder shim backed by embeddings.npz,
so both builds binarize identical Gemini vectors — the only difference is
(dim, k). Re-walks the corpus for chunk text/meta (no embedding); asserts the
walk order matches the cached ids.
"""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / ".spokes" / "muninn.austegard.com"
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]

CONFIGS = [(256, 8), (512, 4), (768, 2)]
BUILD = HERE / "build"


def load_bm():
    spec = importlib.util.spec_from_file_location("bm", SITE / "scripts" / "build_muninn_kb.py")
    bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    return bm


class CachedEmbedder:
    """remax_kb embedder protocol, but encode() returns precomputed vectors.

    Delegates all fingerprint/metadata to a real GeminiGatewayEmbedder (no
    network — fingerprint is static), overrides encode() with a text->vec map.
    """
    def __init__(self, real, text_to_vec):
        self._real = real
        self._map = text_to_vec

    @property
    def full_dim(self): return self._real.full_dim
    @property
    def normalize_l2(self): return self._real.normalize_l2
    @property
    def prompts(self): return self._real.prompts
    @property
    def release_url(self): return self._real.release_url
    @property
    def release_sha256(self): return self._real.release_sha256
    @property
    def model_revision(self): return self._real.model_revision
    def fingerprint(self): return self._real.fingerprint()

    def encode(self, texts, *, prompt="document"):
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self._map.get(t)
            if v is None:
                raise KeyError(f"no cached embedding for text (len={len(t)}): {t[:60]!r}")
            out[i] = v
        return out


def main():
    bm = load_bm()
    d = np.load(HERE / "embeddings.npz", allow_pickle=True)
    vecs = d["vecs"].astype(np.float32); ids = list(d["ids"])
    chunks = list(bm.walk_corpus(SITE))
    assert [c.id for c in chunks] == ids, "corpus walk order drifted from cache"
    print(f"{len(chunks)} chunks; cache aligned")

    text_to_vec = {c.text: vecs[i] for i, c in enumerate(chunks)}
    real = bm.GeminiGatewayEmbedder(
        account_id=os.environ["CF_ACCOUNT_ID"],
        gateway_id=os.environ["CF_GATEWAY_ID"],
        gateway_token=os.environ["CF_API_TOKEN"])
    cached = CachedEmbedder(real, text_to_vec)

    from remax_kb.pack_v2 import KBWriter
    import shutil
    for dim, k in CONFIGS:
        name = f"muninn_d{dim}_k{k}"
        out_dir = BUILD / f"d{dim}_k{k}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        w = KBWriter.create(name=name, output_dir=out_dir, embedder=cached,
                            dim=dim, k=k, seed=0, chunks_uri=f"./{name}.kbc/",
                            source=f"muninn.austegard.com — k-sweep verify build (dim={dim},k={k})")
        w.add_chunks(chunks)
        w.commit()
        kbi = out_dir / f"{name}.kbi"
        kbc = out_dir / f"{name}.kbc"
        shard_kb = sum(p.stat().st_size for p in kbc.glob("shard-*.bin")) / 1024
        print(f"[dim={dim} k={k}] {kbi.name} {kbi.stat().st_size/1024:.1f} KB "
              f"+ {shard_kb:.1f} KB chunks  -> {out_dir}")


if __name__ == "__main__":
    main()
