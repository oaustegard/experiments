"""Copy web/demo/ into the austegard.com site repo as bsky/embedders/, self-contained except for the
EG2 4-layer cut, which it shares with bsky/image-search/ (same model, engine and remex params).

Usage: .venv/bin/python -I run/publish_site.py [SITE_REPO]     (default ~/Projects/oaustegard.github.io)

Layout written under SITE_REPO/bsky/embedders/:
  index.html, demo.js (paths block rewritten), sw.js, engine.js (ettin-native), tail.js
  (embeddinggemma2-webgpu-tail, the tested version), README.md
  models/{ettin,leaf}/  model.onnx, tokenizer.json, meta.json, manifest.json
  index/{ettin,leaf,eg2}/  index.json, codes.u8.bin, aux.f32.bin   (eg2 = the full-EG2 posts index)
  data/corpus.json.gz   authors, rows and post texts, gzip -9 (decompressed in the page)
Refuses to run if the shared image-search files differ from the ones the demo was tested with.
"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
DEMO = ROOT / "web" / "demo"
TAILW = EXP / "embeddinggemma2-webgpu-tail" / "web"

PATHS = '''// --- paths: published layout (ettin-native/run/publish_site.py) ---
import { RemexTail } from "./tail.js";
import { SmallEngine } from "./engine.js";
import { Eg2Engine } from "../image-search/engine.js";
const P = {
  tail: "tail.js", smallEngine: "engine.js", eg2Engine: "../image-search/engine.js",
  params: "../image-search/params.json", corpus: "data/corpus.json.gz",
  ettinModel: "models/ettin/", ettinIndex: "index/ettin/",
  leafModel: "models/leaf/", leafIndex: "index/leaf/",
  eg2Model: "../image-search/model/", eg2Index: "index/eg2/", codes: "codes.u8.bin", aux: "aux.f32.bin",
};
// --- end paths ---'''

README = '''# Three query embedders, side by side

https://austegard.com/bsky/embedders/

Searches 61,686 posts from the "Artificial (and real) Intelligence" Bluesky list with three
in-browser query embedders at once: Ettin-17m, leaf-ir and the EmbeddingGemma 2 4-layer cut.
Each embeds the query on a hand-written WebGPU engine, and a GPU scan (RemexTail) searches
2-bit codes of the posts. "Make available offline" caches everything, after which the page
works with the network off (service worker, cache first).

Ettin-17m and leaf-ir search indexes of the posts they embedded themselves. The cut queries the
full-EG2 index and shares its model, engine and remex parameters with ../image-search/.

Built by oaustegard/experiments `ettin-native/run/publish_site.py`; write-up in
`ettin-native/RESULTS.md`. Needs WebGPU (Chrome, Edge, Safari 26+).
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Projects" / "oaustegard.github.io"
    img = site / "bsky" / "image-search"
    for a, b in ((img / "engine.js", EXP / "eg2-wgsl-engine" / "web" / "engine.js"),
                 (img / "params.json", TAILW / "assets" / "params.json"),
                 *((img / "model" / f, DEMO / "eg2cut" / f) for f in ("model_t3w.onnx", "model_t3w.onnx_data", "manifest.json", "tokenizer.json"))):
        assert sha(a) == sha(b), f"{a} differs from the tested {b}: refusing to publish against it"
    out = site / "bsky" / "embedders"
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    js = (DEMO / "demo.js").read_text()
    js, n = re.subn(r"// --- paths:.*?// --- end paths ---", PATHS, js, flags=re.S)
    assert n == 1
    (out / "demo.js").write_text(js)
    for f in ("index.html", "sw.js"):
        shutil.copy(DEMO / f, out / f)
    shutil.copy(ROOT / "web" / "engine.js", out / "engine.js")
    shutil.copy(TAILW / "tail.js", out / "tail.js")
    (out / "README.md").write_text(README)
    for key, pkg in (("ettin", "e17c-t4-w8L63"), ("leaf", "leafirc-t4-w8L63")):
        (out / "models" / key).mkdir(parents=True)
        for f in ("model.onnx", "tokenizer.json", "meta.json", "manifest.json"):
            shutil.copy(ROOT / "web" / "models" / pkg / f, out / "models" / key / f)
        src = DEMO / "index" / pkg
        (out / "index" / key).mkdir(parents=True)
        shutil.copy(src / "index.json", out / "index" / key / "index.json")
        shutil.copy(src / "codes.u8.txt", out / "index" / key / "codes.u8.bin")
        shutil.copy(src / "aux.f32.txt", out / "index" / key / "aux.f32.bin")
    (out / "index" / "eg2").mkdir(parents=True)
    eg2 = json.loads((TAILW / "bsky" / "index.json").read_text())
    eg2["files"] = {"codes": "codes.u8.bin", "aux": "aux.f32.bin"}
    (out / "index" / "eg2" / "index.json").write_text(json.dumps(eg2, separators=(",", ":")))
    shutil.copy(TAILW / "bsky" / "codes.u8.txt", out / "index" / "eg2" / "codes.u8.bin")
    shutil.copy(TAILW / "bsky" / "aux.f32.txt", out / "index" / "eg2" / "aux.f32.bin")
    (out / "data" / "corpus.json.gz").write_bytes(gzip.compress((DEMO / "data" / "corpus.json").read_bytes(), 9, mtime=0))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(out, f"{total / 1e6:.1f} MB", {str(f.relative_to(out)): f.stat().st_size for f in sorted(out.rglob("*")) if f.is_file() and f.stat().st_size > 1e6})


if __name__ == "__main__":
    main()
