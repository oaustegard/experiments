"""Worker search path (JS) against remax_kb's Python reader: same token ids, same fused top 40.

    EXPERIMENTS_SPOKES_ROOT=... python3 -I run/parity.py KBI_PREFIX [N=40]

Takes the first N queries of each set, runs run/fused_js.mjs (worker/_lib/ettin-cpu.js, the
@huggingface/tokenizers package and kb-reader.js's fuseRanks at alpha 0.7 over a 40-candidate pool)
and Python kb.search(k=40, alpha=0.7, over_fetch=40, min_sim='off') with scripts/ettin_embedder.py.
Writes results/parity.json.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import spoke   # EXPERIMENTS_SPOKES_ROOT holds remax, remax_kb, muninn.austegard.com
for p in (spoke("remax_kb"), spoke("remax") / "src", spoke("muninn.austegard.com") / "scripts"):
    sys.path.insert(0, str(p))
from remax_kb.read_v2 import KB
from ettin_embedder import EttinEmbedder, ensure_package
ROOT = Path(__file__).resolve().parents[1]
prefix = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
Q = sum((json.loads((ROOT / "data" / f).read_text())[:N] for f in ("queries.json", "queries_hard.json")), [])
with tempfile.TemporaryDirectory() as td:
    qf, of = Path(td) / "q.json", Path(td) / "out.json"
    qf.write_text(json.dumps(Q))
    subprocess.run(["node", str(ROOT / "run" / "fused_js.mjs"), prefix + ".kbi", str(ensure_package()),
                    str(spoke("muninn.austegard.com")), str(qf), "0.7", str(of)], check=True)
    J = json.loads(of.read_text())
kb = KB.open(prefix + ".kbi"); emb = EttinEmbedder()
ids = sum(emb.ids(j["query"], "query") == j["ids"] for j in J)
fused = sum([kb._chunk_id_at(h.row) for h in kb.search(j["query"], embedder=emb, k=40, alpha=0.7,
             over_fetch=40, min_sim="off")] == j["fused"] for j in J)
out = {"queries": len(J), "token_ids_equal": ids, "fused_top40_equal": fused}
(ROOT / "results" / "parity.json").write_text(json.dumps(out, indent=1)); print(out)
