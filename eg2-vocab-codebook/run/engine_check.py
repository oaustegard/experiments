"""Check the WGSL engine on an rhq-table model (run/pack_engine.py output) against a NumPy reference.

Usage: python3 -I run/engine_check.py DIR
The reference is eg2-wgsl-engine/run/ref.py's forward pass with the rhq row decode in place of the
3-bit grouped one; it is also checked against the PyTorch student given the same decoded rows.
Writes DIR/queries.json (ids + reference vectors) and runs web/check.html in headless Chromium
(SwiftShader WebGPU). Passes at cosine >= 0.9999 for every query, single and batched.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "eg2-wgsl-engine" / "run"))
sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
import ref  # noqa: E402
import rhq  # noqa: E402
from run_demo import CHROME, CHROME_ARGS, free_port, wait_for_server  # noqa: E402

TEXTS = ref.TEXTS + ["a heron standing in shallow water, reflected, with reeds behind it",
                     "screenshot of a terminal window running htop on a raspberry pi"]


def rhq_rows(data, tbl, ids):
    t = lambda k: ref.tensor(data, tbl[k])  # noqa: E731
    ids = np.asarray(ids)
    out = np.empty((len(ids), 512), np.float32)
    n3 = tbl["n3"]
    sc = t("scales")
    for bits, key, m, off in ((3, "codes3", ids < n3, 0), (2, "codes2", ids >= n3, n3)):
        if m.any():
            codes = rhq.unpack(t(key)[ids[m] - off], bits)
            out[m] = rhq.decode(codes, sc[ids[m]], bits)
    return out


def main():
    d = Path(sys.argv[1]).resolve()
    man = json.loads((d / "manifest.json").read_text())
    data = (d / man["data"]).read_bytes()
    ref.table_rows = rhq_rows
    R = ref.Ref(man, data)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    dec = np.load(d / "table_decoded.npy")
    ids = [tok.encode(ref.QUERY + t).ids[:128] for t in TEXTS]
    out = []
    for t, x in zip(TEXTS, ids):
        r = R(x)
        assert np.allclose(rhq_rows(data, man["table"], x), dec[x], atol=1e-5)
        out.append({"text": t, "ids": x, "ref": r.tolist()})
    (d / "queries.json").write_text(json.dumps(out))
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from playwright.sync_api import sync_playwright
    rel = d.relative_to(REPO)
    url = f"http://127.0.0.1:{port}/{HERE.name}/web/check.html?model=/{rel}/"
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            page = b.new_page()
            page.goto(url)
            page.wait_for_function("['done', 'error'].includes(window.result.phase)", timeout=1_800_000)
            res = page.evaluate("window.result")
            b.close()
    finally:
        server.terminate()
    print(json.dumps(res)[:2000])
    ok = res.get("phase") == "done" and res["cos_single_min"] >= 0.9999 and res["cos_batch_min"] >= 0.9999
    (d / "check.json").write_text(json.dumps(res, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
