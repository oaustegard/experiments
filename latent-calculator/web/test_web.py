"""Headless-Chromium check that the page reproduces pipeline.py exactly.

    python3 test_web.py [--n 3] [--variant fp32] [--query learned]

Serves web/ over a local http.server, loads index.html with ?ort=local (the
vendored onnxruntime-web, because browser egress is not proxied in this
container), runs N prompts on the WASM backend and asserts the latent-route
text, the decoded query and the calculator result all equal pipeline.py's.
Then tries a WebGPU launch and reports whether the EP initializes; if it does,
the same prompts are re-run and compared token for token against WASM.

Playwright's bundled chromium is not the one in this image, so the browser is
launched with an explicit executable_path.
"""

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
PROMPTS = ["4567 + 89 =", "Subtract 987 from 1000.",
           "Which is larger, 3436 or 9549?"]


def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=directory)
    class Q(socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True
    httpd = Q(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def run_browser(port, prompts, variant, query, args_extra=(), backend="wasm",
                timeout_ms=900_000, verbose=True):
    """Returns (backend_used, [row dicts]) or (None, error string)."""
    from playwright.sync_api import sync_playwright
    url = (f"http://127.0.0.1:{port}/index.html?ort=local"
           f"&autoload=0")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROME, headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", *args_extra])
        page = browser.new_page()
        msgs = []
        page.on("console", lambda m: msgs.append(m.text))
        page.on("pageerror", lambda e: msgs.append("PAGEERROR " + str(e)))
        page.goto(url, wait_until="load", timeout=60_000)
        page.select_option("#variant", variant)
        page.select_option("#backendSel", backend)
        page.select_option("#query", query)
        t0 = time.time()
        try:
            page.evaluate("() => window.LC.load()")
            page.wait_for_function("() => window.LC.loaded || window.LC.error",
                                   timeout=timeout_ms)
        except Exception as e:                                # noqa: BLE001
            browser.close()
            return None, f"load timeout/error: {str(e)[:200]} | {msgs[-5:]}"
        err = page.evaluate("() => window.LC.error || null")
        if err:
            browser.close()
            return None, err
        load_s = time.time() - t0
        used = page.evaluate("() => window.LC.backend")
        rows = []
        for p in prompts:
            r = page.evaluate(
                "async (p) => await window.LC.answer(p, "
                "document.getElementById('query').value)", p)
            rows.append(r)
            if verbose:
                print(f"  [{used}] {p!r} -> latent {r['latent']['text']!r} "
                      f"({r['latent']['ms']:.0f} ms, "
                      f"{r['latent']['tokens']} tok)", flush=True)
        browser.close()
    return used, {"rows": rows, "load_seconds": load_s, "console": msgs[-10:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--query", default="learned")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--json", default=os.path.join(HERE, "web_test.json"))
    ap.add_argument("--skip-webgpu", action="store_true")
    args = ap.parse_args()
    prompts = PROMPTS[:args.n]

    # --- python reference
    sys.path.insert(0, HERE)
    from transformers import AutoTokenizer

    import pipeline as P
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    pipe = P.Pipeline(variant=args.variant)
    ref = [pipe.answer(p, tok, query=args.query) for p in prompts]
    for p, r in zip(prompts, ref):
        print(f"  [python] {p!r} -> latent {r['latent']['text']!r}")

    httpd, port = serve(HERE)
    out = {"variant": args.variant, "query": args.query, "prompts": prompts,
           "python": [r["latent"] for r in ref]}

    used, res = run_browser(port, prompts, args.variant, args.query,
                            backend="wasm")
    assert used == "wasm", f"wasm run failed: {res}"
    out["wasm"] = {"backend": used, "load_seconds": res["load_seconds"],
                   "rows": res["rows"]}

    bad = []
    for p, got, want in zip(prompts, res["rows"], ref):
        for field in ("text", "query", "calculator"):
            if got["latent"][field] != want["latent"][field]:
                bad.append((p, field, got["latent"][field],
                            want["latent"][field]))
    out["wasm_matches_pipeline"] = not bad
    out["wasm_mismatches"] = bad

    n_tok = sum(r["latent"]["tokens"] for r in res["rows"])
    ms = sum(r["latent"]["ms"] for r in res["rows"])
    out["wasm_latent_ms_per_answer"] = ms / len(prompts)
    out["wasm_latent_tokens_per_s"] = n_tok / (ms / 1000)
    out["wasm_frozen_ms_per_answer"] = (
        sum(r["frozen"]["ms"] for r in res["rows"]) / len(prompts))
    print(f"\nWASM: {out['wasm_latent_ms_per_answer']:.0f} ms per latent answer, "
          f"{out['wasm_latent_tokens_per_s']:.2f} generated tok/s, "
          f"model load {res['load_seconds']:.1f}s")

    # --- WebGPU attempt
    if not args.skip_webgpu:
        gused, gres = run_browser(
            port, prompts, "fp16" if args.variant == "fp32" else args.variant,
            args.query,
            args_extra=["--enable-unsafe-webgpu",
                        "--enable-features=Vulkan,WebGPU",
                        "--use-angle=swiftshader"],
            backend="webgpu", timeout_ms=600_000)
        out["webgpu"] = {"backend": gused,
                         "detail": gres if gused is None else gres["rows"]}
        if gused == "webgpu":
            g_bad = [(p, g["latent"]["text"], w["latent"]["text"])
                     for p, g, w in zip(prompts, gres["rows"], res["rows"])
                     if g["latent"]["text"] != w["latent"]["text"]]
            out["webgpu_matches_wasm"] = not g_bad
            out["webgpu_mismatches"] = g_bad
            print(f"WebGPU initialized; matches WASM: {not g_bad}")
        else:
            print(f"WebGPU did not initialize: {str(gres)[:300]}")
    httpd.shutdown()

    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    if bad:
        print("MISMATCHES vs pipeline.py:")
        for b in bad:
            print("  ", b)
        sys.exit(1)
    print(f"\nbrowser latent route == pipeline.py on {len(prompts)}/"
          f"{len(prompts)} prompts -> {args.json}")
    print("DONE test_web")


if __name__ == "__main__":
    main()
