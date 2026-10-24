"""Drive web/index.html in headless Chromium (SwiftShader WebGPU): autorun the quick sweeps, save
the profile, count console errors, and screenshot desktop light/dark and phone portrait.

    python3 run/page_check.py [--only bandwidth,reduce] [--name swiftshader_page]
SwiftShader is a CPU rasterizer: this checks that the page works, not GPU speed (METHODS.md).
"""
import argparse, functools, http.server, json, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
OUT = HERE.parent / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--name", default="swiftshader_page")
ap.add_argument("--timeout", type=int, default=1800)
a = ap.parse_args()

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}/index.html"
flags = ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader",
         "--enable-features=Vulkan", "--use-vulkan=swiftshader"]
errors, t0 = [], time.time()
with sync_playwright() as p:
    b = p.chromium.launch(args=flags)
    pg = b.new_page(viewport={"width": 1100, "height": 900})
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    q = "?autorun&quick" + (f"&only={a.only}" if a.only else "")
    pg.goto(url + q)
    pg.wait_for_function("window.__refractometer && window.__refractometer.done", timeout=a.timeout * 1000, polling=2000)
    res = pg.evaluate("window.__refractometer")
    OUT.mkdir(exist_ok=True)
    pg.screenshot(path=str(OUT / f"{a.name}_light.png"), full_page=True)
    pg.emulate_media(color_scheme="dark")
    pg.evaluate("dispatchEvent(new Event('resize'))"); pg.wait_for_timeout(500)
    pg.screenshot(path=str(OUT / f"{a.name}_dark.png"), full_page=True)
    # Phone portrait: reload at 390 px in the same context, so the page renders the profile the
    # autorun saved to localStorage (the saved-profile path) at phone width.
    pg.emulate_media(color_scheme="light")
    pg.set_viewport_size({"width": 390, "height": 844})
    pg.goto(url)
    pg.wait_for_timeout(4000)
    overflow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    rendered = pg.evaluate("document.querySelectorAll('.chart svg').length")
    pg.screenshot(path=str(OUT / f"{a.name}_phone.png"), full_page=True)
    b.close()
prof = res.get("profile") or {}
summary = {
    "error": res.get("error"), "seconds": round(time.time() - t0, 1), "console_errors": errors,
    "phone_horizontal_overflow_px": overflow, "phone_saved_charts": rendered, "key": prof.get("key"), "timing": prof.get("timing"),
    "best": prof.get("best"),
    "statuses": {k: {s: sum(1 for p in v.get("points", []) if p.get("status") == s) for s in ("ok", "wrong", "invalid", "error")} | ({"skipped": v["skipped"]} if v.get("skipped") else {}) for k, v in (prof.get("sweeps") or {}).items()},
}
(OUT / f"{a.name}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ("error", "seconds", "console_errors", "phone_horizontal_overflow_px", "key", "statuses")}, indent=1))
