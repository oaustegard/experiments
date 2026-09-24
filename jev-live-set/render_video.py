"""Render a one-minute video of the Jev Live Set with real Jev decisions.

The page's own code performs the set offline (offlinePerform), one real Jev call per bar through
jev-tag-encoder/jev.py (direct TypeSafe transport), renders the audio with OfflineAudioContext
(renderAudio), then steps a virtual clock frame by frame (frameAt) while Playwright screenshots
a 1280x720 layout at 1.5x. ffmpeg muxes frames and audio.

Math.random is seeded and every Jev answer is cached by request hash, so a rerun with the same
seed replays the same set without new calls.

  python3 render_video.py OUT_DIR [--seed N] [--seconds 60] [--fps 30] [--frames-only-test]
"""
import argparse, base64, hashlib, json, os, subprocess, sys, time, wave
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(os.environ.get("EXPERIMENTS", "/home/user/experiments"))
sys.path.insert(0, str(EXPERIMENTS / "jev-tag-encoder"))
import jev  # noqa: E402
import imageio_ffmpeg  # noqa: E402

STEERS = [{"at": 17.0, "text": "more energy, push toward a drop"},
          {"at": 38.0, "text": "switch to a different style, something dreamy"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--test", action="store_true", help="perform + audio + 4 sample frames, no video")
    ap.add_argument("--audio-only", action="store_true", help="perform + audio, no frames")
    ap.add_argument("--only", default="", help="comma-separated layers to render (audio-only diagnostics)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "jev-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    calls = []

    def bridge(payload_json):
        inp = json.loads(payload_json)
        key = hashlib.sha256(json.dumps([inp["state"], inp["questions"]], sort_keys=True).encode()).hexdigest()
        if key in cache:
            res, dt = cache[key]["res"], cache[key]["dt"]
        else:
            res, dt, _ = jev.post(inp["state"], inp["questions"])
            cache[key] = {"res": res, "dt": dt}
            cache_path.write_text(json.dumps(cache))
        calls.append({"key": key[:12], "dt": dt, "state": inp["state"], "questions": list(inp["questions"])})
        tok = (res.get("usage") or {}).get("input_tokens", 0)
        return json.dumps(res["answers"]) + f"\n[usage: jev-1.13.0 input {tok}, output 0 tokens]"

    page_html = (HERE / "jev-live-set.html").read_text()
    pre = f"""<script>
(() => {{ let a = {a.seed} >>> 0; Math.random = () => {{ a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }}; }})();
window.claude = {{ use: (n) => n === "mcp" ? Promise.resolve({{ callTool: async (server, tool, input) => {{
  const t0 = performance.now(); const text = await window.jevBridge(JSON.stringify(input));
  return {{ payload: text, content: [{{ type: "text", text }}] }}; }} }}) : Promise.resolve(null) }};
</script>"""
    html = ('<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"></head>'
            '<body class="video">' + pre + page_html + '</body></html>')
    (out / "render.html").write_text(html)

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", args=["--autoplay-policy=no-user-gesture-required"])
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:300]))
        pg.expose_function("jevBridge", bridge)
        pg.goto("file://" + str((out / "render.html").resolve()))
        pg.wait_for_function("window.__jevlive && document.getElementById('conn').dataset.kind === 'ok'", timeout=15000)
        pg.evaluate("document.fonts.ready")
        t0 = time.time()
        perf = pg.evaluate("(o) => window.__jevlive.offlinePerform(o)", {"seconds": a.seconds, "steers": STEERS})
        print(f"performed {perf['bars']} bars, {perf['calls']} Jev calls in {time.time()-t0:.1f}s; steers {perf['steers']}", flush=True)
        (out / "run.json").write_text(json.dumps({"seed": a.seed, "steers": STEERS, "perform": perf, "calls": calls}, indent=1))
        au = pg.evaluate("([s, f, o]) => window.__jevlive.renderAudio(s, f, o)", [a.seconds, a.fps, a.only.split(",") if a.only else None])
        pcm = base64.b64decode(au["pcm"])
        with wave.open(str(out / (f"audio-{a.only}.wav" if a.only else "audio.wav")), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(au["sr"]); w.writeframes(pcm)
        print(f"audio {len(pcm)/4/au['sr']:.1f}s peak {au['peak']:.2f}", flush=True)
        rms = au["rms"]
        if a.audio_only:
            print("errors:", errs[:5]); b.close(); return
        pg.evaluate("window.__jevlive.fit()")
        n = int(a.seconds * a.fps)
        if a.test:
            for t in (1.0, 12.0, 30.0, 50.0):
                pg.evaluate("([t, l]) => window.__jevlive.frameAt(t, l)", [t, rms[int(t * a.fps)]])
                pg.screenshot(path=str(out / f"frame-{int(t):02d}.png"))
            print("errors:", errs[:5]); b.close(); return
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps), "-c:v", "mjpeg", "-i", "-",
                                 "-i", str(out / "audio.wav"), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                                 "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out / "jev-live-set.mp4")],
                                stdin=subprocess.PIPE)
        t0 = time.time()
        for i in range(n):
            pg.evaluate("([t, l]) => window.__jevlive.frameAt(t, l)", [i / a.fps, rms[i]])
            proc.stdin.write(pg.screenshot(type="jpeg", quality=92))
            if i % 300 == 0:
                print(f"frame {i}/{n} {time.time()-t0:.0f}s", flush=True)
        proc.stdin.close(); proc.wait()
        print("errors:", errs[:5], "ffmpeg rc", proc.returncode, flush=True)
        b.close()


if __name__ == "__main__":
    main()
