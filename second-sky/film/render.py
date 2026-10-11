"""Render film.html frame by frame in headless Chromium and encode with ffmpeg.

  python3 render.py stills T1 T2 ...      -> stills/t<T>.png (1920x1080), for checking before a full render
  python3 render.py video [WORKERS]       -> parts/part<k>.mp4 per worker, then picture.mp4 (no audio)
"""
import asyncio, json, subprocess, sys
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
TL = json.load(open(HERE / "timeline.json"))
FPS = TL["fps"]


async def page(p):
    b = await p.chromium.launch(executable_path=CHROME, args=["--disable-gpu-vsync"])
    pg = await b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    await pg.goto((HERE / "film.html").as_uri())
    # canvas faces load on first use, so the first caption would fall back to a default serif
    await pg.evaluate("""Promise.all(['italic 27px "IM Fell English"', '50px "IM Fell English SC"', '64px "IM Fell English SC"',
        'italic 24px "IM Fell English"', '13px "IBM Plex Mono"', '15px "IM Fell English SC"'].map(f => document.fonts.load(f)))""")
    await pg.evaluate("document.fonts.ready")
    await pg.wait_for_timeout(1500)
    if errs:
        raise SystemExit(f"page errors: {errs[:3]}")
    return b, pg


async def stills(times):
    out = HERE / "stills"; out.mkdir(exist_ok=True)
    async with async_playwright() as p:
        b, pg = await page(p)
        for t in times:
            await pg.evaluate(f"filmFrame({t})")
            await pg.screenshot(path=str(out / f"t{float(t):07.2f}.png"))
        await b.close()


async def chunk(k, n0, n1):
    dst = HERE / f"parts/part{k}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", str(dst)],
                          stdin=subprocess.PIPE)
    async with async_playwright() as p:
        b, pg = await page(p)
        for n in range(n0, n1):
            await pg.evaluate(f"filmFrame({n / FPS})")
            ff.stdin.write(await pg.screenshot(type="jpeg", quality=93))
            if (n - n0) % 480 == 0:
                print(f"worker {k}: frame {n - n0}/{n1 - n0}", flush=True)
        await b.close()
    ff.stdin.close(); ff.wait()
    assert ff.returncode == 0, f"ffmpeg failed on part {k}"


async def video(workers):
    (HERE / "parts").mkdir(exist_ok=True)
    total = int(TL["duration"] * FPS)
    bounds = [round(total * i / workers) for i in range(workers + 1)]
    await asyncio.gather(*(chunk(k, bounds[k], bounds[k + 1]) for k in range(workers)))
    lst = HERE / "parts/list.txt"
    lst.write_text("".join(f"file 'part{k}.mp4'\n" for k in range(workers)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(HERE / "picture.mp4")], check=True)
    print("picture.mp4", total, "frames")


if __name__ == "__main__":
    if sys.argv[1] == "stills":
        asyncio.run(stills([float(x) for x in sys.argv[2:]]))
    else:
        asyncio.run(video(int(sys.argv[2]) if len(sys.argv) > 2 else 3))
