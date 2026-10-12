"""Write the hosted film page: python3 film_page.py DIR  -> DIR/index.html (video and poster copied by the caller)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
TL = json.load(open(HERE / "timeline.json"))
dur = TL["duration"]; mins, secs = int(dur // 60), int(dur % 60)
pause = next(s["t0"] for s in TL["scenes"] if s["kind"] == "board")
pm, ps = int(pause // 60), int(pause % 60)
URL = "https://muninn.austegard.com/scratch/regulator/film/"
DESC = ("A watchmaker is dead in La Chaux-de-Fonds on the last night of 1899, every clock in the house keeps its own time, "
        "and one witness lies. A detective film that stops to let you solve it.")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "index.html").write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>The Regulator, the film</title>
<meta name="description" content="{DESC}">
<meta property="og:title" content="The Regulator">
<meta property="og:description" content="{DESC}">
<meta property="og:type" content="video.other">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{URL}poster.jpg">
<meta property="og:video" content="{URL}the-regulator.mp4">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#0c1420">
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,400&family=Spectral:ital,wght@0,400;1,400&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root {{ --ground: #0c1420; --ink: #e4e2da; --soft: #97a1ad; --brass: #d3a95b; color-scheme: dark; }}
body {{ margin: 0; background: var(--ground); color: var(--ink); font: 17px/1.6 "Spectral", Georgia, serif; padding-block: 40px 56px; padding-inline: 16px; }}
main {{ max-width: 1100px; margin: 0 auto; display: grid; gap: 18px; }}
h1 {{ margin: 0; font: 400 clamp(32px, 6vw, 56px)/1.05 "Bodoni Moda", Didot, Georgia, serif; }}
.sub {{ margin: 0; font-style: italic; color: var(--soft); font-size: 19px; }}
video {{ width: 100%; max-width: 100%; aspect-ratio: 16 / 9; background: #000; border: 1px solid rgba(211,169,91,.3); }}
p {{ margin: 0; max-width: 68ch; }}
.meta {{ color: var(--soft); font-size: 15px; }}
.eyebrow {{ font: 12px "IBM Plex Mono", monospace; letter-spacing: .14em; text-transform: uppercase; color: var(--brass); margin: 0; }}
a {{ color: var(--brass); }}
</style>
</head>
<body>
<main>
  <p class="eyebrow">La Chaux-de-Fonds · 31 December 1899</p>
  <h1>The Regulator</h1>
  <p class="sub">A detective film in which every clock in the house keeps its own time, and one witness lies</p>
  <video controls preload="metadata" poster="poster.jpg" src="the-regulator.mp4"></video>
  <p>Inspector Léon Matthey tells the case of Aurèle Monnier, a master watchmaker struck down in his atelier an hour and a half before the new century. Each clock and watch reading appears on screen as it is spoken. At {pm}:{ps:02d} the film puts the evidence on one card and stops to let you solve it; the drawing-room reveal follows.</p>
  <p class="meta"><a href="https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/muninn/video/2026/10/the-regulator-720p-master.mp4" download>Download the master</a> (134 MB, H.264 CRF 19, 160 kbps AAC). The player above streams a 55 MB copy of it.</p>
  <p>The same case can be worked by hand, with a bench that converts every reading to true time, at <a href="/scratch/regulator/">The Regulator</a>.</p>
  <p class="meta">{mins} min {secs:02d} s. Paintings by Gemini 3 Pro Image, two takes per shot with Claude Haiku judging each pair; seven voices designed for Gemini 3.8 Flash TTS; ticking, bells and the repeater's chime synthesized. Detectives given the narration up to the pause: Sonnet solved 4 of 4 and Haiku 4 of 16, and seven of the Haiku named the wrong man. Pipeline: <a href="https://github.com/oaustegard/experiments/tree/main/regulator/film">oaustegard/experiments</a>.</p>
</main>
</body>
</html>
""")
print(out / "index.html")
