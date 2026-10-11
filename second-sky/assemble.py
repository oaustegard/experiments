"""Inject work/sky.json into app.html.

  python3 assemble.py              -> second-sky.html (artifact body; the host adds the document shell)
  python3 assemble.py --site DIR   -> DIR/index.html, a standalone page for muninn.austegard.com/scratch/second-sky/
"""
import sys, pathlib
sky = open("work/sky.json").read()
html = open("app.html").read().replace("/*SKY*/null", sky.replace("</", "<\\/"))
if "--site" not in sys.argv:
    open("second-sky.html", "w").write(html)
    print("second-sky.html", round(len(html) / 1e6, 2), "MB"); sys.exit()

out = pathlib.Path(sys.argv[sys.argv.index("--site") + 1]); out.mkdir(parents=True, exist_ok=True)
URL = "https://muninn.austegard.com/scratch/second-sky/"
DESC = ("140 constellations of the Ilvari, a kite-flying people who never lived, drawn on the real stars "
        "with their myths, verses and an epic tour. Built by 1,843 Claude Haiku calls.")
head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{DESC}">
<meta property="og:title" content="The Second Sky">
<meta property="og:description" content="{DESC}">
<meta property="og:type" content="website">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{URL}card.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#0a0f1f">
<link rel="icon" href="/favicon.ico" sizes="any">
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
"""
links = ('<p class="sitelinks"><a href="/">Muninn</a> · <a href="/scratch/">scratch</a> · '
         '<a href="film/">the film</a> · '
         '<a href="https://github.com/oaustegard/experiments/tree/main/second-sky">how it was made</a></p>')
html = html.replace("<canvas id=\"sky\"", "</head>\n<body>\n<canvas id=\"sky\"", 1)
html = html.replace("</style>", ".sitelinks { margin: 8px 0 0; font: 12px var(--f-body); letter-spacing: .06em; color: var(--dim); pointer-events: auto; }\n"
                    ".sitelinks a { color: var(--dim); }\n.sitelinks a:hover { color: var(--cord); }\n</style>", 1)
html = html.replace("drawn on the real stars</p>", "drawn on the real stars</p>\n  " + links, 1)
(out / "index.html").write_text(head + html + "\n</body>\n</html>\n")
print(out / "index.html", round((out / "index.html").stat().st_size / 1e6, 2), "MB")
