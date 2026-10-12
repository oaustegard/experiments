"""Inject work/case.json into app.html.

  python3 assemble.py              -> regulator.html (artifact body; the host adds the document shell)
  python3 assemble.py --site DIR   -> DIR/index.html for muninn.austegard.com/scratch/regulator/
"""
import sys, pathlib
case = open("work/case.json").read()
html = open("app.html").read().replace("/*CASE*/null", case.replace("</", "<\\/"))
if "--site" not in sys.argv:
    open("regulator.html", "w").write(html); print("regulator.html", len(html) // 1024, "KB"); sys.exit()
out = pathlib.Path(sys.argv[sys.argv.index("--site") + 1]); out.mkdir(parents=True, exist_ok=True)
URL = "https://muninn.austegard.com/scratch/regulator/"
DESC = ("A fair-play murder mystery set in a watchmaker's house in La Chaux-de-Fonds on the last night of 1899. "
        "Every clock keeps its own time and one witness lies. Proven unique by a solver; written and checked by Claude Haiku.")
head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{DESC}">
<meta property="og:title" content="The Regulator">
<meta property="og:description" content="{DESC}">
<meta property="og:type" content="website">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{URL}card.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.ico" sizes="any">
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
"""
html = html.replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1)
html = html.replace("Written by Muninn with Claude Haiku 5.5 subagents.",
    'Written by <a href="/">Muninn</a> with Claude Haiku 5.5 subagents · <a href="/scratch/">scratch</a> · '
    '<a href="https://github.com/oaustegard/experiments/tree/main/regulator">how it was made</a>.')
(out / "index.html").write_text(head + html + "\n</body>\n</html>\n")
print(out / "index.html", (out / "index.html").stat().st_size // 1024, "KB")
