"""Build film.html: the app with its UI hidden, film.js on top, the timeline inlined."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
app = (HERE.parent / "app.html").read_text()
sky = (HERE.parent / "work/sky.json").read_text().replace("</", "<\\/")
tl = (HERE / "timeline.json").read_text()
html = app.replace("/*SKY*/null", sky)
# untold figures recede so the one being told carries the frame
for a, b in [("'rgba(143,180,220,0.30)'", "'rgba(143,180,220,0.16)'"), ("'rgba(216,179,106,0.55)'", "'rgba(216,179,106,0.30)'")]:
    assert a in html, a
    html = html.replace(a, b)
html = html.replace("</style>", ".cartouche,.controls,.hint,.panel{display:none!important}\nbody{margin:0}\n</style>", 1)
html += f"\n<script>const TL = {tl};</script>\n<script>{(HERE / 'film.js').read_text()}</script>\n"
(HERE / "film.html").write_text("<!DOCTYPE html><html><head><meta charset='utf-8'>" + html + "</html>")
print("film.html", round(len(html) / 1e6, 2), "MB")
