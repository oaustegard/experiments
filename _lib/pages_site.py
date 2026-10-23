"""Build the GitHub Pages site for this repo from pages.json.

  python3 _lib/pages_site.py [OUT]     (default OUT: _site)

Copies each listed file, or each listed directory's web files, to the same
relative path under OUT, so a page keeps working with its relative links. Writes
OUT/index.html (the listing) and OUT/.nojekyll. Exits 1 when a listed path does
not exist or an entry has neither `path` nor `url`, so a stale manifest fails CI
instead of shipping a dead link.
"""
import html, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = {".html", ".htm", ".js", ".mjs", ".css", ".json", ".png", ".jpg", ".jpeg", ".gif", ".svg",
       ".webp", ".avif", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".wasm", ".mp3", ".mp4", ".webm", ".txt", ".bin"}
MAX_FILE = 50 * 1024 * 1024  # GitHub Pages rejects files over 100 MB; stay well under


def copy_file(src, out):
    """Copy one file; an HTML page with no charset declaration gets one, or browsers read UTF-8 as Windows-1252."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".html", ".htm"):
        text = src.read_text(encoding="utf-8")
        if "charset" not in text[:2048].lower():
            text = '<meta charset="utf-8">\n' + text
        out.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, out)


def copy_dir(src, dst):
    n = 0
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix.lower() in WEB and not any(p.startswith((".", "_", "__")) for p in f.relative_to(src).parts):
            if f.stat().st_size > MAX_FILE:
                raise SystemExit(f"pages: {f.relative_to(ROOT)} is over {MAX_FILE >> 20} MB")
            copy_file(f, dst / f.relative_to(src)); n += 1
    return n


def build(out):
    manifest = json.loads((ROOT / "pages.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    errors, rows = [], []
    for e in manifest["pages"]:
        title, exp = e.get("title", "?"), e.get("experiment", "")
        if "url" in e:
            rows.append((e, [e["url"]], True)); continue
        paths = e.get("path")
        if not paths:
            errors.append(f"{title}: needs `path` or `url`"); continue
        paths = [paths] if isinstance(paths, str) else paths
        links = []
        for rel in paths:
            src = ROOT / rel
            if not src.exists():
                errors.append(f"{title}: {rel} does not exist"); continue
            if src.is_dir():
                if not (src / "index.html").exists():
                    errors.append(f"{title}: directory {rel} has no index.html"); continue
                copy_dir(src, out / rel); links.append(rel.rstrip("/") + "/")
            else:
                copy_file(src, out / rel); links.append(rel)
        if links:
            rows.append((e, links, False))
    if errors:
        print("pages: manifest errors\n  " + "\n  ".join(errors), file=sys.stderr)
        raise SystemExit(1)
    (out / ".nojekyll").write_text("")
    (out / "index.html").write_text(index_html(rows), encoding="utf-8")
    files = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"pages: {len(rows)} entries, {files} files, {size / 1e6:.1f} MB -> {out}")


def index_html(rows):
    esc = html.escape
    def item(e, links, ext):
        more = ""
        if len(links) > 1:
            more = '<p class="more">' + " · ".join(f'<a href="{esc(l)}">{esc(Path(l).stem)}</a>' for l in links) + "</p>"
        exp = esc(e.get("experiment", ""))
        return (f'<li><a href="{esc(links[0])}">{esc(e["title"])}</a>{" <span class=ext>↗</span>" if ext else ""}'
                f'<p>{esc(e.get("blurb", ""))}</p>{more}'
                f'<small><a href="https://github.com/oaustegard/experiments/tree/main/{exp}">{exp}</a></small></li>')
    items = "\n".join(item(*r) for r in rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiments</title>
<meta name="description" content="Interactive pages from oaustegard/experiments.">
<style>
:root {{ --bg: #fbfaf7; --fg: #1d1f24; --dim: #5d6370; --accent: #2f5d8a; --rule: #e3e0d8; color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #14161b; --fg: #e6e4df; --dim: #9aa0ab; --accent: #8db4dc; --rule: #2a2d35; color-scheme: dark; }} }}
body {{ background: var(--bg); color: var(--fg); font: 17px/1.55 Georgia, "Iowan Old Style", serif; margin: 0; padding-block: 48px; padding-inline: 16px; }}
main {{ max-width: 680px; margin: 0 auto; }}
h1 {{ font-weight: 400; font-size: 34px; margin: 0; }}
.lede {{ color: var(--dim); margin: 8px 0 32px; }}
ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 22px; }}
li {{ border-top: 1px solid var(--rule); padding-top: 16px; }}
li > a {{ font-size: 21px; color: var(--accent); text-decoration: none; }}
li > a:hover {{ text-decoration: underline; }}
li p {{ margin: 4px 0 2px; }}
small, .ext {{ color: var(--dim); font: 13px ui-monospace, Menlo, monospace; }}
small a {{ color: var(--dim); }}
.more a {{ color: var(--accent); }}
</style>
</head>
<body>
<main>
<h1>Experiments</h1>
<p class="lede">Interactive pages from <a href="https://github.com/oaustegard/experiments" style="color:var(--accent)">oaustegard/experiments</a>. Each links its experiment's directory, where the writeup and code live. ↗ marks a page hosted elsewhere.</p>
<ul>
{items}
</ul>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site")
