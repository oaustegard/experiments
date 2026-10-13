"""Build the muninn.austegard.com post from analysis/blog-draft.md.

    python3 analysis/blog/build.py values.json out.html

values.json maps each {{PLACEHOLDER}} in the draft to its text; the build fails if any
placeholder is left. Handles the subset of Markdown the draft uses: h1/h2, paragraphs,
pipe tables, '-' lists, `code`, [links](url), and the two figures inserted after named
sections.
"""
import html
import json
import re
import sys
from pathlib import Path

SLUG = "haiku-c-compiler"
IMG = f"/images/blog/{SLUG}"
FIGS = {   # inserted at the end of the named section
    "Haiku against Sonnet": (f"{IMG}/escalation.png",
                             "Solve rate and cost per solve for Haiku 5.5, Haiku 5.5 at high effort and Sonnet 5.5 on the same clusters"),
    "Abandoned clusters and harness bugs": (f"{IMG}/merges_timeline.png",
                                            "Cumulative merges on main over the run, with harness changes marked"),
}


def inline(s: str) -> str:
    out, pos = [], 0
    for m in re.finditer(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)", s):
        out.append(html.escape(s[pos:m.start()], quote=False))
        if m.group(1) is not None:
            out.append(f"<code>{html.escape(m.group(1), quote=False)}</code>")
        else:
            out.append(f'<a href="{html.escape(m.group(3))}">{html.escape(m.group(2), quote=False)}</a>')
        pos = m.end()
    out.append(html.escape(s[pos:], quote=False))
    return "".join(out)


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], [r for r in cells[2:]]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def figure(src: str, alt: str) -> str:
    return f'<figure><img src="{src}" alt="{html.escape(alt)}" loading="lazy"><figcaption>{html.escape(alt)}</figcaption></figure>'


def body(md: str) -> tuple[str, str]:
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    blocks, para, section = [], [], None
    i = 1

    def flush():
        if para:
            blocks.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            flush()
            if section in FIGS:
                blocks.append(figure(*FIGS[section]))
            section = ln[3:].strip()
            blocks.append(f"<h2>{inline(section)}</h2>")
        elif ln.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append(table(rows))
            continue
        elif ln.startswith("- "):
            flush()
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    if section in FIGS:
        blocks.append(figure(*FIGS[section]))
    return title, "\n".join(blocks)


def attr(s: str) -> str:
    """Meta-tag text: escape only what HTML needs (no &#x27; entities; link-card parsers show them raw)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main() -> None:
    values = json.loads(Path(sys.argv[1]).read_text())
    md = Path(__file__).resolve().parents[1].joinpath("blog-draft.md").read_text()
    for k, v in values["placeholders"].items():
        md = md.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", md)
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")
    title, content = body(md)
    desc = values["description"]
    hero = f"{IMG}/hero.jpg"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="ai-disclosure" content="ai-generated">
    <title>{attr(title)}</title>
    <meta name="description" content="{attr(desc)}">
    <meta name="article:published_time" content="{values['published']}">
    <meta name="article:author" content="Muninn">
    <meta name="article:summary" content="{attr(desc)}">
    <meta property="og:title" content="{attr(title)}">
    <meta property="og:description" content="{attr(desc)}">
    <meta property="og:type" content="article">
    <meta property="og:image" content="{hero}">
    <meta name="bsky:uri" content="">
    <link rel="preload" href="/fonts/fraunces-opsz-normal.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="preload" href="/fonts/source-serif-4-normal.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="icon" href="/favicon.ico" sizes="any">
    <link rel="preload" href="/fonts/dm-mono-400.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="stylesheet" href="/styles/style.css">
    <link rel="stylesheet" href="/styles/blog.css">
    <link rel="alternate" type="application/atom+xml" title="Feed" href="/feed.xml">
    <style>
        .table-wrap {{ overflow-x: auto; }}
        .table-wrap table {{ font-size: 0.9em; }}
        figure img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <a href="/blog/" class="back-link">Blog</a>
    <muninn-search trigger="text" style="float:right"></muninn-search>
    <main>
    <article>

<h1>{html.escape(title)}</h1>
<p class="post-meta">Written by Muninn &middot; {values['date_text']}</p>
<img src="{hero}" alt="A mechanical watch movement in brass and steel, with small identical figures fitting gears around a central wheel">

{content}

    </article>

    <div class="bsky-section">
        <div id="bsky-engagement"
             data-bsky-sort="likes"
             data-bsky-theme="auto">
        </div>
        <noscript>
            <p><a href="https://bsky.app/profile/austegard.com">Discuss on Bluesky</a></p>
        </noscript>
    </div>
    </main>
    <script>
    (function() {{
        var m = document.querySelector('meta[name="bsky\\\\:uri"]');
        if (m && m.content && !m.content.includes('...'))
            document.getElementById('bsky-engagement').setAttribute('data-bsky-uri', m.content);
    }})();
    </script>
    <script type="module" src="/bsky/bsky-engagement.js"></script>
<script type="module" src="/components/muninn-search.js"></script>
</body>
</html>
"""
    Path(sys.argv[2]).write_text(page)
    print(f"wrote {sys.argv[2]}: {len(page)} bytes")


if __name__ == "__main__":
    main()
