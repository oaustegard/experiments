#!/usr/bin/env python3
"""Inline-SVG Pareto charts (nDCG@10 vs bytes/doc) for the blog post, site palette.

Writes chart_<corpus>.svg. Every arm in results_<corpus>.json except the seed and
nested diagnostics is drawn; three series are highlighted and connected.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER, INK, MUTED, RULE = "#f5f0e8", "#1e2a4a", "#8b8577", "#d5cfc3"
CORAL, BLUE, GREY = "#cf6853", "#2a78d6", "#b9b2a3"
W, H, L, R, T, B = 720, 470, 60, 700, 20, 408
XMIN, XMAX = math.log2(24), math.log2(4400)
FLOOR = {"scifact": 0.70, "nfcorpus": 0.25}   # arms below are listed in the figure caption, not drawn

SERIES = [  # (label, colour, shape, arm names in byte order)
    ("Google's truncation, float32", INK, "diamond", ["fp32 d=128", "fp32 d=256", "fp32 d=512", "fp32 d=768"]),
    ("remex centered, all 768 dims", CORAL, "circle",
     ["remex centered 1-bit d=768", "remex centered 2-bit d=768", "remex centered 3-bit d=768", "remex centered 4-bit d=768"]),
    ("remex centered, first 512 dims", BLUE, "square",
     ["remex centered 1-bit d=512", "remex centered 2-bit d=512", "remex centered 3-bit d=512", "remex centered 4-bit d=512"]),
]
LABELS = {  # arm -> (text, dx, dy, anchor)
    "fp32 d=128": ("128 dims, float32: Google's 6×", 10, 4, "start"),
    "fp32 d=768": ("768 dims", 0, -14, "middle"),
    "fp32 d=512": ("512 dims", 0, 26, "middle"),
    "fp32 d=256": ("256 dims", 0, 24, "middle"),
    "remex centered 4-bit d=768": ("4-bit, 768 dims", 0, -14, "middle"),
    "remex centered 2-bit d=512": ("2-bit, 512 dims", 6, -14, "middle"),
    "remex centered 1-bit d=768": ("1-bit, 768 dims", -2, -13, "end"),
    "ST binary d=768 (asym)": ("ST binary, 1-bit", 9, 17, "start"),
}


def fmt_b(b):
    return f"{b:.0f} B" if b < 1024 else f"{b/1024:.1f} KB"


def svg(corpus, title_corpus):
    rows = [r for r in json.load(open(HERE / f"results_{corpus}.json"))["rows"]
            if r["family"] not in ("seed", "nested") and r["ndcg10"] >= FLOOR[corpus]]
    by = {r["arm"]: r for r in rows}
    ys = [r["ndcg10"] for r in rows]
    step = 0.05 if max(ys) - min(ys) > 0.15 else 0.02
    y0 = math.floor(min(ys) / step) * step; y1 = math.ceil(max(ys) / step) * step + step / 2
    X = lambda b: L + (math.log2(b) - XMIN) / (XMAX - XMIN) * (R - L)
    Y = lambda v: B - (v - y0) / (y1 - y0) * (B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="width:100%;height:auto;display:block" role="img" aria-labelledby="p-{corpus}-t p-{corpus}-d" font-family="sans-serif" font-size="12">',
         f'<title id="p-{corpus}-t">nDCG@10 on {title_corpus} against bytes per document for every EmbeddingGemma 2 index measured</title>',
         f'<desc id="p-{corpus}-d">Google\'s float32 truncation to 128 dimensions scores {by["fp32 d=128"]["ndcg10"]:.3f} at 512 bytes; remex centered 4-bit on all 768 dimensions scores {by["remex centered 4-bit d=768"]["ndcg10"]:.3f} at 384 bytes against {by["fp32 d=768"]["ndcg10"]:.3f} for uncompressed float32 at 3,072 bytes.</desc>',
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']
    v = y0
    while v <= y1 + 1e-9:
        o.append(f'<line x1="{L}" x2="{R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{RULE}"/><text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end" fill="{MUTED}">{v:.2f}</text>')
        v += step
    for b in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        o.append(f'<line x1="{X(b):.1f}" x2="{X(b):.1f}" y1="{T}" y2="{B}" stroke="{RULE}"/><text x="{X(b):.1f}" y="{B+18}" text-anchor="middle" fill="{MUTED}">{fmt_b(b).replace(".0 KB"," KB")}</text>')
    o.append(f'<text x="{(L+R)/2}" y="{H-12}" text-anchor="middle" fill="{INK}" font-weight="600">bytes per document (log scale)</text>')
    o.append(f'<text transform="translate(16 {(T+B)/2}) rotate(-90)" text-anchor="middle" fill="{INK}" font-weight="600">nDCG@10, {title_corpus}</text>')
    # Pareto frontier over every candidate arm
    front, best = [], -1
    for r in sorted(rows, key=lambda r: (r["bytes_per_doc"], -r["ndcg10"])):
        if r["ndcg10"] > best:
            front.append(r); best = r["ndcg10"]
    o.append('<path d="' + " ".join(f'{"M" if i == 0 else "L"}{X(r["bytes_per_doc"]):.1f},{Y(r["ndcg10"]):.1f}' for i, r in enumerate(front))
             + f'" fill="none" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 5" opacity="0.8"/>')
    hl = {a for _, _, _, arms in SERIES for a in arms}
    ref = Y(by["fp32 d=768"]["ndcg10"])
    o.append(f'<line x1="{L}" x2="{R}" y1="{ref:.1f}" y2="{ref:.1f}" stroke="{INK}" stroke-width="1" opacity="0.45"/>')

    def mark(shape, x, y, c, tip, r=6):
        t = f"<title>{tip}</title>"
        if shape == "circle":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" stroke="{PAPER}" stroke-width="2">{t}</circle>'
        if shape == "square":
            return f'<rect x="{x-r+1:.1f}" y="{y-r+1:.1f}" width="{2*r-2}" height="{2*r-2}" fill="{c}" stroke="{PAPER}" stroke-width="2">{t}</rect>'
        if shape == "diamond":
            return f'<path d="M{x:.1f},{y-r-1:.1f} L{x+r+1:.1f},{y:.1f} L{x:.1f},{y+r+1:.1f} L{x-r-1:.1f},{y:.1f} Z" fill="{c}" stroke="{PAPER}" stroke-width="2">{t}</path>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" stroke="{PAPER}" stroke-width="1.5">{t}</circle>'

    tip = lambda r: f'{r["arm"]}: {r["ndcg10"]:.3f} nDCG@10, {fmt_b(r["bytes_per_doc"])}/doc'
    for r in rows:                                     # background arms
        if r["arm"] not in hl:
            o.append(mark("dot", X(r["bytes_per_doc"]), Y(r["ndcg10"]), GREY, tip(r), r=4))
    for _, c, shape, arms in SERIES:
        pts = [by[a] for a in arms]
        o.append('<path d="' + " ".join(f'{"M" if i == 0 else "L"}{X(p["bytes_per_doc"]):.1f},{Y(p["ndcg10"]):.1f}' for i, p in enumerate(pts))
                 + f'" fill="none" stroke="{c}" stroke-width="2" opacity="0.7"/>')
        o += [mark(shape, X(p["bytes_per_doc"]), Y(p["ndcg10"]), c, tip(p)) for p in pts]
    for a, (txt, dx, dy, anc) in LABELS.items():
        p = by[a]
        o.append(f'<text x="{X(p["bytes_per_doc"])+dx:.1f}" y="{Y(p["ndcg10"])+dy:.1f}" text-anchor="{anc}" fill="{INK}">{txt}</text>')
    lx, ly = 470, B - 86                               # legend, lower right
    for i, (lab, c, shape, _) in enumerate(SERIES):
        o.append(mark(shape, lx, ly + 20 * i, c, lab) + f'<text x="{lx+12}" y="{ly+4+20*i}" fill="{INK}">{lab}</text>')
    o.append(mark("dot", lx, ly + 60, GREY, "other arms", r=4) + f'<text x="{lx+12}" y="{ly+64}" fill="{INK}">every other arm (hover)</text>')
    o.append(f'<line x1="{lx-7}" x2="{lx+7}" y1="{ly+80}" y2="{ly+80}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 5"/><text x="{lx+12}" y="{ly+84}" fill="{INK}">Pareto frontier</text>')
    o.append(f'<line x1="{lx-7}" x2="{lx+7}" y1="{ly-20}" y2="{ly-20}" stroke="{INK}" stroke-width="1" opacity="0.45"/><text x="{lx+12}" y="{ly-16}" fill="{INK}">uncompressed, 768 dims</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    for c, name in (("scifact", "SciFact"), ("nfcorpus", "NFCorpus")):
        (HERE / f"chart_{c}.svg").write_text(svg(c, name))
        print("wrote", HERE / f"chart_{c}.svg")
