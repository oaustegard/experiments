#!/usr/bin/env python3
"""Build scene.json: a narrated Kinescript explainer, ColBERT -> MaxSim -> index size -> pooling -> remex
-> the measured trade-off (from ../lateon-pool-remex). One timeline, ten voice lines in Muninn's voice.

Every section's layers start on words of its own voice line ("voK:mark") and leave when the next line
starts, so re-voicing at another rate re-times the whole video.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
BG = "#0b0e14"
INK = "#eef0f4"
DIM = "#8b93a7"
ACC = "#ff7a45"   # orange: the point being made
BLUE = "#4cc2ff"
GREEN = "#7ee787"
RED = "#ff6b6b"
PURPLE = "#b48cff"
CHIP = "#1b2130"
MONO = "IBM Plex Mono"
SPEAKER = "bm_fable,bm_fable,bm_lewis"   # Muninn's voice
RATE = 0.88
COLBERT = "[ColBERT](/kQlbˈɛə/)"
REMEX = "[remex](/ɹˈiːmɛks/)"

# ---------------------------------------------------------------------------------------------- narration
VO = [
    # 1 title
    f"Most search engines squeeze a document into one vector. [[col]]{COLBERT} keeps one for every token. "
    f"Here's how it scores, why that gets big, and how to make it [[small]]small.",
    # 2 single vs multi
    f"A single-vector model [[averages]]averages the whole text into one point. [[colb]]{COLBERT} gives each "
    f"token its own vector, a hundred and twenty-eight numbers long. So a specific word, like the D in "
    f"vitamin D, keeps its own [[say]]say in the score.",
    # 3 MaxSim
    f"To score a document, each query token looks for its most similar document token. [[take]]Take that best "
    f"match for every query token, and [[add]]add them up. That's max-sim: [[late]]late interaction, because "
    f"query and document only meet at the end.",
    # 4 cost
    f"The catch is size. [[lateon]]LateOn, a strong {COLBERT} model, keeps about two hundred and thirty "
    f"vectors per document. At four bytes a number, that's [[kb]]a hundred and eighteen kilobytes, for every "
    f"single document.",
    # 5 levers
    f"There are [[two]]two ways to shrink it. [[fewer]]Keep fewer vectors. [[bits]]Or spend fewer bits on "
    f"each one.",
    # 6 pooling
    f"Pooling merges tokens that point the same way. [[ward]]Ward clustering groups a document's tokens, and "
    f"each group becomes its [[avg]]average. [[half]]Pool by two, and half the vectors remain. Pool by four, "
    f"and a [[quarter]]quarter.",
    # 7 remex
    f"Quantizing is {REMEX}'s job. [[cone]]LateOn's vectors crowd into a narrow cone, so {REMEX} first "
    f"[[subtract]]subtracts their average. Then it [[rotate]]rotates them, and [[snap]]snaps every coordinate "
    f"to a few levels: sixteen levels at four bits. Just [[one]]two at one bit.",
    # 8 result
    f"So which wins at the same size? On SciFact, at about four kilobytes per document, [[every]]every token "
    f"at one bit scores point seven two. [[quarter]]A quarter of the tokens at four bits scores point seven "
    f"six: [[nearly]]nearly the full index, at one thirtieth of its size.",
    # 9 integrated
    f"We also let the quantizer decide how far to pool: merge until the merge error matches the coding "
    f"error. [[free]]That's free at three and four bits. [[but]]But it didn't beat plain pooling. It won on "
    f"one dataset, and [[lost]]lost on the other.",
    # 10 takeaway
    f"So, for a token index: [[pool]]pool the tokens, [[keep]]keep at least two bits, and [[center]]center "
    f"before you quantize.",
]
N = len(VO)

layers: list[dict] = []
audio: list[dict] = []


def nxt(k: int) -> str:
    """When section k's layers leave: as the next line starts."""
    return f"vo{k + 1}.in+0.05" if k < N else "end"


def hexhsl(h: float, s: float, l: float) -> str:
    import colorsys
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def add(layer: dict, k: int, exit_: str | dict | None = "fade") -> dict:
    if "dur" not in layer:
        layer.setdefault("until", nxt(k))
    if exit_ and "exit" not in layer:
        layer["exit"] = {"name": "fade", "dur": 0.3} if exit_ == "fade" else exit_
    layers.append(layer)
    return layer


def text(id_, s, x, y, k, at, size=56, fill=INK, **kw):
    return add({"id": id_, "type": "text", "text": s, "x": x, "y": y, "size": size, "fill": fill, "at": at,
                **kw}, k)


def chip_w(s: str, size: int) -> float:
    return len(s) * size * 0.56 + 40


def chips(prefix, words, cx, y, k, at, size=46, gap=22, stagger=0.06, enter="rise", fill=INK, box=CHIP):
    """A row of token chips centred on cx; returns their centre x's and widths."""
    ws = [chip_w(w, size) for w in words]
    total = sum(ws) + gap * (len(ws) - 1)
    x = cx - total / 2
    out = []
    for i, (w, wd) in enumerate(zip(words, ws)):
        c = x + wd / 2
        out.append((c, wd))
        add({"id": f"{prefix}{i}", "type": "text", "text": w, "x": round(c), "y": y, "size": size, "fill": fill,
             "box": {"fill": box, "pad": [10, 18], "radius": 12}, "at": f"{at}+{i * stagger:.2f}",
             "enter": {"name": enter, "dur": 0.4}}, k)
        x += wd + gap
    return out


# ---------------------------------------------------------------------------------------------- audio
for k, line in enumerate(VO, 1):
    audio.append({"id": f"vo{k}", "role": "voice", "text": line, "speaker": SPEAKER, "rate": RATE,
                  "at": 0.6 if k == 1 else f"vo{k - 1}.out+0.35"})
    # word-lit captions of each line, under everything else
    layers.append({"id": f"cc{k}", "type": "captions", "voice": f"vo{k}", "mode": "karaoke", "active": ACC,
                   "size": 40, "y": "91%", "maxWidth": 1500, "at": f"vo{k}.in", "until": nxt(k)})

# ---------------------------------------------------------------------------------------------- 1 title
k = 1
text("kicker", "HOW A TOKEN INDEX GETS SMALL", "50%", 330, k, "vo1.in+0.2", size=40, fill=ACC, weight=700,
     letterSpacing=6, enter={"name": "fade", "dur": 0.6})
text("title", "ColBERT  →  pooling  →  remex", "50%", 450, k, "vo1.in+0.4", size=120, weight=800,
     split="words", enter={"name": "rise", "dur": 0.6, "stagger": 0.18})
text("sub", "one vector per document, or one per token?", "50%", 590, k, "vo1:col", size=48, fill=DIM,
     enter="fade")
add({"id": "tbar", "type": "rect", "x": "50%", "y": 660, "w": 360, "h": 6, "fill": ACC, "radius": 3,
     "at": "vo1:small", "enter": "wipe"}, k)

# ---------------------------------------------------------------------------------------------- 2 vectors
k = 2
SENT = ["vitamin", "D", "raises", "bone", "density", "in", "older", "women"]
row = chips("s2c", SENT, W / 2, 250, k, "vo2.in+0.3")
text("s2h", "one vector for the whole text", "50%", 130, k, "vo2:averages", size=48, fill=DIM, enter="fade",
     until="vo2:colb")
for i, (cx, wd) in enumerate(row):  # every chip feeds one point
    add({"id": f"s2l{i}", "type": "line", "points": [[round(cx), 285], [W // 2, 640]], "stroke": "#3a4256",
         "strokeWidth": 3, "at": f"vo2:averages+{0.04 * i:.2f}", "until": "vo2:colb", "enter": "draw"}, k)
add({"id": "s2dot", "type": "circle", "x": "50%", "y": 640, "r": 34, "fill": BLUE, "at": "vo2:averages+0.6",
     "until": "vo2:colb", "enter": "pop", "glow": {"color": BLUE, "size": 20}}, k)
text("s2one", "1 point", W // 2 + 130, 640, k, "vo2:averages+0.8", size=48, fill=BLUE, until="vo2:colb",
     enter="fade")
text("s2h2", "one vector per token  ·  128 numbers each", "50%", 130, k, "vo2:colb", size=48, fill=DIM,
     enter="fade")
for i, (cx, wd) in enumerate(row):
    hue = 190 + (i * 23) % 140
    add({"id": f"s2v{i}", "type": "rect", "x": round(cx), "y": 340, "w": 46, "h": 20, "radius": 3,
         "fill": hexhsl(hue, 0.55, 0.55), "anchor": "top",
         "repeat": {"count": 14, "y": "=i * 26", "opacity": f"=0.45 + 0.55 * abs(sin(i * {1.3 + 0.37 * i:.2f}))"},
         "at": f"vo2:colb+{0.07 * i:.2f}", "enter": {"name": "wipe-down", "dur": 0.5}}, k)
cx_d = row[1][0]
add({"id": "s2ring", "type": "rect", "x": round(cx_d), "y": 520, "w": 110, "h": 560, "radius": 18,
     "stroke": ACC, "strokeWidth": 5, "at": "vo2:say-0.2", "enter": "pop",
     "emphasis": {"name": "pulse", "at": 0.5}}, k)

# ---------------------------------------------------------------------------------------------- 3 MaxSim
k = 3
text("s3t", "MaxSim", "50%", 110, k, "vo3.in+0.2", size=84, weight=800, enter="rise")
text("s3qh", "query", 380, 220, k, "vo3.in+0.3", size=44, fill=DIM, enter="fade")
text("s3dh", "document", 1520, 200, k, "vo3.in+0.3", size=44, fill=DIM, enter="fade")
Q = ["vitamin", "D", "bone"]
DOC = ["calcium", "and", "vitamin", "D", "build", "strong", "bone"]
qy = [380, 520, 660]
dy = [290 + 85 * j for j in range(len(DOC))]
best = {0: 2, 1: 3, 2: 6}
sims = {0: "0.93", 1: "0.88", 2: "0.91"}
for i, w in enumerate(Q):
    add({"id": f"s3q{i}", "type": "text", "text": w, "x": 380, "y": qy[i], "size": 50,
         "box": {"fill": "#23324a", "pad": [10, 20], "radius": 12}, "fill": INK, "at": f"vo3.in+{0.4 + 0.1 * i:.1f}",
         "enter": "rise"}, k)
for j, w in enumerate(DOC):
    add({"id": f"s3d{j}", "type": "text", "text": w, "x": 1520, "y": dy[j], "size": 44,
         "box": {"fill": CHIP, "pad": [8, 18], "radius": 12}, "fill": INK, "at": f"vo3.in+{0.5 + 0.06 * j:.2f}",
         "enter": "rise"}, k)
x0 = 380 + chip_w("vitamin", 50) / 2 + 10
for i in range(len(Q)):
    for j in range(len(DOC)):
        x1 = 1520 - chip_w(DOC[j], 44) / 2 - 10
        add({"id": f"s3l{i}_{j}", "type": "line", "points": [[round(x0), qy[i]], [round(x1), dy[j]]],
             "stroke": "#2f3648", "strokeWidth": 2, "at": f"vo3.in+{0.9 + 0.05 * (i * 7 + j):.2f}",
             "enter": "draw"}, k)
for i in range(len(Q)):
    j = best[i]
    x1 = 1520 - chip_w(DOC[j], 44) / 2 - 10
    add({"id": f"s3b{i}", "type": "line", "points": [[round(x0), qy[i]], [round(x1), dy[j]]], "stroke": ACC,
         "strokeWidth": 7, "at": f"vo3:take+{0.25 * i:.2f}", "enter": {"name": "draw", "dur": 0.5}}, k)
    mx, my = (x0 + x1) / 2, (qy[i] + dy[j]) / 2
    text(f"s3s{i}", sims[i], round(mx), round(my) - 34, k, f"vo3:take+{0.25 * i + 0.3:.2f}", size=40, fill=ACC,
         font=MONO, weight=700, box={"fill": BG, "pad": [4, 10], "radius": 8}, enter="pop")
text("s3sum", "0.93 + 0.88 + 0.91 = 2.72", "50%", 840, k, "vo3:add", size=60, font=MONO, weight=700, fill=INK,
     enter={"name": "rise", "dur": 0.5})
text("s3late", "late interaction: query and document meet only at the end", "50%", 905, k, "vo3:late", size=40,
     fill=DIM, enter="fade")

# ---------------------------------------------------------------------------------------------- 4 cost
k = 4
text("s4h", "LateOn, one document", "50%", 220, k, "vo4:lateon", size=56, fill=DIM, enter="fade")
text("s4eq", "230 vectors  ×  128 numbers  ×  4 bytes", "50%", 400, k, "vo4:lateon+0.5", size=70, weight=700,
     split="words", enter={"name": "rise", "dur": 0.45, "stagger": 0.12})
add({"id": "s4n", "type": "counter", "value": {"from": 0, "to": 117760, "dur": 1.2}, "format": "= {:,.0f} bytes",
     "x": "50%", "y": 580, "size": 120, "weight": 800, "fill": ACC, "font": MONO, "at": "vo4:kb",
     "enter": "fade"}, k)
text("s4kb", "≈ 118 KB per document, before any compression", "50%", 730, k, "vo4:kb+1.2", size=48, fill=DIM,
     enter="fade")

# ---------------------------------------------------------------------------------------------- 5 levers
k = 5
text("s5h", "Two ways to shrink it", "50%", 170, k, "vo5:two", size=80, weight=800, enter="rise")
for i, (name, what, col, mark) in enumerate([("Pooling", "fewer vectors", BLUE, "fewer"),
                                             ("Quantizing", "fewer bits per number", PURPLE, "bits")]):
    cx = 560 if i == 0 else 1360
    add({"id": f"s5c{i}", "type": "rect", "x": cx, "y": 590, "w": 640, "h": 560, "radius": 28, "fill": "#151a26",
         "stroke": col, "strokeWidth": 4, "at": f"vo5:{mark}", "enter": "pop"}, k)
    text(f"s5n{i}", name, cx, 400, k, f"vo5:{mark}+0.15", size=72, weight=800, fill=col, enter="fade")
    text(f"s5w{i}", what, cx, 490, k, f"vo5:{mark}+0.25", size=46, fill=INK, enter="fade")
# pooling icon: 8 dots -> 2
for j in range(8):
    add({"id": f"s5p{j}", "type": "circle", "x": 380 + 46 * j, "y": 640, "r": 13, "fill": BLUE, "at": "vo5:fewer+0.4",
         "enter": "pop"}, k)
text("s5arr", "→", 560, 710, k, "vo5:fewer+0.6", size=56, fill=DIM, enter="fade")
for j in range(2):
    add({"id": f"s5q{j}", "type": "circle", "x": 520 + 80 * j, "y": 780, "r": 22, "fill": BLUE, "at": "vo5:fewer+0.8",
         "enter": "pop"}, k)
text("s5b1", "0.8137254", 1360, 640, k, "vo5:bits+0.4", size=56, font=MONO, fill=INK, enter="fade")
text("s5arr2", "→", 1360, 710, k, "vo5:bits+0.6", size=56, fill=DIM, enter="fade")
text("s5b2", "level 13 of 16", 1360, 780, k, "vo5:bits+0.8", size=56, font=MONO, fill=PURPLE, enter="fade")

# ---------------------------------------------------------------------------------------------- 6 pooling
k = 6
text("s6h", "Pooling: merge tokens that point the same way", "50%", 120, k, "vo6.in+0.2", size=56, weight=700,
     enter="rise")
rng = random.Random(7)
cents = [(620, 420), (900, 640), (1230, 400), (1380, 700), (760, 780), (1100, 520)]
cols = [BLUE, GREEN, ACC, PURPLE, "#ffd166", "#ff8fab"]
pts = []
for c, (cx, cy) in enumerate(cents):
    for _ in range(4):
        pts.append((c, cx + rng.gauss(0, 55), cy + rng.gauss(0, 45)))
for n, (c, px, py) in enumerate(pts):
    add({"id": f"s6g{n}", "type": "circle", "x": round(px), "y": round(py), "r": 14, "fill": "#6b7387",
         "at": f"vo6.in+{0.5 + 0.03 * n:.2f}", "until": "vo6:ward", "enter": "pop", "exit": None}, k, exit_=None)
    add({"id": f"s6c{n}", "type": "circle", "x": round(px), "y": round(py), "r": 14, "fill": cols[c],
         "at": f"vo6:ward+{0.02 * n:.2f}", "until": "vo6:avg", "enter": {"name": "fade", "dur": 0.25}}, k,
        exit_=None)
    cx, cy = cents[c]
    add({"id": f"s6m{n}", "type": "circle", "r": 14, "fill": cols[c], "at": "vo6:avg", "dur": 0.9,
         "x": [[0, round(px)], [0.7, cx, "inOutCubic"]], "y": [[0, round(py)], [0.7, cy, "inOutCubic"]],
         "opacity": [[0, 1], [0.6, 1], [0.9, 0]]}, k, exit_=None)
for c, (cx, cy) in enumerate(cents):
    add({"id": f"s6M{c}", "type": "circle", "x": cx, "y": cy, "r": 26, "fill": cols[c], "at": "vo6:avg+0.65",
         "enter": "pop", "outline": {"width": 4, "color": INK}}, k)
text("s6n1", "24 tokens", 230, 300, k, "vo6.in+0.6", size=48, fill=DIM, enter="fade", until="vo6:avg+0.7")
text("s6n2", "6 means", 230, 300, k, "vo6:avg+0.75", size=48, fill=ACC, weight=700, enter="fade")
text("s6pf", "pool by 2 → ½ the vectors   ·   pool by 4 → ¼", "50%", 870, k, "vo6:half", size=48, fill=INK,
     enter="fade")

# ---------------------------------------------------------------------------------------------- 7 remex
k = 7
text("s7h", "Quantizing with remex", "50%", 110, k, "vo7.in+0.2", size=64, weight=800, enter="rise")
OX, OY = 520, 560
rng = random.Random(3)
cone = [(-38 + rng.gauss(0, 7), 300 + rng.gauss(0, 18)) for _ in range(14)]   # narrow cone, up and right
mean_ang = sum(a for a, _ in cone) / len(cone)
for n, (ang, ln) in enumerate(cone):
    ex, ey = OX + ln * math.cos(math.radians(ang)), OY + ln * math.sin(math.radians(ang))
    add({"id": f"s7a{n}", "type": "arrow", "x1": OX, "y1": OY, "x2": round(ex), "y2": round(ey), "stroke": "#7d879c",
         "strokeWidth": 4, "head": 14, "at": f"vo7:cone+{0.04 * n:.2f}", "until": "vo7:subtract+0.6",
         "enter": "draw"}, k)
mx_, my_ = OX + 290 * math.cos(math.radians(mean_ang)), OY + 290 * math.sin(math.radians(mean_ang))
add({"id": "s7mu", "type": "arrow", "x1": OX, "y1": OY, "x2": round(mx_), "y2": round(my_), "stroke": ACC,
     "strokeWidth": 9, "head": 22, "at": "vo7:subtract", "until": "vo7:subtract+0.6", "enter": "draw"}, k)
text("s7cone", "all point the same way", OX, 820, k, "vo7:cone", size=44, fill=DIM, enter="fade",
     until="vo7:subtract")
text("s7sub", "subtract the average", OX, 820, k, "vo7:subtract", size=44, fill=ACC, enter="fade",
     until="vo7:rotate")
text("s7rot", "rotate", OX, 820, k, "vo7:rotate", size=44, fill=PURPLE, enter="fade")
# centred residuals, scaled up so they are visible, in a group that rotates
kids = []
for n, (ang, ln) in enumerate(cone):
    ex, ey = ln * math.cos(math.radians(ang)) - (mx_ - OX), ln * math.sin(math.radians(ang)) - (my_ - OY)
    ex, ey = ex * 4.0, ey * 4.0
    s = 230 / max(1.0, math.hypot(ex, ey)) if math.hypot(ex, ey) > 230 else 1.0
    kids.append({"id": f"r{n}", "type": "arrow", "x1": 0, "y1": 0, "x2": round(ex * s), "y2": round(ey * s),
                 "stroke": BLUE, "strokeWidth": 4, "head": 14})
add({"id": "s7res", "type": "group", "x": OX, "y": OY, "w": 10, "h": 10, "children": kids,
     "at": "vo7:subtract+0.5", "enter": {"name": "fade", "dur": 0.4},
     "rotate": {"from": 0, "to": 40, "dur": 1.2, "delay": 0.0, "ease": "inOutCubic"}}, k)
# number lines: Lloyd-Max levels for a Gaussian coordinate
LM4 = [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326]
LM1 = [0.7979]


def numline(prefix, y, levels, label, col, at):
    L, R, C, S = 1080, 1760, 1420, 118
    add({"id": f"{prefix}ax", "type": "line", "points": [[L, y], [R, y]], "stroke": "#4a5266", "strokeWidth": 3,
         "at": at, "enter": "draw"}, k)
    lv = sorted([-v for v in levels] + levels)
    for j, v in enumerate(lv):
        add({"id": f"{prefix}t{j}", "type": "rect", "x": round(C + v * S), "y": y, "w": 4, "h": 34, "fill": col,
             "at": f"{at}+{0.2 + 0.02 * j:.2f}", "enter": "pop"}, k)
    text(f"{prefix}lab", label, C, y - 70, k, f"{at}+0.1", size=44, fill=col, weight=700, enter="fade")
    rng2 = random.Random(len(levels))
    for j in range(9):
        v = max(-2.6, min(2.6, rng2.gauss(0, 1)))
        q = min(lv, key=lambda l: abs(l - v))
        add({"id": f"{prefix}d{j}", "type": "circle", "r": 11, "fill": INK, "y": y + 40,
             "x": [[0, round(C + v * S)], [0.5, round(C + v * S)], [1.0, round(C + q * S), "outBack"]],
             "at": f"{at}+0.4", "enter": "pop"}, k)


numline("s7n4", 470, LM4, "4 bits: 16 levels", PURPLE, "vo7:snap")
numline("s7n1", 760, LM1, "1 bit: 2 levels", ACC, "vo7:one")

# ---------------------------------------------------------------------------------------------- 8 result
k = 8
text("s8h", "Same size, about 4 KB per document  ·  SciFact", "50%", 120, k, "vo8.in+0.2", size=52, weight=700,
     enter="rise")
text("s8m", "nDCG@10, higher is better", "50%", 190, k, "vo8.in+0.4", size=40, fill=DIM, enter="fade")
COLS = [("every", "all 230 tokens", "1 bit each", "4.1 KB", 0.718, DIM, 380),
        ("quarter", "¼ of the tokens", "4 bits each", "3.8 KB", 0.758, ACC, 960),
        ("nearly", "full index", "32-bit floats", "118 KB", 0.763, BLUE, 1540)]
for i, (mark, l1, l2, kb, score, col, cx) in enumerate(COLS):
    at = f"vo8:{mark}"
    add({"id": f"s8c{i}", "type": "rect", "x": cx, "y": 580, "w": 500, "h": 560, "radius": 26, "fill": "#151a26",
         "stroke": col if i == 1 else "#2a3142", "strokeWidth": 5 if i == 1 else 3, "at": f"vo8.in+{0.5 + 0.15 * i:.2f}", "enter": "pop"}, k)
    text(f"s8a{i}", l1, cx, 380, k, f"{at}+0.1", size=48, weight=700, fill=INK, enter="fade")
    text(f"s8b{i}", l2, cx, 445, k, f"{at}+0.2", size=44, fill=DIM, enter="fade")
    add({"id": f"s8s{i}", "type": "counter", "value": {"from": 0.5, "to": score, "dur": 0.8}, "format": "{:.3f}",
         "x": cx, "y": 590, "size": 110, "weight": 800, "font": MONO, "fill": col, "at": f"{at}+0.3",
         "enter": "fade"}, k)
    text(f"s8k{i}", kb, cx, 720, k, f"{at}+0.5", size=56, font=MONO, fill=INK, enter="fade")
add({"id": "s8x", "type": "text", "text": "1/30 of the full index", "x": 960, "y": 790, "size": 38, "fill": ACC,
     "weight": 700, "at": "vo8:nearly+0.6", "enter": "fade"}, k)

# ---------------------------------------------------------------------------------------------- 9 integrated
k = 9
text("s9h", "Let the quantizer set the pooling?", "50%", 140, k, "vo9.in+0.2", size=68, weight=800, enter="rise")
text("s9r", "merge while merge error ≤ coding error", "50%", 240, k, "vo9.in+0.8", size=46, fill=DIM, font=MONO,
     enter="fade")
text("s9free", "✓ free at 3 and 4 bits", "50%", 360, k, "vo9:free", size=56, fill=GREEN, weight=700, enter="pop")
text("s9vs", "against uniform pooling at the same size (nDCG@10):", "50%", 480, k, "vo9:but", size=44, fill=INK,
     enter="fade")
AX = 960
add({"id": "s9ax", "type": "line", "points": [[AX, 540], [AX, 800]], "stroke": "#5a6378", "strokeWidth": 3,
     "at": "vo9:but+0.2", "enter": "draw"}, k)
add({"id": "s9b1", "type": "rect", "x": AX, "y": 600, "anchor": "left", "w": 0.013 * 30000, "h": 70, "fill": GREEN,
     "radius": 8, "at": "vo9:but+0.5", "enter": "wipe"}, k)
text("s9l1", "SciFact  +0.013", AX + 0.013 * 30000 + 30, 600, k, "vo9:but+0.8", size=46, fill=GREEN, font=MONO,
     anchor="left", enter="fade")
add({"id": "s9b2", "type": "rect", "x": AX, "y": 730, "anchor": "right", "w": 0.002 * 30000, "h": 70, "fill": RED,
     "radius": 8, "at": "vo9:lost", "enter": "wipe-left"}, k)
text("s9l2", "NFCorpus  −0.002", AX - 0.002 * 30000 - 30, 730, k, "vo9:lost+0.2", size=46, fill=RED, font=MONO,
     anchor="right", enter="fade")

# ---------------------------------------------------------------------------------------------- 10 takeaway
k = 10
text("s10h", "For a token index", "50%", 200, k, "vo10.in+0.1", size=64, fill=DIM, weight=700, enter="rise")
for i, (mark, s) in enumerate([("pool", "1   Pool the tokens"), ("keep", "2   Keep at least 2 bits"),
                               ("center", "3   Center before you quantize")]):
    text(f"s10l{i}", s, 560, 370 + 130 * i, k, f"vo10:{mark}", size=72, weight=800, fill=[BLUE, PURPLE, ACC][i],
         anchor="left", enter={"name": "rise", "dur": 0.4})
add({"id": "s10f", "type": "text", "text": "LateOn on SciFact and NFCorpus  ·  oaustegard/experiments, lateon-pool-remex",
     "x": "50%", "y": 860, "size": 36, "fill": DIM, "at": "vo10.out+0.2", "dur": 3.2, "enter": "fade",
     "until": None}, k)
layers[-1].pop("until")  # timed by dur: the scene ends with it

scene = {"canvas": {"w": W, "h": H, "fps": 30, "bg": BG, "duration": "auto", "poster": "title.in+1.2"},
         "layers": layers, "audio": audio, "mix": {"loudness": -17}}
(HERE / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False))
print(f"{len(layers)} layers, {len(audio)} voice lines -> scene.json")
