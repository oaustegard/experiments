"""Build hier-router/video/scene.json: the hierarchical spiky ball, explained in about 75 seconds.

    python3 hier-router/video/build_scene.py

Six beats, each on its own narration line: the spiky ball; why one flat router runs out of room; the
tree of routers; the ball getting rounder as broad spikes split into narrow ones; the numbers from
RESULTS.md; and the catch (a router node that sat at chance for 19 epochs).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, MUTE, BG = "#f2f0e8", "#9299a8", "#0b0e14"
ROUTER, LEAF, HOT, CORE = "#4a90e2", "#7bd88f", "#ff8a3d", "#2a3242"
SPEAKER, RATE = "bm_fable,bm_fable,bm_lewis", 0.88

VO = [
    "This is a spiky ball: one small, frozen decision model, with a [[lora]]LoRA adapter for each kind of "
    "decision it is good at. Each spike is half a megabyte.",
    "A [[router]]router picks the spike. It reads every spike's description and the case together, in one "
    "[[seq]]sequence. At about [[full]]fifty-six spikes the descriptions fill it, and the case has no room left.",
    "So make it a [[tree]]tree. A router's options can be spikes, or other routers, and each router reads only "
    "its own children. [[dim]]Dim the lights goes to [[voice]]voice, then [[home]]smart home, then [[lights]]"
    "lights, then [[leaf]]dim. Three short reads instead of one long one.",
    "No router ever reads the whole list, so the library can keep [[grow]]growing. Broad spikes split into "
    "narrow ones: voice commands become fifty-four separate intents. The ball fills in, and gets [[round]]rounder.",
    "On sixty-two leaves, the tree [[acc]]routes within about a point of one flat router, reading fewer tokens. "
    "And a [[new]]new spike, given only its description, finds its cases more often, because it competes only "
    "with its siblings.",
    "There is a catch. Every router has to learn its own split. The [[inv]]invoice router sat at chance for "
    "nineteen epochs, then [[click]]clicked. So check every node, not just the total.",
]


def text(id, s, x, y, size=56, at=0, until=None, dur=None, fill=INK, weight=600, enter="fade", exit="fade", **kw):
    L = {"id": id, "type": "text", "text": s, "x": x, "y": y, "size": size, "fill": fill, "weight": weight,
         "at": at, "enter": enter, "exit": exit, **kw}
    if until is not None:
        L["until"] = until
    if dur is not None:
        L["dur"] = dur
    return L


def ball(id, n, length, at, until, width=10, enter="pop", core_label=True, exit="fade", spin=0):
    """A core circle with n spikes round it; spikes are one line layer repeated n times."""
    g = {"id": id, "type": "group", "x": 960, "y": 520, "at": at, "until": until, "enter": enter, "exit": exit,
         "children": [
             {"id": f"{id}-sp", "type": "line", "points": [[0, 0], [length, 0]], "stroke": LEAF, "strokeWidth": width,
              "repeat": {"count": n, "rotate": f"=i * {360 / n} - 90", "stroke": f"=hsl(100 + i * {200 / n}, 0.55, 0.62)"},
              "rotate": f"=t * {spin}"},
             {"id": f"{id}-core", "type": "circle", "r": 120, "fill": CORE, "stroke": "#5b6678", "strokeWidth": 4},
         ]}
    if core_label:
        g["children"].append({"id": f"{id}-lbl", "type": "text", "text": "Ettin-17m\nfrozen", "size": 34,
                              "fill": INK, "weight": 700, "lineHeight": 1.1})
    return g


def node(id, label, x, y, kind, at, until, w=None, size=30):
    w = w or max(130, 20 * len(label) + 44)
    return {"id": id, "type": "group", "x": x, "y": y, "at": at, "until": until, "enter": "pop", "exit": "fade",
            "children": [
                {"id": f"{id}-box", "type": "rect", "w": w, "h": 58, "radius": 14,
                 "fill": "#16202e" if kind == "router" else "#13261a",
                 "stroke": ROUTER if kind == "router" else LEAF, "strokeWidth": 3},
                {"id": f"{id}-t", "type": "text", "text": label, "size": size, "fill": INK, "weight": 600}]}


def edge(id, a, b, at, until, stroke="#3a4558", width=3, enter="draw"):
    return {"id": id, "type": "line", "points": [[a[0], a[1] + 29], [b[0], b[1] - 29]], "stroke": stroke,
            "strokeWidth": width, "at": at, "until": until, "enter": {"name": enter, "dur": 0.5}, "exit": "fade"}


def main():
    audio = []
    for k, line in enumerate(VO, 1):
        audio.append({"id": f"vo{k}", "role": "voice", "src": f"voice/vo{k}.wav", "text": line, "speaker": SPEAKER,
                      "rate": RATE, "at": 0.6 if k == 1 else f"vo{k - 1}.out+0.5"})
    L = [{"id": "bg", "type": "shader", "preset": "mesh", "opacity": 0.35}]

    # 1. the spiky ball
    L.append(text("title", "The **hierarchical** spiky ball", "50%", 104, size=72, weight=800, emph=HOT,
                  at=0.2, until="vo1:lora", enter="rise"))
    L.append(ball("ball5", 5, 330, "vo1.in", "vo2.in", width=16))
    names = ["transactions", "emails", "reviews", "support", "invoices"]
    for i, nm in enumerate(names):
        import math
        a = math.radians(i * 72 - 90)
        L.append(text(f"b5-{nm}", nm, 960 + math.cos(a) * 480, 520 + math.sin(a) * 405, size=38, fill=INK,
                      at="vo1:lora", until="vo2.in"))
    L.append(text("mb", "one spike = one LoRA adapter, 0.5 MB", "50%", 905, size=40, fill=LEAF, at="vo1:lora+1.2",
                  until="vo2.in"))

    # 2. the flat router's ceiling
    t2 = ("vo2.in", "vo3.in")
    L.append(text("h2", "One flat router reads everything at once", "50%", 140, size=60, weight=700, at=t2[0], until=t2[1]))
    L.append({"id": "seqframe", "type": "rect", "x": "50%", "y": 470, "w": 1640, "h": 120, "radius": 12,
              "stroke": MUTE, "strokeWidth": 3, "fill": "#00000000", "at": "vo2:router", "until": t2[1], "enter": "wipe",
              "exit": "fade"})
    L.append(text("seqlbl", "one sequence: 1,024 tokens", 140, 370, size=34, fill=MUTE, anchor="left",
                  at="vo2:router", until=t2[1]))
    # 56 description chips appear one by one; the case block shrinks as they arrive
    L.append({"id": "chips", "type": "rect", "x": 167, "y": 470, "w": 22, "h": 96, "radius": 4, "fill": ROUTER,
              "at": "vo2:seq+0.4", "until": t2[1], "exit": "fade",
              "repeat": {"count": 56, "x": "=i * 28.6", "opacity": "=clamp((t - i * 0.06) * 6, 0, 1)"}})
    L.append({"id": "case", "type": "rect", "x": 1775, "y": 470, "anchor": "right", "h": 96, "radius": 6, "fill": HOT,
              "w": [[0, 1590], [3.4, 0, "linear"]], "at": "vo2:seq+0.4", "until": t2[1], "exit": "fade"})
    L.append(text("caselbl", "the case", 1700, 570, size=34, fill=HOT, at="vo2:seq+0.4", until="vo2:full"))
    L.append({"id": "cnt", "type": "counter", "x": "50%", "y": 680, "size": 64, "weight": 800, "fill": INK,
              "value": {"from": 0, "to": 56, "dur": 3.4, "ease": "linear"}, "format": "{:.0f} spikes",
              "at": "vo2:seq+0.4", "until": t2[1], "enter": "fade", "exit": "fade"})
    L.append(text("noroom", "no room left for the case", "50%", 790, size=48, fill=HOT, weight=700,
                  at="vo2:full+0.6", until=t2[1], enter="pop"))

    # 3. the tree
    t3 = ("vo3.in", "vo4.in")
    L.append(text("h3", "Routers can pick spikes, or other routers", "50%", 90, size=54, weight=700,
                  at=t3[0], until=t3[1]))
    P = {"root": (960, 190), "voice": (420, 340), "business": (820, 340), "transactions": (1180, 340),
         "emails": (1450, 340), "reviews": (1680, 340), "home": (260, 500), "calendar": (500, 500),
         "music": (700, 500), "invoices": (900, 500), "security": (1120, 500), "lights": (240, 660),
         "plugs": (440, 660), "dimL": (180, 820), "off": (340, 820), "colour": (510, 820)}
    labels = {"root": "router", "voice": "voice", "business": "business", "transactions": "transactions",
              "emails": "emails", "reviews": "reviews", "home": "smart home", "calendar": "calendar",
              "music": "music", "invoices": "invoices", "security": "security", "lights": "lights",
              "plugs": "plugs", "dimL": "dim", "off": "off", "colour": "colour"}
    kinds = {k: "router" for k in ("root", "voice", "business", "home", "lights", "plugs", "invoices", "security",
                                   "calendar", "music")}
    tree_edges = [("root", "voice"), ("root", "business"), ("root", "transactions"), ("root", "emails"),
                  ("root", "reviews"), ("voice", "home"), ("voice", "calendar"), ("voice", "music"),
                  ("business", "invoices"), ("business", "security"), ("home", "lights"), ("home", "plugs"),
                  ("lights", "dimL"), ("lights", "off"), ("lights", "colour")]
    for a, b in tree_edges:
        L.append(edge(f"e-{a}-{b}", P[a], P[b], "vo3:tree+0.2", t3[1]))
    for k, (x, y) in P.items():
        L.append(node(f"n-{k}", labels[k], x, y, kinds.get(k, "leaf"), "vo3:tree", t3[1], size=32))
    L.append(text("more", "+ 13 more scenarios, 54 intents", 760, 600, size=32, fill=MUTE, at="vo3:tree+0.4",
                  until=t3[1]))
    L.append(text("legend1", "router", 1500, 560, size=32, fill=ROUTER, at="vo3:tree+0.4", until=t3[1], anchor="left"))
    L.append(text("legend2", "spike (leaf)", 1500, 610, size=32, fill=LEAF, at="vo3:tree+0.4", until=t3[1], anchor="left"))
    L.append(text("query", "“dim the lights a bit”", 1330, 790, size=46, fill=HOT, weight=700, at="vo3:dim",
                  until=t3[1], enter="pop"))
    path = [("root", "voice", "vo3:voice"), ("voice", "home", "vo3:home"), ("home", "lights", "vo3:lights"),
            ("lights", "dimL", "vo3:leaf")]
    for a, b, cue in path:
        L.append(edge(f"hot-{a}-{b}", P[a], P[b], cue, t3[1], stroke=HOT, width=8))
    L.append({"id": "hotleaf", "type": "circle", "x": P["dimL"][0], "y": P["dimL"][1], "r": 46, "stroke": HOT,
              "strokeWidth": 6, "fill": "#00000000", "at": "vo3:leaf+0.3", "until": t3[1], "enter": "pop", "exit": "fade",
              "emphasis": {"name": "pulse", "loop": True}})

    # 4. the ball gets rounder
    t4 = ("vo4.in", "vo5.in")
    L.append(text("h4", "Broad spikes split into narrow ones", "50%", 110, size=60, weight=700, at=t4[0], until=t4[1]))
    L.append(ball("ball5b", 5, 330, t4[0], "vo4:grow+0.8", width=16, exit="fade", core_label=False))
    L.append(ball("ball69", 69, 230, "vo4:grow+0.4", t4[1], width=7, enter={"name": "pop", "dur": 0.8}, spin=6))
    L.append(text("c5", "5 broad spikes", 560, 905, size=44, fill=MUTE, at=t4[0], until="vo4:grow+0.8"))
    L.append(text("c69", "69 narrow spikes, up to 4 routers deep", "50%", 905, size=44, fill=LEAF,
                  at="vo4:grow+1.0", until=t4[1], enter="pop"))

    # 5. the numbers
    t5 = ("vo5.in", "vo6.in")
    L.append(text("h5", "62 leaves, held-out test cases", "50%", 150, size=56, weight=700, at=t5[0], until=t5[1]))
    cards = [("acc", "vo5:acc", "routing accuracy", "0.839", "tree", "0.843", "flat"),
             ("tok", "vo5:acc+1.2", "tokens read per decision", "610", "tree", "1,198", "flat"),
             ("new", "vo5:new", "new spike, description only", "0.62", "tree", "0.51", "flat")]
    for j, (cid, cue, title, a, al, b, bl) in enumerate(cards):
        x = 400 + j * 560
        L.append({"id": f"card-{cid}", "type": "group", "x": x, "y": 520, "at": cue, "until": t5[1],
                  "enter": "rise", "exit": "fade", "children": [
                      {"id": f"card-{cid}-bg", "type": "rect", "w": 500, "h": 420, "radius": 24, "fill": "#151b26",
                       "stroke": "#2b3546", "strokeWidth": 2},
                      {"id": f"card-{cid}-h", "type": "text", "text": title, "y": -150, "size": 34, "fill": MUTE,
                       "maxWidth": 440},
                      {"id": f"card-{cid}-a", "type": "text", "text": a, "x": -120, "y": 0, "size": 76, "weight": 800,
                       "fill": LEAF},
                      {"id": f"card-{cid}-al", "type": "text", "text": al, "x": -120, "y": 80, "size": 34, "fill": LEAF},
                      {"id": f"card-{cid}-b", "type": "text", "text": b, "x": 120, "y": 0, "size": 76, "weight": 800,
                       "fill": MUTE},
                      {"id": f"card-{cid}-bl", "type": "text", "text": bl, "x": 120, "y": 80, "size": 34, "fill": MUTE}]})

    # 6. the catch
    t6 = ("vo6.in", None)
    L.append(text("h6", "Every router node has to learn", "50%", 150, size=60, weight=700, at=t6[0], until="end"))
    # validation loss of the invoice node's own adapter, per epoch (RESULTS.md round 4b)
    kl = [1.144, 1.096, 1.087, 1.087, 1.097, 1.094, 1.08, 1.082, 1.083, 1.088, 1.108, 1.12, 1.115, 1.114, 1.092,
          1.09, 1.055, 1.065, 1.081, 0.571, 0.457, 0.479, 0.227, 0.187, 0.118, 0.033, 0.147, 0.128, 0.111, 0.123]
    pts = [[i * 40, -k * 300] for i, k in enumerate(kl)]
    L.append({"id": "axis", "type": "line", "x": 380, "y": 760, "points": [[0, -380], [0, 0], [1200, 0]],
              "stroke": MUTE, "strokeWidth": 3, "at": "vo6.in+0.8", "until": "end", "enter": "draw", "exit": "fade"})
    L.append({"id": "curve", "type": "line", "x": 380, "y": 760, "points": pts, "stroke": HOT, "strokeWidth": 7,
              "at": "vo6:inv+0.3", "until": "end", "enter": {"name": "draw", "dur": 2.6, "ease": "linear"},
              "exit": "fade"})
    L.append(text("ylbl", "validation loss", 400, 360, size=30, fill=MUTE, anchor="left", at="vo6.in+0.8", until="end"))
    L.append(text("xlbl", "epochs →", 1580, 800, size=30, fill=MUTE, anchor="right", at="vo6.in+0.8", until="end"))
    L.append(text("chance", "chance: 19 epochs", 760, 390, size=38, fill=INK, at="vo6:inv+1.0", until="end"))
    L.append(text("clicked", "then it learned", 1330, 600, size=42, fill=HOT, weight=700, at="vo6:click", until="end",
                  enter="pop"))
    L.append(text("url", "austegard.com/experiments · hier-router", "50%", 905, size=36, fill=MUTE,
                  at="vo6:click+1.5", until="end"))

    L.append({"id": "cc", "type": "captions", "voice": "vo1", "mode": "karaoke", "y": "95%", "size": 38})
    caps = [{"id": f"cc{k}", "type": "captions", "voice": f"vo{k}", "mode": "karaoke", "y": "90%", "size": 40,
             "active": HOT} for k in range(1, 7)]
    L = [x for x in L if x["id"] != "cc"] + caps
    L.append({"id": "hold", "type": "rect", "w": 2, "h": 2, "fill": "#00000000", "at": "vo6.out", "dur": 2.5})
    scene = {"canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": BG, "duration": "auto"},
             "layers": L, "audio": audio, "markers": {}}
    del scene["markers"]
    (HERE / "scene.json").write_text(json.dumps(scene, indent=1))
    print(len(L), "layers")


if __name__ == "__main__":
    main()
