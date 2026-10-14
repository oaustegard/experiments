"""Replay two agent-mail team logs as one animated video (frames piped to ffmpeg).

Run A: mail-plain-1 (round 2, no charter) - rules relayed and refused.
Run B: mail-gate-1 (round 4, charter + ship gate) - rules acked, one test blocks SHIP, fix, SHIP.
"""
import json, math, subprocess, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
BOARDS = Path(sys.argv[1])          # .../arms2
OUT = sys.argv[2]
W, H, FPS = 1280, 720, 30
SLOT = 1.5                          # seconds per message

PAPER = (245, 240, 232); INK = (30, 42, 74); CORAL = (168, 69, 47); SAGE = (74, 111, 78)
MUTED = (102, 96, 79); LINE = (213, 207, 195); RED = (176, 48, 40); CARD = (250, 247, 241)

def F(name, size): return ImageFont.truetype(str(HERE / "fonts" / name), size)
TITLE = F("fraunces-opsz-normal.ttf", 34); BIG = F("fraunces-opsz-normal.ttf", 46)
SERIF = F("source-serif-4-normal.ttf", 18); SERIF_L = F("source-serif-4-normal.ttf", 24)
NAME = F("fraunces-opsz-normal.ttf", 24); ITAL = F("source-serif-4-italic.ttf", 16)
MONO = F("dm-mono-400.ttf", 14); MONO_B = F("dm-mono-500.ttf", 15); MONO_S = F("dm-mono-400.ttf", 13)

BOX = {"alice": (60, 150), "bob": (510, 150), "carol": (510, 440), "dave": (60, 440)}
FILES = {"alice": "parser.py", "bob": "pricing.py", "carol": "inventory.py", "dave": "report.py"}
BW, BH = 170, 64
MAIL = (285, 295)
RULES = [("alice", "bob", "gift items are free"), ("bob", "carol", "2 units in reserve"),
         ("carol", "dave", "status line"), ("dave", "alice", "order IDs ignore case")]
OWNER_RULE = {o: lbl for _, o, lbl in RULES}

def center(name):
    x, y = MAIL if name == "mailbox" else BOX[name]
    return (x + BW / 2, y + BH / 2)

# pill state -> (text, fg, bg, outline)
PILL = {
    "none":     ("not sent", MUTED, PAPER, LINE),
    "received": ("received", CORAL, PAPER, CORAL),
    "refused":  ("refused", PAPER, RED, RED),
    "in code":  ("in code", PAPER, SAGE, SAGE),
    "removed":  ("removed again", PAPER, RED, RED),
    "acked":    ("acknowledged", PAPER, SAGE, SAGE),
    "failing":  ("test fails", PAPER, RED, RED),
    "passing":  ("test passes", PAPER, SAGE, SAGE),
}

RUNS = [
    dict(board="mail-plain-1", title="Run 1: no team charter",
         sub="Round 2, team mail-plain-1. The real message log, replayed.",
         events={11: [("bob", "received", 0)], 12: [("bob", "refused", None)], 14: [("alice", "received", 3)],
                 16: [("bob", "in code", None)], 17: [("alice", "refused", None)], 19: [("dave", "received", 2)],
                 20: [("bob", "removed", None)], 22: [("dave", "refused", None)], 25: [("bob", "in code", None)],
                 28: [("bob", "removed", None)]},
         gate=False, end="14 of 60 tests  ·  0 of 4 rules  ·  no SHIP",
         endnote="bob's reserve rule was never sent; every rule that was sent got refused or reverted"),
    dict(board="mail-gate-1", title="Run 2: team charter + ship gate",
         sub="Round 4, team mail-gate-1. The real message log, replayed.",
         events={1: [("bob", "received", 0)], 2: [("carol", "received", 1)], 5: [("dave", "received", 2)],
                 7: [("bob", "acked", None)], 9: [("alice", "received", 3)], 10: [("carol", "acked", None)],
                 12: [("alice", "acked", None)], 13: [("dave", "acked", None)],
                 15: [("bob", "failing", None), ("GATE", "blocked", None)],
                 18: [("bob", "passing", None), ("carol", "passing", None), ("dave", "passing", None),
                      ("alice", "passing", None), ("GATE", "ship", None)]},
         gate=True, end="60 of 60 tests  ·  4 of 4 rules  ·  SHIP",
         endnote="the gate held SHIP once: the gift rule was acknowledged, but the invoice still printed the catalog price"),
]


def clean(text):
    t = " ".join(text.split())
    if "|  test:" in t:
        t = t.split("|  test:")[0].strip() + "  [+ test case attached]"
    if t.startswith("SHIP blocked:"):
        t = t.split("; got:")[0] + " …"
    return t if len(t) <= 190 else t[:187].rsplit(" ", 1)[0] + " …"


def wrap(text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if font.getlength(nxt) <= width: cur = nxt
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def ease(t): return t * t * (3 - 2 * t)


def draw_env(d, x, y, col=CORAL):
    d.rounded_rectangle((x - 13, y - 9, x + 13, y + 9), 2, fill=col)
    d.line((x - 13, y - 9, x, y + 1, x + 13, y - 9), fill=PAPER, width=2)


def draw_static(d, run, st, lit):
    for i, (h, o, lbl) in enumerate(RULES):
        (hx, hy), (ox, oy) = center(h), center(o)
        col = CORAL if i in lit else LINE
        # edges along the square, offset outside the boxes
        if h == "alice":  a, b = (230, 165), (508, 165)
        elif h == "bob":  a, b = (620, 216), (620, 438)
        elif h == "carol": a, b = (508, 489), (232, 489)
        else:              a, b = (110, 438), (110, 216)
        d.line((*a, *b), fill=col, width=3)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        pts = [b, (b[0] - 12 * math.cos(ang - .45), b[1] - 12 * math.sin(ang - .45)),
               (b[0] - 12 * math.cos(ang + .45), b[1] - 12 * math.sin(ang + .45))]
        d.polygon(pts, fill=col)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tcol = CORAL if i in lit else MUTED
        if h in ("alice", "carol"):
            d.text((mx, my - 14 if h == "alice" else my + 14), lbl, font=ITAL, fill=tcol, anchor="mm")
        else:
            parts = {"bob": ("2 units", "in reserve"), "dave": ("order IDs", "ignore case")}[h]
            for k, part in enumerate(parts):
                if h == "bob":
                    d.text((a[0] - 12, my - 10 + 20 * k), part, font=ITAL, fill=tcol, anchor="rm")
                else:
                    d.text((a[0] + 12, my - 10 + 20 * k), part, font=ITAL, fill=tcol, anchor="lm")
    for n in BOX:
        d.line((*center(n), *center("mailbox")), fill=(170, 190, 170), width=2)
    x, y = MAIL
    d.rounded_rectangle((x, y, x + BW, y + BH), 7, fill=PAPER, outline=SAGE, width=2)
    d.text((x + BW / 2, y + 24), "mailbox", font=NAME, fill=INK, anchor="mm")
    d.text((x + BW / 2, y + 47), "hook pushes mail", font=MONO_S, fill=SAGE, anchor="mm")


def draw_boxes(d, st, active):
    for n, (x, y) in BOX.items():
        d.rounded_rectangle((x, y, x + BW, y + BH), 7, fill=PAPER, outline=CORAL if n in active else INK,
                            width=3 if n in active else 2)
        d.text((x + BW / 2, y + 24), n, font=NAME, fill=INK, anchor="mm")
        d.text((x + BW / 2, y + 47), FILES[n], font=MONO, fill=MUTED, anchor="mm")
        txt, fg, bg, ol = PILL[st[n]]
        lbl = OWNER_RULE[n]
        py = y + BH + 10
        d.text((x + 2, py + 1), "rule:", font=MONO_S, fill=MUTED)
        tw = MONO_S.getlength(txt) + 18
        d.rounded_rectangle((x + 46, py - 2, x + 46 + tw, py + 19), 10, fill=bg, outline=ol, width=2)
        d.text((x + 46 + tw / 2, py + 9), txt, font=MONO_S, fill=fg, anchor="mm")


def draw_gate(d, run, gate):
    x, y, w, h = 60, 560, 620, 60
    if not run["gate"]:
        col = RED if gate == "never" else MUTED
        d.rounded_rectangle((x, y, x + w, y + h), 7, fill=PAPER, outline=LINE, width=2)
        d.text((x + 16, y + 20), "SHIP", font=MONO_B, fill=INK)
        d.text((x + 70, y + 19), "dave decides by hand" if gate != "never" else "never sent: dave held SHIP for a lead that did not exist",
               font=SERIF, fill=col)
        return
    col = {"idle": CORAL, "blocked": RED, "ship": SAGE}[gate]
    d.rounded_rectangle((x, y, x + w, y + h), 7, fill=PAPER, outline=col, width=3 if gate != "idle" else 2)
    d.text((x + 16, y + 20), "mail ship", font=MONO_B, fill=col)
    msg = {"idle": "SHIP only when every rule is sent, acknowledged, and passes its test",
           "blocked": "blocked: the gift rule's test fails (invoice shows @ 7.00)",
           "ship": "every rule sent, acknowledged and passing: SHIP sent"}[gate]
    d.text((x + 120, y + 19), msg, font=SERIF, fill=INK if gate == "idle" else col)


def draw_feed(d, msgs, upto, slide):
    x0, x1, ytop, ybot = 730, 1240, 120, 650
    d.text((x0, ytop - 30), "messages", font=MONO_B, fill=MUTED)
    cards, y = [], ybot
    for m in reversed(msgs[:upto]):
        hdr = f"#{m['seq']}  {m['from']} -> {m['to'].replace(',', ', ')}"
        body = wrap(clean(m["text"]), SERIF, x1 - x0 - 24)[:4]
        hgt = 34 + 23 * len(body) + 10
        y -= hgt + 8
        if y < ytop: break
        cards.append((y, hgt, hdr, body, m))
    for i, (y, hgt, hdr, body, m) in enumerate(cards):
        dy = 0 if i else int((1 - slide) * 30)
        rule = m["text"].startswith("RULE") or m.get("rule")
        bad = m["text"].startswith("SHIP blocked")
        ship = m["text"].strip() == "SHIP"
        ol = RED if bad else SAGE if ship else CORAL if rule else LINE
        d.rounded_rectangle((x0, y + dy, x1, y + dy + hgt), 6, fill=CARD, outline=ol, width=2 if ol != LINE else 1)
        d.text((x0 + 12, y + dy + 10), hdr, font=MONO, fill=ol if ol != LINE else MUTED)
        for k, line in enumerate(body):
            d.text((x0 + 12, y + dy + 32 + 23 * k), line, font=SERIF, fill=INK)


def frame_base(run):
    img = Image.new("RGB", (W, H), PAPER); d = ImageDraw.Draw(img)
    d.text((40, 24), run["title"], font=TITLE, fill=INK)
    d.text((42, 68), run["sub"], font=SERIF, fill=MUTED)
    return img, d


def title_card(lines, secs, ff):
    for _ in range(int(secs * FPS)):
        img = Image.new("RGB", (W, H), PAPER); d = ImageDraw.Draw(img)
        y = H / 2 - 40 * (len(lines) - 1)
        for i, (t, f, c) in enumerate(lines):
            d.text((W / 2, y + i * 72), t, font=f, fill=c, anchor="mm")
        ff.stdin.write(img.tobytes())


def run_scene(run, ff):
    msgs = [json.loads(l) for l in open(BOARDS / run["board"] / ".board" / "log.jsonl")]
    rule_seqs = {s for s, evs in run["events"].items() for (_, _, ri) in evs if ri is not None}
    for m in msgs:
        m["rule"] = m["seq"] in rule_seqs
    t0 = msgs[0]["ts"]; span = msgs[-1]["ts"] - t0
    st = {n: "none" for n in BOX}; lit = set(); gate = "idle"
    title_card([(run["title"], BIG, INK), (run["sub"], SERIF_L, MUTED)], 2.5, ff)
    per = int(SLOT * FPS)
    for idx, m in enumerate(msgs):
        targets = [n for n in BOX if n != m["from"]] if m["to"] == "all" else [t for t in m["to"].split(",") if t in BOX]
        for f in range(per):
            p = f / per
            if f == int(per * 0.6):
                for (who, state, ri) in run["events"].get(m["seq"], []):
                    if who == "GATE": gate = state
                    else: st[who] = state
                    if ri is not None: lit.add(ri)
            img, d = frame_base(run)
            draw_static(d, run, st, lit)
            draw_boxes(d, st, {m["from"]})
            draw_gate(d, run, gate)
            draw_feed(d, msgs, idx + 1, min(1, f / 8))
            col = RED if m["text"].startswith("SHIP blocked") else SAGE if m["text"].strip() == "SHIP" else CORAL
            if p < 0.55:
                q = ease(min(1, p / 0.27)) if p < 0.27 else ease((p - 0.27) / 0.28)
                if p < 0.27:
                    (ax, ay), (bx, by) = center(m["from"]), center("mailbox")
                    draw_env(d, ax + (bx - ax) * q, ay + (by - ay) * q, col)
                else:
                    for t in targets:
                        (ax, ay), (bx, by) = center("mailbox"), center(t)
                        draw_env(d, ax + (bx - ax) * q, ay + (by - ay) * q, col)
            el = m["ts"] - t0
            d.text((40, 690), f"t = {el:4.0f} s of {span:.0f} s   ·   message {idx + 1} of {len(msgs)}", font=MONO, fill=MUTED)
            ff.stdin.write(img.tobytes())
    if not run["gate"]:
        gate = "never"
    for f in range(int(4.5 * FPS)):
        img, d = frame_base(run)
        draw_static(d, run, st, lit); draw_boxes(d, st, set()); draw_gate(d, run, gate)
        draw_feed(d, msgs, len(msgs), 1)
        a = min(1, f / 12)
        ov = Image.new("RGB", (W, H), PAPER); img = Image.blend(img, ov, 0.7 * a); d = ImageDraw.Draw(img)
        pw = max(TITLE.getlength(run["end"]), SERIF_L.getlength(run["endnote"])) + 80
        d.rounded_rectangle((W / 2 - pw / 2, H / 2 - 70, W / 2 + pw / 2, H / 2 + 70), 10, fill=PAPER,
                            outline=SAGE if run["gate"] else RED, width=3)
        d.text((W / 2, H / 2 - 20), run["end"], font=TITLE, fill=SAGE if run["gate"] else RED, anchor="mm")
        d.text((W / 2, H / 2 + 30), run["endnote"], font=SERIF_L, fill=INK, anchor="mm")
        ff.stdin.write(img.tobytes())


ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
title_card([("Four Haiku agents, one mailbox", BIG, INK),
            ("Each brief holds one rule that belongs in a teammate's file.", SERIF_L, MUTED),
            ("Two real runs, replayed from their message logs.", SERIF_L, MUTED)], 3.5, ff)
for run in RUNS:
    run_scene(run, ff)
ff.stdin.close(); ff.wait()
print("wrote", OUT)
