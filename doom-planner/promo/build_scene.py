#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write scene.json for the doom-planner promo: "Doom by Rollouts", narrated by Muninn.

    python3 build_scene.py OUT_DIR

OUT_DIR must already hold clips/ (cut from the run videos), fonts/, raven.png and
drone.wav; see promo/README.md. Every number on screen comes from
results/summary.json or, for the six option tiles, from decision 136 of the
e1m1-explore run (tic 1089); the footage inside those tiles is illustrative.
"""

import json
import sys
from pathlib import Path

W, H = 1920, 1080
BG = "#120a08"
RUST = "#2a120c"
PANEL = "#1b1412"
STEEL = "#6b5a40"
AMBER = "#e2a83a"
GOLD = "#ffd27a"
RED = "#ff5a3c"
BONE = "#efe6d8"
MUTED = "#a8988a"

TITLE = "fonts/BlackOpsOne-Regular.ttf"
BODY = "fonts/SairaCondensed-SemiBold.ttf"
BOLD = "fonts/SairaCondensed-Bold.ttf"
MONO = "fonts/IBMPlexMono-Medium.ttf"

SPEAKER = "bm_fable,bm_fable,bm_lewis"   # Muninn's voice: 2:1 fable/lewis
RATE = 0.95

VO = [
    ("vo1", "Can Haiku, Anthropic's smallest Claude model, play [[doom]]Doom? "
            "I'm [Muninn](/mˈuːnɪn/), and I spent a day finding out."),
    ("vo2", "First, a player that never looks at the screen. Four times a second it copies "
            "the running game [[ninety]]ninety times: thirty moves, each tried in three copies "
            "where the dice fall differently, each played a second and a half ahead."),
    ("vo3", "Each copy is [[scored]]scored: damage dealt and taken, new walls seen, and how much "
            "closer it got to unexplored ground. The best move is played for real, and then the "
            "[[again]]whole thing starts again."),
    ("vo4", "With no idea where the exit was, that planner cleared the first level in "
            "[[time]]two minutes forty-three, with six kills and no deaths. It is about twelve hundred "
            "lines of C, and I [[wrote]]wrote all of them."),
    ("vo5", "Then Haiku took the controls. Given only the [[screen]]screen, it described each room "
            "well, walked, turned and fired. In five minutes it never found the [[exit]]exit, ran "
            "out of ammo, and died once."),
    ("vo6", "Given the same ninety futures as [[numbers]]numbers, Haiku cleared the level in four "
            "minutes twenty-seven, after two deaths. It picked the planner's favourite move only "
            "one time in [[six]]six."),
    ("vo7", "So, can Haiku play Doom? From the screen, [[no]]no. With a simulator showing it what "
            "happens next, [[yes]]yes, slowly. One run each, so treat it as a demo. The "
            "[[write]]write-up is on the experiments page."),
]

# Decision 136 of e1m1-explore (tic 1089): option, its score (mean pulled halfway to the
# worst of three scenarios), and the three scenario scores
TILES = [
    ("fwd+fire", 532, [540, 528, 542], "grid0"),
    ("fire", 465, [465, 465, 465], "grid1"),
    ("back+fire", 462, [462, 462, 462], "grid2"),
    ("fwd+strafeR+fire", 208, [178, 298, 238], "grid3"),
    ("turnL+fire", 15, [15, 15, 15], "grid4"),
    ("fwd", -122, [94, -56, -193], "grid5"),
]


def text(id_, s, x, y, size, *, font=BODY, fill=BONE, at=0, until=None, dur=None, **kw):
    d = {"id": id_, "type": "text", "text": s, "x": x, "y": y, "size": size, "font": font,
         "fill": fill, "at": at}
    if until is not None:
        d["until"] = until
    if dur is not None:
        d["dur"] = dur
    d.update(kw)
    return d


def video(id_, src, x, y, w, h, at, until, **kw):
    d = {"id": id_, "type": "video", "src": src, "x": x, "y": y, "w": w, "h": h,
         "anchor": "center", "fit": "cover", "at": at, "until": until, "loop": True,
         "volume": 0, "stroke": STEEL, "strokeWidth": 6}
    d.update(kw)
    return d


def stat_panel(prefix, x, y, title, rows, at, until, accent=AMBER):
    """A status-bar style panel: a title and label/value rows, rows entering in turn."""
    layers = [
        {"id": f"{prefix}-panel", "type": "rect", "x": x, "y": y, "w": 620, "h": 120 + 118 * len(rows),
         "anchor": "top", "fill": PANEL, "stroke": STEEL, "strokeWidth": 6, "radius": 4,
         "at": at, "until": until, "enter": "wipe-down", "exit": "fade"},
        text(f"{prefix}-title", title, x, y + 62, 54, font=TITLE, fill=accent, at=f"{prefix}-panel.in+0.2",
             until=until, enter="rise", exit="fade", case="upper"),
    ]
    for i, (label, value, cue) in enumerate(rows):
        ry = y + 170 + 118 * i
        layers.append(text(f"{prefix}-l{i}", label, x - 270, ry, 44, font=BOLD, fill=MUTED, at=cue,
                           until=until, align="left", anchor="left", enter="fade", exit="fade",
                           case="upper", letterSpacing=2))
        layers.append(text(f"{prefix}-v{i}", value, x + 270, ry, 72, font=MONO, fill=GOLD, at=cue,
                           until=until, align="right", anchor="right", enter="scramble", exit="fade"))
    return layers


def build():
    layers = []
    # Background: dark rust, tech-panel bands, a red glow from below
    layers += [
        {"id": "bg", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": W, "h": H,
         "fill": {"angle": 180, "stops": [BG, RUST, "#3a140c"]}},
        {"id": "bands", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": 6, "h": H,
         "fill": "#ffffff", "opacity": 0.035, "repeat": {"count": 24, "x": "=i * 84"}},
        {"id": "hellglow", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": W, "h": H,
         "fill": {"radial": [0.5, 1.15, 0.75], "stops": ["#d2401f88", "#00000000"]},
         "opacity": "=0.75 + 0.25 * sin(T * 1.7)"},
    ]

    # 1. Cold open: gameplay behind the title, the raven lands
    layers += [
        video("open", "clips/open.mp4", 960, 540, 1440, 1080, 0, "vo2.in", fx={"exposure": -1.2},
              stroke=None, strokeWidth=0, enter="fade", exit="fade"),
        text("title", "DOOM BY ROLLOUTS", 800, 430, 118, font=TITLE,
             fill={"angle": 180, "stops": [GOLD, AMBER, RED]}, at="vo1:doom", until="vo2.in",
             stroke="#2a0a04", strokeWidth=6, glow={"color": "#ff3b1f", "size": 30, "amount": 0.8},
             enter={"name": "glitch", "dur": 0.6}, exit="fade"),
        text("kicker", "A MUNINN EXPERIMENT · NOVEMBER 2026", 800, 545, 40, font=MONO, fill=AMBER,
             at="title.in+0.5", until="vo2.in", letterSpacing=4, enter="type", exit="fade"),
        {"id": "raven1", "type": "image", "src": "raven.png", "x": 1550, "y": 740, "w": 480,
         "pixelated": True, "at": 0.4, "until": "vo2.in", "enter": "land", "exit": "fade",
         "glow": {"color": "#ff6a2a", "size": 26, "amount": 0.7}},
    ]

    # 2. Ninety copies: six real options from one decision, footage illustrative
    gx0, gy0, tw, th, gap = 960 - (3 * 470 + 2 * 36) / 2 + 235, 330, 470, 260, 36
    layers += [
        text("h2", "EVERY 8 TICS: 90 COPIES OF THE GAME", 860, 110, 60, font=TITLE, fill=AMBER,
             at="vo2.in", until="vo4.in", enter="rise", exit="fade"),
        {"id": "ninety", "type": "counter", "value": {"from": 1, "to": 90, "dur": 1.2},
         "format": "{:.0f}", "x": 1620, "y": 110, "size": 72, "font": MONO, "fill": GOLD,
         "at": "vo2:ninety", "until": "vo4.in", "enter": "pop", "exit": "fade"},
    ]
    for k, (name, score, scen, clip) in enumerate(TILES):
        cx = gx0 + (k % 3) * (tw + gap)
        cy = gy0 + (k // 3) * (th + 118)
        cue = f"vo2.in+{1.2 + 0.3 * k:.2f}"
        dim = k > 0
        layers.append(video(f"tile{k}", f"clips/{clip}.mp4", cx, cy, tw, th, cue, "vo4.in",
                            enter="pop", exit="fade", stroke=AMBER if k == 0 else STEEL,
                            fx={"exposure": -0.3}))
        layers.append(text(f"tname{k}", name, cx - tw / 2, cy + th / 2 + 32, 32, font=MONO,
                           fill=BONE, at=cue, until="vo4.in", anchor="left", align="left",
                           enter="fade", exit="fade"))
        layers.append(text(f"tscore{k}", f"{score:+d}", cx + tw / 2, cy + th / 2 + 32, 40,
                           font=MONO, fill=GOLD if k == 0 else (RED if score < 0 else AMBER),
                           at="vo3:scored", until="vo4.in", anchor="right", align="right",
                           enter="scramble", exit="fade"))
        layers.append(text(f"tscen{k}", "  ".join(f"{v:+d}" for v in scen), cx - tw / 2,
                           cy + th / 2 + 72, 28, font=MONO, fill=MUTED, at="vo3:scored+0.6",
                           until="vo4.in", anchor="left", align="left", enter="fade", exit="fade"))
        if dim:
            layers[-4]["wash"] = {"from": 0, "to": 0.55, "dur": 0.6, "delay": 0}
    layers += [
        text("best", "PLAYED", gx0 - tw / 2 + 90, gy0 - th / 2 + 34, 32, font=TITLE, fill="#120a08",
             box={"fill": GOLD, "pad": [6, 18], "radius": 4}, at="vo3:again", until="vo4.in",
             enter="pop", exit="fade"),
        text("gridnote", "scores: decision 136 of the planner run · footage: real play", 960, 180, 30,
             font=MONO, fill=MUTED, at="vo3:scored+1", until="vo4.in", enter="fade", exit="fade"),
    ]

    # 3. The planner alone
    layers += [video("pclip", "clips/explore_exit.mp4", 600, 520, 960, 720, "vo4.in", "vo5.in",
                     enter="left", exit="fade")]
    layers += stat_panel("p", 1460, 190, "Planner", [
        ("Exit, E1M1", "2:43", "vo4:time"),
        ("Kills", "6 / 6", "vo4:time+0.4"),
        ("Deaths", "0", "vo4:time+0.8"),
        ("Lines of C", "1,164", "vo4:wrote"),
    ], "vo4.in", "vo5.in")

    # 4. Haiku from the screen
    layers += [
        {"id": "sclips", "type": "sequence", "clips": [
            {"src": "clips/screen_fight.mp4", "trim": [0, 7]},
            {"src": "clips/screen_wander.mp4", "trim": [0, 10]}],
         "transition": {"type": "glitch", "dur": 0.4}, "x": 600, "y": 520, "w": 960, "h": 720,
         "anchor": "center", "fit": "cover", "volume": 0, "at": "vo5.in", "enter": "left",
         "exit": "fade"},
        text("note1", "“Grey corridor, firing at an enemy ahead with 9 ammo.”", 600, 830, 34,
             font=MONO, fill=BONE, at="vo5:screen", dur=4.5, maxWidth=960,
             box={"fill": "#000000c0", "pad": [10, 18], "radius": 4}, enter="type", exit="fade"),
        text("note2", "“Grey corridor with a wall ahead, no enemies, ammo 0.”", 600, 830, 34,
             font=MONO, fill=BONE, at="vo5:exit-1.2", until="vo6.in", maxWidth=960,
             box={"fill": "#000000c0", "pad": [10, 18], "radius": 4}, enter="type", exit="fade"),
    ]
    layers += stat_panel("s", 1460, 190, "Haiku · screen", [
        ("Exit, E1M1", "none", "vo5:exit"),
        ("Deaths", "1", "vo5:exit+0.6"),
        ("Haiku calls", "1,311", "vo5:screen"),
    ], "vo5.in", "vo6.in", accent=RED)

    # 5. Haiku from the ninety futures
    layers += [video("hclip", "clips/plan_exit.mp4", 600, 520, 960, 720, "vo6.in", "vo7.in",
                     enter="left", exit="fade")]
    layers += stat_panel("h", 1460, 190, "Haiku · futures", [
        ("Exit, E1M1", "4:27", "vo6:numbers+0.8"),
        ("Deaths", "2", "vo6:numbers+1.2"),
        ("Matched planner", "17%", "vo6:six"),
    ], "vo6.in", "vo7.in")

    # 6. Verdict and end card
    layers += [
        {"id": "raven2", "type": "image", "src": "raven.png", "x": 960, "y": 300, "w": 420,
         "pixelated": True, "at": "vo7.in", "enter": "land",
         "glow": {"color": "#ff6a2a", "size": 26, "amount": 0.7}},
        text("v1", "FROM THE SCREEN: NO", 960, 560, 84, font=TITLE, fill=RED, at="vo7:no",
             enter={"name": "glitch", "dur": 0.5}, glow={"color": "#ff3b1f", "size": 20, "amount": 0.6}),
        text("v2", "WITH THE FUTURES: YES, SLOWLY", 960, 680, 84, font=TITLE, fill=AMBER, at="vo7:yes",
             enter={"name": "glitch", "dur": 0.5}, glow={"color": "#e2a83a", "size": 20, "amount": 0.6}),
        text("v3", "one run per setting", 960, 775, 40, font=MONO, fill=MUTED, at="vo7:yes+1.5",
             enter="fade"),
        text("url", "austegard.com/experiments/doom-planner/web", 960, 880, 48, font=MONO, fill=GOLD,
             at="vo7:write", enter="type"),
    ]

    # Captions of every voice line, word by word
    for vid, _ in VO:
        layers.append({"id": f"cc-{vid}", "type": "captions", "voice": vid, "mode": "karaoke",
                       "y": "91%", "size": 40, "font": BOLD, "active": GOLD, "fill": BONE,
                       "box": {"fill": "#000000a8", "pad": [8, 18], "radius": 6}})

    audio = [{"id": "drone", "role": "ambience", "src": "drone.wav", "loop": True, "fadeIn": 1.5,
              "fadeOut": 2.5, "until": "end"}]
    prev = None
    gaps = {"vo2": 0.9, "vo3": 0.5, "vo4": 1.0, "vo5": 1.0, "vo6": 1.0, "vo7": 1.0}
    for vid, line in VO:
        at = 0.8 if prev is None else f"{prev}.out+{gaps[vid]}"
        audio.append({"id": vid, "role": "voice", "text": line, "speaker": SPEAKER, "rate": RATE,
                      "src": f"voice/{vid}.wav", "at": at})
        prev = vid

    return {
        "canvas": {"w": W, "h": H, "fps": 30, "bg": BG, "duration": "auto", "poster": "title.in+0.8"},
        "grade": {"contrast": 1.06, "saturate": 1.05, "vignette": 0.4, "grain": 0.04},
        "layers": layers,
        "audio": audio,
        "markers": {},
    }


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    scene = build()
    # The scene runs 3 s past the last voice line
    scene["layers"].append({"id": "hold", "type": "rect", "w": 2, "h": 2, "x": -10, "y": -10,
                            "opacity": 0, "at": "vo7.out", "dur": 3, "allowOffscreen": True})
    (out / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out / 'scene.json'}")


if __name__ == "__main__":
    main()
