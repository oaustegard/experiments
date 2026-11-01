"""Build promo.json, the Kinescript scene for the games promo.

The edit is the same one the storyboard sets out: one 4/4 bar of the score is 66 frames
(2.2 s at 109.09 bpm), so every cut lands on a beat. Durations live in SCENES and every start
is derived from them. Captured gameplay is in clips/<game>.mp4 (make_clips.sh).

    python3 build_scene.py            # writes promo.json next to this file
"""
import json, math, random, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H, FPS = 1080, 1920, 30
C = dict(paper="#12140f", panel="#171a13", ink="#e4e5df", ink2="#c3c6bc", mute="#9a9e93",
         olive="#94a86d", oliveD="#b6c993", red="#e66b6b", hair="#e4e5df1a", bezel="#0a0b08", edge="#2c3026")
F = dict(display="fonts/Fraunces-SemiBold.ttf", bold="fonts/Fraunces-Bold.ttf", italic="fonts/Fraunces-Italic.ttf",
         body="fonts/DMSans-Regular.ttf", bodyB="fonts/DMSans-Bold.ttf", mono="fonts/JetBrainsMono-Regular.ttf")

SCENES = [
    dict(kind="hook", dur=132),
    dict(kind="phone", dur=66, shot="plumb", kicker="01 · Tilt", title="Plumb", sub="Tip the phone to drop."),
    dict(kind="phone", dur=66, shot="prism-break", kicker="01 · Tilt", title="Prism Break", sub="Glass shatters in WebGPU light."),
    dict(kind="phone", dur=66, shot="tilt-invaders", kicker="01 · Tilt", title="Tilt Invaders", sub="Motherships, on the beat."),
    dict(kind="phone", dur=66, shot="tilt-chomp", kicker="01 · Tilt", title="Tilt Chomp", sub="Four ghosts, arcade targeting."),
    dict(kind="phone", dur=66, shot="lantern-eel", kicker="01 · Tilt", title="Lantern Eel", sub="Light fades as you dive."),
    dict(kind="phone", dur=66, shot="careen", kicker="01 · Tilt", title="Careen", sub="Tilt picks where tiles fall."),
    dict(kind="phone", dur=66, shot="asteroids-3d", kicker="01 · Tilt", title="Asteroids 3D", sub="Rocks all around you."),
    dict(kind="phone", dur=132, shot="tilt-rally", landscape=True, kicker="02 · Wheel", title="Tilt Rally",
         sub="Turn the phone like a wheel. Tip it for gas and brake.", note="Terrain carved on the GPU",
         stat=("3 laps", "against three computer cars")),
    dict(kind="phone", dur=132, shot="doom", landscape=True, kicker="03 · GPU", title="g**P**u(doom)",
         sub="Doom's engine as WebGPU compute shaders.", note="One GPU thread per column",
         stat=("17", "compute kernels per game tic")),
    dict(kind="phone", dur=132, shot="lemmings", landscape=True, kicker="03 · GPU", title="Lemmings",
         sub="One small kernel per lemming state.", note="Eight hatches, one forest",
         stat=("10,000", "lemmings at once")),
    dict(kind="pair", dur=132, shots=("sounding", "scoop"), labels=("Sounding", "Scoop"), kicker="04 · Twice",
         title="Minesweeper, twice", sub="Sonar on the sea floor. Poop on the lawn."),
    dict(kind="phone", dur=132, shot="mahjong", kicker="05 · Table", title="Mahjong", sub="Hong Kong or Taiwanese rules.",
         note="Friends join by QR code"),
    dict(kind="phone", dur=99, shot="poker", kicker="05 · Table", title="Poker", sub="Hold'em, Omaha, Stud, Razz, Draw.",
         note="No server · no accounts"),
    dict(kind="pair", dur=99, shots=("asteroids", "blaster"), labels=("Asteroids", "Blaster"), kicker="06 · Origins",
         title="Where it started", sub="The first two, still on the site."),
    dict(kind="cta", dur=198),
]
GRID = ["ball-maze", "plumb", "prism-break", "tilt-invaders", "tilt-chomp", "lantern-eel", "careen", "asteroids-3d",
        "tilt-rally", "doom", "lemmings", "sounding", "scoop", "mahjong", "poker", "asteroids", "blaster"]

starts, t = [], 0
for s in SCENES:
    starts.append(t); t += s["dur"]
TOTAL = t
sec = lambda f: round(f / FPS, 4)


def clip_len(name):
    """Seconds of footage in clips/<name>.mp4, or None when the clip is missing."""
    p = HERE / "clips" / f"{name}.mp4"
    if not p.exists():
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def text(id_, s, x, y, size, font, fill=C["ink"], **kw):
    d = dict(id=id_, type="text", text=s, x=x, y=y, size=size, font=font, fill=fill)
    d.update(kw)
    return d


def caption(p, sc):
    """Kicker, title and subline, top left, entering in turn."""
    out = [
        dict(id=f"{p}-kbar", type="rect", x=72, y=128, anchor="left", w=40, h=3, fill=C["red"], enter="wipe"),
        text(f"{p}-kick", sc["kicker"].upper(), 128, 128, 44, F["mono"], C["olive"], anchor="left", align="left",
             letterSpacing=4, enter="fade"),
        text(f"{p}-title", sc["title"], 68, 222, 104, F["display"], anchor="left", align="left", emph=C["red"],
             letterSpacing=-2, enter={"name": "rise", "dur": 0.45}),
        text(f"{p}-sub", sc["sub"], 72, 292, 46, F["body"], C["ink2"], anchor="topleft", align="left", maxWidth=936,
             at=0.13, enter={"name": "rise", "dur": 0.45, "dist": 24}),
    ]
    return out


def device(p, name, w, landscape, dur_s, x, y_top, footage_speed=True, punch=True, extra=None):
    """A phone (bezel group, clipped) playing clips/<name>.mp4. x is the centre, y_top the top edge."""
    sw = w
    sh = round(w * 390 / 844) if landscape else round(w * 844 / 390)
    b, r = round(w * 0.028), round(w * 0.12)
    TW, TH = sw + 2 * b, sh + 2 * b
    ox, oy = -TW / 2, -TH / 2  # children are drawn from the group's anchor point, its centre
    n = clip_len(name)
    kids = []
    if n:
        v = dict(id=f"{p}-v", type="video", src=f"clips/{name}.mp4", x=ox + b, y=oy + b, anchor="topleft", w=sw, h=sh,
                 fit="cover", radius=r, volume=0)
        if footage_speed and n < dur_s - 0.05:
            v["speed"] = round(n / dur_s, 4)  # stretch a short capture to fill its slot
        v["dur"] = dur_s
        kids.append(v)
    else:
        kids.append(dict(id=f"{p}-missing", type="rect", x=ox + b, y=oy + b, anchor="topleft", w=sw, h=sh, radius=r, fill="#000000"))
        kids.append(text(f"{p}-missing-t", f"missing {name}", 0, 0, 48, F["mono"], C["red"]))
    if punch:
        kids.append(dict(id=f"{p}-flash", type="rect", x=ox + b, y=oy + b, anchor="topleft", w=sw, h=sh, radius=r, fill="#ffffff",
                         opacity=[[0, 0.22], [0.15, 0]], dur=0.2))
    # the island: a pill at the top edge (portrait) or the left edge (sideways)
    if landscape:
        kids.append(dict(id=f"{p}-isl", type="rect", x=ox + b + 10, y=0, anchor="left", w=round(w * 0.028),
                         h=round(w * 0.16), radius=30, fill="#000000"))
    else:
        kids.append(dict(id=f"{p}-isl", type="rect", x=0, y=oy + b + 10, anchor="top", w=round(w * 0.26),
                         h=round(w * 0.055), radius=30, fill="#000000"))
    g = dict(id=p, type="group", x=x, y=y_top + TH / 2, anchor="center", w=TW, h=TH, radius=r + b, fill=C["bezel"],
             clip=True, children=kids, shadow={"x": 0, "y": 40, "blur": 70, "color": "#000000aa"})
    if punch:
        g["scale"] = [[0, 1.035], [0.27, 1, "outCubic"]]
    if extra:
        g.update(extra)
    return g, TH


def pill(id_, s, y, at=0.5):
    return text(id_, s, W / 2, y, 44, F["mono"], C["oliveD"], at=at, enter={"name": "rise", "dur": 0.4, "dist": 16},
                box={"fill": "#171a13e0", "pad": [14, 30], "radius": 40})


def contour_layers():
    """The site's map-sheet contour lines, drifting up slowly behind everything."""
    rnd = random.Random(7)
    centres = [(200 + rnd.random() * 700, 300 + rnd.random() * 1300) for _ in range(3)]
    out = []
    for k in range(14):
        cx, cy = centres[k % 3]
        r0 = 90 + (k % 5) * 120
        noise = [rnd.random() - 0.5 for _ in range(18)]
        pts = []
        for i in range(72):
            a = i / 72 * math.tau
            rr = r0 * (1 + 0.18 * math.sin(a * 3 + k) + 0.08 * math.sin(a * 7 + k * 2) + 0.1 * noise[i % 18])
            pts.append(f"{cx + math.cos(a) * rr * 1.15:.1f} {cy + math.sin(a) * rr:.1f}")
        major = k % 5 == 0
        out.append(dict(id=f"topo{k}", type="path", d="M" + " L".join(pts) + " Z", x=0, y="=-T*7.5", anchor="topleft",
                        stroke="#94a86d29" if major else C["hair"], strokeWidth=2.2 if major else 1.4,
                        fill="transparent", allowOffscreen=True))
    return out


layers = [dict(id="paper", type="rect", x=0, y=0, anchor="topleft", w=W, h=H, fill=C["paper"])]
layers += contour_layers()

for i, sc in enumerate(SCENES):
    at, dur = sec(starts[i]), sec(sc["dur"])
    p = f"s{i:02d}"
    kids = []
    if sc["kind"] == "hook":
        kids += [
            text(f"{p}-kick", "KARTBLAD FLATØY · DISTRIKT", 72, 128, 44, F["mono"], C["olive"], anchor="left", align="left",
                 letterSpacing=4, at=0.2, enter="fade"),
            text(f"{p}-tilt", "Tilt**.**", 62, 292, 210, F["bold"], anchor="left", align="left", emph=C["red"],
                 letterSpacing=-6, at=0.2, enter={"name": "rise", "dur": 0.6, "dist": 60}),
            text(f"{p}-sub", "Your phone is the controller.", 72, 440, 64, F["italic"], C["ink2"], anchor="left",
                 align="left", at=1.3, enter={"name": "rise", "dur": 0.5, "dist": 30}),
        ]
        wob = "clamp((t-0.33)/0.67, 0, 1)"
        g, _ = device(f"{p}-ph", "ball-maze", 520, False, dur, W / 2, 640, punch=False,
                      extra={"rotate": f"=sin(t*30/13)*9*{wob}", "rotateX": f"=cos(t*30/17)*10*{wob}",
                             "enter": {"name": "rise", "dur": 0.7, "dist": 500}})
        kids.append(g)
    elif sc["kind"] == "phone":
        land = sc.get("landscape", False)
        w = 980 if land else 580
        top = 560 if land else 500
        g, h = device(f"{p}-ph", sc["shot"], w, land, dur, W / 2, top)
        kids += caption(p, sc) + [g]
        if sc.get("note"):
            kids.append(pill(f"{p}-note", sc["note"], top + h + 90 if land else 1770))
        if sc.get("stat"):
            big, lab = sc["stat"]
            kids.append(text(f"{p}-stat", big, W / 2, top + h + 330, 230, F["bold"], letterSpacing=-6, at=0.75,
                             enter={"name": "rise", "dur": 0.5, "dist": 40}))
            kids.append(text(f"{p}-statl", lab.upper(), W / 2, top + h + 520, 44, F["mono"], C["olive"], letterSpacing=3,
                             at=0.85, enter="fade"))
    elif sc["kind"] == "pair":
        kids += caption(p, sc)
        w = 410
        for j, (shot, lab) in enumerate(zip(sc["shots"], sc["labels"])):
            cx = W / 2 + (-1 if j == 0 else 1) * (w / 2 + 11 + 32)
            g, h = device(f"{p}-ph{j}", shot, w, False, dur, cx, 540, punch=False,
                          extra={"rotate": (-1 if j == 0 else 1) * 2.5, "at": 0.17 * j,
                                 "enter": {"name": "rise", "dur": 0.6, "dist": 300}})
            kids.append(g)
            kids.append(text(f"{p}-lab{j}", lab, cx, 540 + h + 55, 44, F["mono"], C["oliveD"], letterSpacing=2,
                             at=0.4 + 0.17 * j, enter="fade"))
    else:  # cta
        kids += [
            text(f"{p}-kick", "DISTRIKT · GAMES", 72, 170, 44, F["mono"], C["olive"], anchor="left", align="left",
                 letterSpacing=4, enter="fade"),
            text(f"{p}-n", "17 games**.**", 64, 290, 168, F["bold"], anchor="left", align="left", emph=C["red"],
                 letterSpacing=-5, enter={"name": "rise", "dur": 0.55, "dist": 50}),
            text(f"{p}-sub", "Every sound synthesized. Free, offline, in your browser.", 72, 392, 46, F["body"], C["ink2"],
                 anchor="topleft", align="left", maxWidth=936, at=0.3, enter="fade"),
        ]
        cols, tw, th, gap = 6, 150, 230, 14
        gx, gy = (W - (cols * tw + (cols - 1) * gap)) / 2, 600
        for k, gname in enumerate(GRID):
            r_, c_ = divmod(k, cols)
            shift = (cols - (len(GRID) - 2 * cols)) * (tw + gap) / 2 if r_ == 2 else 0
            n = clip_len(gname)
            tile = dict(id=f"{p}-t{k}", type="video" if n else "rect", x=gx + shift + c_ * (tw + gap), y=gy + r_ * (th + gap),
                        anchor="topleft", w=tw, h=th, radius=18, at=round(0.27 + k * 0.083, 3),
                        enter={"name": "pop", "dur": 0.5}, stroke=C["edge"], strokeWidth=2)
            if n:
                tile.update(src=f"clips/{gname}.mp4", fit="cover", volume=0, loop=True, trim=[round(n * 0.4, 2), round(n, 2)],
                            dur=round(dur - tile["at"], 3))
                tile.pop("stroke"); tile.pop("strokeWidth")
            else:
                tile["fill"] = "#000000"
            kids.append(tile)
        kids += [
            text(f"{p}-url", "austegard.com/games", W / 2, 1480, 64, F["bodyB"], C["paper"], at=2.0,
                 box={"fill": C["olive"], "pad": [28, 64], "radius": 999}, enter={"name": "pop", "dur": 0.5}),
            text(f"{p}-tag", "TILT · TAP · PLAY · INSTALL", W / 2, 1640, 44, F["mono"], C["mute"],
                 letterSpacing=2, at=2.2, enter="fade"),
            dict(id=f"{p}-fade", type="rect", x=0, y=0, anchor="topleft", w=W, h=H, fill=C["paper"],
                 opacity=[[0, 0], [-0.95, 0], ["end", 1, "linear"]]),
        ]
    layers.append(dict(id=p, type="group", at=at, dur=dur, x=0, y=0, anchor="topleft", w=W, h=H, children=kids))

lines = json.load(open(HERE / "vo" / "lines.json"))
audio = [dict(id="score", role="music", src="music/score.wav", at=0)]
audio += [dict(id=l["id"], role="voice", src=f"vo/{l['id']}.wav", at=l["at"], text=None) for l in lines]
for a in audio:
    a.pop("text", None)

scene = dict(canvas=dict(w=W, h=H, fps=FPS, bg=C["paper"], duration=sec(TOTAL)), layers=layers, audio=audio,
             grade=dict(vignette=0.25, grain=0.02))
(HERE / "promo.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False))
print("promo.json", sec(TOTAL), "s,", len(layers), "top-level layers,", len(audio), "sounds")
