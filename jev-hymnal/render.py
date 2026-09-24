"""Render a run to an MP4: title card, 32 animated bars with Jev's probabilities, results card.

python3 render.py runs/v2-sampled-3.json out.mp4
Frames are drawn with PIL and piped to ffmpeg (imageio-ffmpeg's bundled binary) with the synth audio.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from hymnal import ARC, DRUMS, BASS, KEYS, LEAD, PROGRESSIONS, N_BARS, section, energy
from synth import events, render_audio, write_wav, BAR, STEP, SR

W, H, FPS = 1280, 720, 30
TITLE_S, TAIL_S, CARD_S = 3.0, 2.0, 6.0
BG, FG, DIM, GRID = (14, 17, 22), (230, 232, 236), (120, 128, 140), (38, 44, 54)
COL = {"drums": (242, 169, 59), "bass": (79, 195, 247), "keys": (156, 204, 101), "pad": (126, 87, 194),
       "lead": (255, 111, 145), "progression": (255, 213, 79)}
SEC = {"intro": (96, 125, 139), "verse": (79, 195, 247), "build": (255, 167, 38), "drop": (239, 83, 80),
       "breakdown": (126, 87, 194), "finale": (239, 83, 80), "outro": (96, 125, 139)}
F = "/usr/share/fonts/truetype/dejavu/"
def font(sz, bold=False, mono=False):
    return ImageFont.truetype(F + ("DejaVuSansMono.ttf" if mono else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)
f12, f14, f16, f18, f22, f28, f40, f56 = (font(12), font(14), font(16), font(18, True), font(22, True),
                                         font(28, True), font(40, True), font(56, True))
m12 = font(12, mono=True)
CATALOG = {"drums": DRUMS, "bass": BASS, "keys": KEYS, "lead": LEAD, "progression": PROGRESSIONS}


def mix(c, k, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * k) for i in range(3))


def title_card(d, t, lat):
    k = min(1, t / 0.8)
    d.text((80, 200), "Jev plays from the hymnal", font=f56, fill=mix(FG, k))
    lines = ["A 32-bar piece in A minor, chosen one bar at a time.",
             "Each bar, one Jev call answers five Choice questions: which drum, bass, keys and melody",
             "pattern plays next, and at each 4-bar phrase, which chord progression.",
             "Every option is a vetted pattern from a fixed catalog, so nothing can come out broken.",
             f"jev-1.13.0 · median decision {np.median(lat):.2f} s · a bar lasts 2.00 s"]
    for i, s in enumerate(lines):
        d.text((82, 300 + i * 34), s, font=f18 if i == 4 else f16, fill=mix(COL["progression"] if i == 4 else DIM, k))


def board(d, x, y, w, layer, rec, prog_rec):
    src = prog_rec if layer == "progression" else rec
    probs = src["probs"][layer]
    chosen = src["progression"] if layer == "progression" else src["choice"][layer]
    cat = CATALOG[layer]
    names = list(cat)
    c = COL["pad"] if layer == "keys" and chosen == "pad" else COL[layer]
    d.rounded_rectangle((x, y, x + w, y + 84), 8, fill=(22, 26, 33), outline=GRID)
    d.text((x + 12, y + 8), layer.upper(), font=f12, fill=DIM)
    d.text((x + 12, y + 24), chosen, font=f28, fill=c)
    d.text((x + 12, y + 60), f"No. {names.index(chosen) + 1}", font=m12, fill=DIM)
    desc = cat[chosen][0]
    if layer == "progression":
        desc = " ".join(cat[chosen][1])
    d.text((x + 70, y + 60), desc[:26] + ("…" if len(desc) > 26 else ""), font=f12, fill=DIM)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
    bx, bw = x + 250, w - 262
    for i, (k, p) in enumerate(top):
        yy = y + 10 + i * 18
        hit = k == chosen
        d.text((bx, yy), k[:10], font=m12, fill=FG if hit else DIM)
        d.rectangle((bx + 80, yy + 3, bx + 80 + int((bw - 130) * p), yy + 13), fill=c if hit else mix(c, 0.3))
        d.text((bx + bw - 44, yy), f"{p * 100:3.0f}%", font=m12, fill=FG if hit else DIM)
    if chosen != max(probs, key=probs.get):
        nx = x + 12 + d.textlength(chosen, font=f28) + 10
        d.text((nx, y + 38), "sampled", font=f12, fill=c)


DRUM_ROWS = [("kick", ("kick",)), ("snare", ("snare",)), ("hats", ("hat", "ohat", "shaker")), ("crash", ("crash",))]


def grid(d, x, y, w, bar_ev, pos):
    cw = (w - 60) / 16
    for r, (label, voices) in enumerate(DRUM_ROWS):
        yy = y + r * 26
        d.text((x, yy + 4), label, font=f12, fill=DIM)
        for s in range(16):
            cx = x + 60 + s * cw
            on = any(e["step"] == s and e["voice"] in voices for e in bar_ev)
            age = pos - s
            if on:
                k = 1.0 if 0 <= age < 1 else (0.55 if age < 0 else 0.35)
                col = mix(COL["drums"], k, (22, 26, 33))
            else:
                col = (30, 35, 43) if s % 4 else (40, 46, 56)
            d.rounded_rectangle((cx + 2, yy + 2, cx + cw - 2, yy + 22), 3, fill=col)
    px = x + 60 + pos * cw
    d.line((px, y - 4, px, y + 4 * 26), fill=FG, width=2)


def roll(d, x, y, w, h, bar_ev, pos):
    d.rounded_rectangle((x, y, x + w, y + h), 8, fill=(18, 21, 27), outline=GRID)
    pitched = [e for e in bar_ev if e["midi"] is not None]
    lo, hi = 24, 96
    cw = (w - 20) / 16
    for e in pitched:
        v = e["voice"]
        c = COL["pad"] if v == "pad" else COL[v]
        x0 = x + 10 + e["step"] * cw
        x1 = x0 + max(cw * e["dur"] / STEP - 2, 3)
        yy = y + h - 8 - (e["midi"] - lo) / (hi - lo) * (h - 16)
        active = e["step"] <= pos < e["step"] + e["dur"] / STEP
        d.rounded_rectangle((x0, yy - 3, min(x1, x + w - 6), yy + 3), 2, fill=c if active else mix(c, 0.35, (18, 21, 27)))
    px = x + 10 + pos * cw
    d.line((px, y + 4, px, y + h - 4), fill=mix(FG, 0.6), width=1)
    for i, (lab, key) in enumerate([("bass", "bass"), ("keys", "keys"), ("pad", "pad"), ("lead", "lead")]):
        d.text((x + 12 + i * 60, y + 6), lab, font=f12, fill=COL[key])


def timeline(d, log, mbar, frac, lat):
    x0, y0, w = 40, 596, W - 80
    cw = w / N_BARS
    for a, b, name, _, target in ARC:
        xa, xb = x0 + (a - 1) * cw, x0 + b * cw
        d.rectangle((xa + 1, y0 + 74, xb - 1, y0 + 80), fill=SEC[name])
        if b - a >= 2:
            d.text((xa + 3, y0 + 84), name, font=f12, fill=SEC[name])
        ty = y0 + 70 - target / 3 * 60
        d.line((xa, ty, xb, ty), fill=mix(FG, 0.45), width=1)
    for i, r in enumerate(log):
        bx = x0 + i * cw
        if i + 1 <= mbar:
            e = energy(r["choice"]); hh = e / 3 * 60
            c = SEC[r["section"]]
            d.rectangle((bx + 3, y0 + 70 - hh, bx + cw - 3, y0 + 70), fill=c if i + 1 == mbar else mix(c, 0.55))
        lt = r.get("latency", 0)
        if i + 1 <= mbar + 1:
            d.rectangle((bx + 3, y0 - 2, bx + 3 + (cw - 6) * lt / BAR, y0 + 2), fill=COL["progression"])
    px = x0 + (mbar - 1 + frac) * cw
    d.line((px, y0 - 6, px, y0 + 82), fill=FG, width=2)
    d.text((x0, y0 - 20), "yellow: Jev's decision time per bar, to scale against the 2 s bar  ·  bars: energy of the chosen patterns; line: the arc's target",
           font=f12, fill=DIM)


def frame_main(d, log, ev_by_bar, tm, lat, kick_flash):
    bi = min(int(tm // BAR), N_BARS - 1)
    frac = min((tm - bi * BAR) / BAR, 0.999)
    pos = frac * 16
    rec = log[bi]
    ph = bi - bi % 4
    prog_rec = log[ph]
    name, intent, _, i, n = section(bi + 1)
    d.text((40, 18), f"BAR {bi + 1:02d}/{N_BARS}", font=f40, fill=FG)
    d.text((300, 22), name.upper(), font=f22, fill=SEC[name])
    d.text((300, 52), intent, font=f14, fill=DIM)
    chord = rec["choice"]["harmony"]
    d.text((1090, 12), chord, font=f56, fill=COL["progression"])
    chords = PROGRESSIONS[prog_rec["progression"]][1]
    for k, ch in enumerate(chords):
        d.text((1000 + k * 62, 80), ch, font=f18, fill=COL["progression"] if k == bi % 4 else DIM)
    for k, layer in enumerate(["progression", "drums", "bass", "keys", "lead"]):
        board(d, 40, 112 + k * 92, 590, layer, rec, prog_rec)
    grid(d, 660, 120, 580, ev_by_bar[bi + 1], pos)
    roll(d, 660, 238, 580, 330, ev_by_bar[bi + 1], pos)
    if kick_flash > 0:
        d.ellipse((960 - 14, 44 - 14, 960 + 14, 44 + 14), fill=mix(COL["drums"], kick_flash))
    nxt = log[bi + 1] if bi + 1 < N_BARS else None
    if nxt:
        t_dec = nxt["latency"]
        if frac * BAR < t_dec:
            s = f"asking Jev for bar {bi + 2}…"
        else:
            s = f"bar {bi + 2} decided in {t_dec:.2f} s"
        d.text((W - 40 - d.textlength(s, font=f14), 572), s, font=f14, fill=COL["progression"])
    timeline(d, log, bi + 1, frac, lat)


def card(d, t, metrics):
    k = min(1, t / 0.8)
    d.text((80, 90), "What the run shows", font=f40, fill=mix(FG, k))
    rows = metrics
    for i, (a, b) in enumerate(rows):
        d.text((82, 180 + i * 60), a, font=f18, fill=mix(FG, k))
        d.text((82, 204 + i * 60), b, font=f14, fill=mix(DIM, k))


def main(run_path, out_path, card_rows):
    log = json.loads(Path(run_path).read_text())
    lat = [r["latency"] for r in log]
    ev = events(log)
    ev_by_bar = {b: [e for e in ev if e["bar"] == b] for b in range(1, N_BARS + 1)}
    kicks = sorted(e["t"] for e in ev if e["voice"] == "kick")
    music_s = N_BARS * BAR + TAIL_S
    total = TITLE_S + music_s + CARD_S
    audio = render_audio(ev, music_s)
    pre = np.zeros((int(TITLE_S * SR), 2), dtype=np.float32)
    post = np.zeros((int(CARD_S * SR), 2), dtype=np.float32)
    wav = Path(out_path).with_suffix(".wav")
    write_wav(wav, np.concatenate([pre, audio[: int(music_s * SR)], post]))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium",
                          "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                          "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    ki = 0
    for fi in range(int(total * FPS)):
        t = fi / FPS
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        if t < TITLE_S:
            title_card(d, t, lat)
        elif t < TITLE_S + N_BARS * BAR + TAIL_S:
            tm = min(t - TITLE_S, N_BARS * BAR - 1e-3)
            while ki < len(kicks) and kicks[ki] <= t - TITLE_S:
                ki += 1
            last = kicks[ki - 1] if ki else -9
            flash = max(0.0, 1 - (t - TITLE_S - last) / 0.25)
            frame_main(d, log, ev_by_bar, tm, lat, flash)
        else:
            card(d, t - TITLE_S - music_s, card_rows)
        p.stdin.write(img.tobytes())
    p.stdin.close(); p.wait()
    wav.unlink()
    return out_path


if __name__ == "__main__":
    m = json.loads((Path(__file__).parent / "results" / "metrics-v2.json").read_text())
    run = Path(sys.argv[1]).stem
    r, rnd = m[run], m["random_summary"]
    rows = [
        (f"Follows the arc: energy vs target r = {r['arc_r']:.2f}",
         f"random picks from the same catalog: r = {rnd['arc_r_mean']:.2f} (mean of {rnd['n']} runs)"),
        (f"The hook played only where the arc asked for it: {r['hook_in_drop_or_finale']} bars",
         "the drop and finale intents name the hook; Jev never played it anywhere else"),
        (f"Decisions in time: median {r['latency_p50']:.2f} s, max {r['latency_max']:.2f} s",
         "each bar lasts 2.00 s, so bar n+1 is chosen while bar n plays"),
        ("A chord per bar collapsed to an Am / E vamp",
         "first version, a chord per bar: 3-4 of 7 chords, mostly Am and E; 4-bar progressions from the catalog use all 7"),
        ("Rendered offline from a recorded run",
         "Jev's probabilities and latencies are real; audio is numpy synthesis"),
    ]
    main(sys.argv[1], sys.argv[2], rows)
