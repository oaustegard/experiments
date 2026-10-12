"""Make every legible painted clock face show the time the narration states over it.

Gemini paints clock hands at whatever time it likes (or none). For each still whose dial is legible,
erase the painted hands by inpainting dark pixels inside the dial, then draw hands at the stated
time, antialiased at 4x. The dial is an ellipse (rx, ry) so a face seen at an angle still works.
Writes stills/<shot>_dial.png and points chosen.json at it.

  python3 fix_dials.py          -> fixed stills + /tmp check sheet
"""
import json, math
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
# shot: (source still, cx, cy, rx, ry, time, erase painted hands?, hand colour)
# OLD: angles (degrees clockwise from 12) of painted hands too low-contrast for the threshold, erased along a line
OLD = {"kitchen_clock": [(-8, 0.7), (205, 0.62)], "morbier": [(60, 0.75), (-30, 0.5)], "hall": [(200, 0.6)]}
NIGHT = {"hall": (820, 60, 930, 660)}
DIALS = {
 "regulator":     ("stills/regulator_1.png",     661, 108, 96, 96, "10:05", False, (24, 30, 46)),
 "kitchen_clock": ("stills/kitchen_clock_1.png", 416, 433, 78, 78, "11:13", True, (22, 20, 18)),
 "morbier":       ("stills/morbier_0.png",       412, 95, 38, 72, "11:00", True, (28, 22, 18)),
 "hall":          ("stills/hall_0.png",          939, 111, 37, 41, "11:02", True, (28, 22, 18)),
 "atelier_night": ("stills/atelier_night_0.png", 822, 44, 42, 42, "10:00", True, (24, 22, 20)),
}

def fix(shot, src, cx, cy, rx, ry, hhmm, erase, col):
    img = cv2.imread(str(HERE / src))
    if erase:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        inside = ((xx - cx) / (rx * 0.74)) ** 2 + ((yy - cy) / (ry * 0.74)) ** 2 <= 1
        bg = cv2.medianBlur(g.astype(np.uint8), 21).astype(np.float32)
        mask = (inside & (g < bg - 18)).astype(np.uint8) * 255
        for deg, ln in OLD.get(shot, []):
            a = math.radians(deg)
            cv2.line(mask, (cx, cy), (int(cx + rx * ln * math.sin(a)), int(cy - ry * ln * math.cos(a))), 255, max(3, int((rx + ry) * 0.06)))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    if shot in NIGHT:  # a doorway that reads as daylight: darken it with a soft-edged mask
        x0, y0, x1, y1 = NIGHT[shot]; m = np.zeros(img.shape[:2], np.float32); m[y0:y1, x0:x1] = 1
        m = cv2.GaussianBlur(m, (0, 0), 18)[..., None]
        img = (img * (1 - 0.55 * m)).astype(np.uint8)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    S = 4
    ov = Image.new("RGBA", (im.width * S, im.height * S), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    h, m = map(int, hhmm.split(":"))
    def pt(ang, frac):  # angle clockwise from 12, on the dial ellipse
        return ((cx + rx * frac * math.sin(ang)) * S, (cy - ry * frac * math.cos(ang)) * S)
    r = (rx + ry) / 2
    for ang, ln, wd, tail in (((h % 12 + m / 60) / 12 * 2 * math.pi, 0.5, 0.07, 0.12), (m / 60 * 2 * math.pi, 0.78, 0.04, 0.16)):
        a, b = pt(ang + math.pi, tail), pt(ang, ln)
        d.line([a, b], fill=col + (255,), width=max(2, int(r * wd * S)))
        bx, by = b; d.ellipse([bx - r * wd * S * 0.5, by - r * wd * S * 0.5, bx + r * wd * S * 0.5, by + r * wd * S * 0.5], fill=col + (255,))
    hub = r * 0.06 * S
    d.ellipse([cx * S - hub, cy * S - hub, cx * S + hub, cy * S + hub], fill=col + (255,))
    ov = ov.resize(im.size, Image.LANCZOS)
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    out = HERE / f"stills/{shot}_dial.png"; im.save(out)
    return out, im

if __name__ == "__main__":
    ch = json.load(open(HERE / "chosen.json")); tiles = []
    for shot, spec in DIALS.items():
        out, im = fix(shot, *spec)
        ch[shot] = str(out.relative_to(HERE))
        cx, cy, rx, ry = spec[1:5]; R = int(max(rx, ry) * 1.5)
        tiles.append(im.crop((max(0, cx - R), max(0, cy - R), cx + R, cy + R)).resize((300, 300)))
    json.dump(ch, open(HERE / "chosen.json", "w"), indent=1)
    sh = Image.new("RGB", (300 * len(tiles), 300))
    for i, t in enumerate(tiles): sh.paste(t, (300 * i, 0))
    sh.save("/tmp/claude-0/dials_fixed.png"); print("fixed", list(DIALS))
