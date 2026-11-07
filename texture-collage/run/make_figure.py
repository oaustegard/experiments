"""results/figure.jpg: per row, input | regions filled with their mean colour | texture collage.
Rows: three Kodak images at K=8 from run/eval.mjs outputs (out/), plus the collage made on the Mac with
fresh 128 px textures (results/mac/kodim23_collage128.jpg)."""
from pathlib import Path
from PIL import Image, ImageDraw
d = Path(__file__).resolve().parents[1]
W, H = 384, 256
rows = []
for name, K in [("kodim23", 12), ("kodim05", 12), ("kodim20", 6)]:
    rows.append([d / f"data/{name}.ppm", d / f"out/{name}-K{K}-solid.ppm", d / f"out/{name}-K{K}-structure.ppm"])
sheet = Image.new("RGB", (3 * W, len(rows) * H + H), (11, 11, 13))
for r, files in enumerate(rows):
    for c, f in enumerate(files):
        sheet.paste(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS), (c * W, r * H))
mac = Image.open(d / "results/mac/kodim23_collage128.jpg").resize((W, H), Image.LANCZOS)
sheet.paste(Image.open(d / "out/kodim23-K12-colour.ppm").resize((W, H), Image.LANCZOS), (0, len(rows) * H))
sheet.paste(Image.open(d / "out/kodim23-K12-random.ppm").resize((W, H), Image.LANCZOS), (W, len(rows) * H))
sheet.paste(mac, (2 * W, len(rows) * H))
sheet.save(d / "results/figure.jpg", quality=86)
print("ok", sheet.size)
