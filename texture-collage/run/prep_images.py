"""Convert the Kodak test images (image-to-wgsl/data/kodak, fetched by its fetch.sh) to PPM for the Node harness."""
import sys
from pathlib import Path
from PIL import Image

here = Path(__file__).resolve().parent
src = here.parents[1] / "image-to-wgsl" / "data" / "kodak"
out = here.parent / "data"
out.mkdir(exist_ok=True)
for p in sorted(src.glob("*.png")):
    im = Image.open(p).convert("RGB")
    im.save(out / (p.stem + ".ppm"))
    print(p.stem, im.size)
