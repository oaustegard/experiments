"""Compile the gallery's scenes and SVGs to minified WGSL and write web/shaders.js.

    python3 web/build.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "examples"))
from forge.minify import minify   # noqa: E402
from forge.svg import read_svg     # noqa: E402
import blobs                       # noqa: E402
import sunrise                     # noqa: E402

ITEMS = [
    ("Sunrise", "Python scene, animated", sunrise.build, None),
    ("Blobs", "Python scene, SDF-only: smooth union, subtract, onion, glow", blobs.build, None),
    ("Lion", "Twemoji 1f981 (SVG, CC-BY 4.0)", None, "tests/svg/real/tw_1f981.svg"),
    ("Rust", "Simple Icons (SVG, CC0)", None, "tests/svg/real/si_rust.svg"),
    ("Settings", "Feather icon, strokes only (SVG, MIT)", None, "tests/svg/real/fe_settings.svg"),
    ("Gradients", "test SVG: bbox, userSpace, radial, gradientTransform", None, "tests/svg/gradients.svg"),
    ("Transforms", "test SVG: use, skew, non-uniform scale, CSS classes", None, "tests/svg/transforms.svg"),
    ("Paths", "test SVG: cubic, quadratic, arcs, evenodd, caps", None, "tests/svg/paths.svg"),
]


def main():
    out = []
    for title, note, build, svg in ITEMS:
        if build:
            scene = build()
            src_bytes, src_kind = None, "python"
        else:
            text = (HERE / svg).read_text()
            scene = read_svg(text)
            src_bytes, src_kind = len(text.encode()), "svg"
        wgsl = scene.compile()
        mini = minify(wgsl)
        vx, vy, vw, vh = scene.viewbox
        out.append({"title": title, "note": note, "kind": src_kind, "src_bytes": src_bytes,
                    "wgsl_bytes": len(wgsl), "min_bytes": len(mini), "aspect": vw / vh, "wgsl": mini})
    js = "export const SHADERS = " + json.dumps(out, indent=1) + ";\n"
    (HERE / "web" / "shaders.js").write_text(js)
    print(f"wrote {len(out)} shaders, {len(js)} bytes")


if __name__ == "__main__":
    main()
