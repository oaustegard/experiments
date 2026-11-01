"""Fixed cost of a compiled scene: minified bytes of the smallest scenes of each kind.

    python3 tests/overhead.py [--json results/overhead.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.scene import Scene   # noqa: E402

CASES = {
    "one_circle": lambda s: s.circle(5, 5, 3),
    "one_line_path_fill": lambda s: s.path("M1 1L9 1L5 9Z"),
    "one_curve_path_fill": lambda s: s.path("M1 1Q5 9 9 1Z"),
    "one_curve_path_fill_and_stroke": lambda s: s.path("M1 1Q5 9 9 1Z", stroke="red"),
    "one_gradient_rect": lambda s: s.rect(1, 1, 8, 8, fill=__import__("forge.scene", fromlist=["linear"]).linear(0, 0, 10, 0, ["red", "blue"])),
}


def main():
    out = {}
    for name, fn in CASES.items():
        s = Scene(10, 10)
        fn(s)
        out[name] = {"wgsl": len(s.compile()), "min": len(s.compile(minify=True))}
        print(f"{name:<32} {out[name]}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
