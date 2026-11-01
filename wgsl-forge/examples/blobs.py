"""SDF-native shapes the SVG model cannot express: smooth unions that merge as they move,
a subtraction, an onion ring and a soft glow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge.scene import (Scene, T, Circle, Rect, cos, onion, sin, smooth_union,  # noqa: E402
                         subtract, radial, Ellipse)


def build():
    s = Scene(320, 200, background="#101418")
    s.circle(160, 100, 70, fill="#4fd1c5", opacity=0.25, soft=18)
    blob = smooth_union(Circle(130 + 25 * sin(T), 100, 34), Circle(190 - 25 * sin(T), 100, 30),
                        Circle(160, 70 + 18 * cos(T * 1.3), 20), k=22)
    s.add(blob, fill=radial(160, 100, 80, ["#a7f3d0", "#10b981", "#065f46"]), stroke="#ecfdf5",
          stroke_width=2)
    s.add(subtract(Rect(24, 24, 70, 50, rx=12), Circle(59, 49, 16 + 4 * sin(T * 2))), fill="#f59e0b")
    s.add(onion(Ellipse(262, 150, 36, 22), 4), fill="#f472b6", rotate=(T * 40, 262, 150))
    return s


if __name__ == "__main__":
    print(build().compile())
