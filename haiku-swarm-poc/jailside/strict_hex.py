"""Inside the jail: re-verify hexagon packings with a scale-aware test.

The stock evaluator tests containment with an absolute 1e-6 tolerance on an
unnormalised cross product, which a tiny outer hexagon slips through. Here
containment uses each vertex's signed distance to each outer edge (normalised),
and disjointness uses the separating-axis gap, both with a 1e-9 tolerance.
Usage: python3 strict_hex.py PROG.py [PROG.py ...]; prints one JSON line each.
"""
import importlib.util
import json
import math
import sys


def verts(x, y, s, ang):
    a = math.radians(ang)
    return [(x + s * math.cos(a + k * math.pi / 3), y + s * math.sin(a + k * math.pi / 3)) for k in range(6)]


def axes(poly):
    out = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ex, ey = x2 - x1, y2 - y1
        n = math.hypot(ex, ey)
        out.append((-ey / n, ex / n))
    return out


def sep_gap(p, q):
    """Largest separating gap over all SAT axes (positive = disjoint)."""
    best = -math.inf
    for ax in axes(p) + axes(q):
        pp = [v[0] * ax[0] + v[1] * ax[1] for v in p]
        qq = [v[0] * ax[0] + v[1] * ax[1] for v in q]
        best = max(best, min(qq) - max(pp), min(pp) - max(qq))
    return best


def worst_outside(points, outer):
    """Largest signed distance of any point outside the outer polygon (counter-clockwise)."""
    w = -math.inf
    for i in range(len(outer)):
        (x1, y1), (x2, y2) = outer[i], outer[(i + 1) % len(outer)]
        ex, ey = x2 - x1, y2 - y1
        n = math.hypot(ex, ey)
        for (px, py) in points:
            d = -((ex * (py - y1) - ey * (px - x1)) / n)   # >0 means outside
            w = max(w, d)
    return w


for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("p", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    inner, outer, side = m.hexagon_packing_11()
    inner = [tuple(map(float, r)) for r in inner]
    ox, oy, oang = map(float, outer)
    side = float(side)
    hexes = [verts(x, y, 1.0, a) for x, y, a in inner]
    out = verts(ox, oy, side, oang)
    gaps = [sep_gap(hexes[i], hexes[j]) for i in range(11) for j in range(i + 1, 11)]
    outside = worst_outside([v for h in hexes for v in h], out)
    print(json.dumps({"prog": path, "side": side, "score": (1 / side) / (1 / 3.930092),
                      "min_gap": min(gaps), "max_outside": outside,
                      "valid_strict": min(gaps) >= -1e-9 and outside <= 1e-9}))
