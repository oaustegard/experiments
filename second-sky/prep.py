"""Partition the bright-star sky into regions and build each region's local drawing frame.

Output: work/regions.json — one entry per region with its stars in a local
gnomonic frame (x right = west on the sky as seen looking up, y up = north),
scaled to 0..1000, plus the region's house and neighbours.
"""
import json, math, pathlib, random
import numpy as np

HERE = pathlib.Path(__file__).parent
K = 140          # number of regions
MAG_CUT = 5.2    # stars an agent may use in a figure
random.seed(7); np.random.seed(7)

stars = json.load(open(HERE / "data/stars.6.json"))["features"]
names = json.load(open(HERE / "data/starnames.json"))

def unit(ra, dec):
    a, d = math.radians(ra), math.radians(dec)
    return np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])

cat = []
for f in stars:
    ra, dec = f["geometry"]["coordinates"]
    ra %= 360
    p = f["properties"]
    cat.append(dict(id=f["id"], ra=ra, dec=dec, mag=p["mag"], bv=float(p["bv"] or 0.6)))
bright = [s for s in cat if s["mag"] <= MAG_CUT]
V = np.array([unit(s["ra"], s["dec"]) for s in bright])
W = np.array([10 ** (-0.2 * s["mag"]) for s in bright])  # brighter stars pull harder

# Spherical k-means, k-means++ init on brightness weights.
idx = [np.random.choice(len(V), p=W / W.sum())]
for _ in range(K - 1):
    d = np.clip(1 - (V @ V[idx].T).max(axis=1), 0, None)
    p = d * W; idx.append(np.random.choice(len(V), p=p / p.sum()))
C = V[idx].copy()
for it in range(60):
    lab = (V @ C.T).argmax(axis=1)
    for k in range(K):
        m = lab == k
        if m.any():
            c = (V[m] * W[m, None]).sum(0); C[k] = c / np.linalg.norm(c)
# Merge tiny clusters into their nearest neighbour.
lab = (V @ C.T).argmax(axis=1)

EPS = math.radians(23.4393)
def ecl_lat(v):
    # ecliptic north pole in equatorial coords
    pole = np.array([0, -math.sin(EPS), math.cos(EPS)])
    return math.degrees(math.asin(float(v @ pole)))

def house(c):
    dec = math.degrees(math.asin(c[2]))
    if dec > 55: return "The Spindle"
    if dec < -50: return "The Unflown"
    if abs(ecl_lat(c)) < 12: return "The Lamp Road"
    return "The Windward Reach" if dec > 0 else "The Leeward Reach"

regions = []
for k in range(K):
    members = [bright[i] for i in np.where(lab == k)[0]]
    if len(members) < 4:
        continue
    c = C[k]
    ra0 = math.atan2(c[1], c[0]); dec0 = math.asin(c[2])
    e = np.array([-math.sin(ra0), math.cos(ra0), 0])
    n = np.array([-math.sin(dec0) * math.cos(ra0), -math.sin(dec0) * math.sin(ra0), math.cos(dec0)])
    pts = []
    for s in members:
        v = unit(s["ra"], s["dec"])
        xi, eta = (v @ e) / (v @ c), (v @ n) / (v @ c)
        pts.append((-xi, eta))  # sky-view: east to the left
    xs, ys = zip(*pts)
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1e-3
    scale = 800 / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    loc = []
    for s, (x, y) in zip(members, pts):
        nm = names.get(str(s["id"]), {})
        loc.append(dict(id=s["id"], x=round(500 + (x - cx) * scale), y=round(500 - (y - cy) * scale),
                        mag=s["mag"], ra=round(s["ra"], 4), dec=round(s["dec"], 4), bv=s["bv"],
                        iau=nm.get("c", "")))
    loc.sort(key=lambda s: s["mag"])
    regions.append(dict(rid=len(regions), center=dict(ra=round(math.degrees(ra0) % 360, 3), dec=round(math.degrees(dec0), 3)),
                        frame=dict(cx=cx, cy=cy, scale=scale), house=house(c), stars=loc))

# Neighbours: 5 nearest centres.
Cs = np.array([unit(r["center"]["ra"], r["center"]["dec"]) for r in regions])
for r, v in zip(regions, Cs):
    order = np.argsort(-(Cs @ v))[1:6]
    r["neighbours"] = [int(i) for i in order]
    r["width_deg"] = round(math.degrees(math.atan(800 / r["frame"]["scale"] / 2)) * 2, 1)

out = HERE / "work"; out.mkdir(exist_ok=True)
json.dump(regions, open(out / "regions.json", "w"))
json.dump([dict(id=s["id"], ra=round(s["ra"], 4), dec=round(s["dec"], 4), mag=s["mag"], bv=s["bv"]) for s in cat],
          open(out / "catalog.json", "w"))
from collections import Counter
print(len(regions), "regions;", Counter(r["house"] for r in regions))
print("stars/region", sorted(len(r["stars"]) for r in regions)[::10])
print("width deg", sorted(r["width_deg"] for r in regions)[::10])
