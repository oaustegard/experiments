"""Assemble the final sky: map every constellation's local drawing back onto the celestial sphere.

  python3 build.py SRC  -> work/sky.json  (stars, Milky Way, constellations, epic)
"""
import json, math, sys, pathlib
import numpy as np

regions = json.load(open("work/regions.json"))

def basis(r):
    ra0, dec0 = math.radians(r["center"]["ra"]), math.radians(r["center"]["dec"])
    c = np.array([math.cos(dec0) * math.cos(ra0), math.cos(dec0) * math.sin(ra0), math.sin(dec0)])
    e = np.array([-math.sin(ra0), math.cos(ra0), 0])
    n = np.array([-math.sin(dec0) * math.cos(ra0), -math.sin(dec0) * math.sin(ra0), math.cos(dec0)])
    return c, e, n

def to_sky(r, x, y):
    """Inverse of prep.py: local (x right, y down) -> (ra, dec) degrees."""
    c, e, n = basis(r); f = r["frame"]
    xi = -(f["cx"] + (x - 500) / f["scale"])
    eta = f["cy"] - (y - 500) / f["scale"]
    v = c + xi * e + eta * n; v /= np.linalg.norm(v)
    return round(math.degrees(math.atan2(v[1], v[0])) % 360, 3), round(math.degrees(math.asin(v[2])), 3)

def selftest():
    worst = 0
    for r in regions:
        for s in r["stars"]:
            ra, dec = to_sky(r, s["x"], s["y"])
            d = math.degrees(math.acos(min(1, math.sin(math.radians(dec)) * math.sin(math.radians(s["dec"])) +
                 math.cos(math.radians(dec)) * math.cos(math.radians(s["dec"])) * math.cos(math.radians(ra - s["ra"])))))
            worst = max(worst, d)
    assert worst < 0.1, worst
    return worst

def lonlat(ra, dec):  # d3 longitudes run -180..180
    return [round(ra - 360 if ra > 180 else ra, 3), round(dec, 3)]

import re
NAME_RE = re.compile(r"^[ktslrnmvhaeiouáéíóú'-]+$", re.I)
def clean_name(woven, original):
    """Woven names sometimes arrive as '*Lenkat* — "the Lidded Jar"'; keep the Ilvari word only."""
    for cand in (woven or "", original):
        w = re.split(r"\s+[—–-]\s+|\s*\(|,", cand.replace("*", "").replace('"', "").strip())[0].strip()
        if w and NAME_RE.match(w): return w[0].upper() + w[1:]
    return original

def build(src):
    print("inverse-projection self-test, worst error deg:", round(selftest(), 4))
    cat = json.load(open("work/catalog.json"))
    stars = [[*lonlat(s["ra"], s["dec"]), s["mag"], round(s["bv"], 2)] for s in cat]
    mw = json.load(open("data/mw.json"))
    def rnd(o):
        if isinstance(o, list): return [rnd(x) for x in o]
        return round(o, 1) if isinstance(o, float) else o
    milky = [rnd(f["geometry"]["coordinates"]) for f in mw["features"]]
    weave = pathlib.Path("work/out3")
    blind = sorted(pathlib.Path("work").glob("out2b_*"))
    cons = []
    for r in regions:
        p = pathlib.Path(src) / f"r{r['rid']:03d}.json"
        if not p.exists(): continue
        d = json.load(open(p)); o = d["output"]
        pos = {s["id"]: s for s in r["stars"]}
        w = weave / f"r{r['rid']:03d}.json"
        wo = json.load(open(w))["output"] if w.exists() else {}
        used = sorted({i for e in o["lines"] for i in e if i in pos})
        vs = np.array([[math.cos(math.radians(pos[i]["dec"])) * math.cos(math.radians(pos[i]["ra"])),
                        math.cos(math.radians(pos[i]["dec"])) * math.sin(math.radians(pos[i]["ra"])),
                        math.sin(math.radians(pos[i]["dec"]))] for i in used]).mean(0)
        cra, cdec = math.degrees(math.atan2(vs[1], vs[0])) % 360, math.degrees(math.asin(vs[2] / np.linalg.norm(vs)))
        last_guess = d.get("guess")
        cons.append(dict(
            id=r["rid"], name=clean_name(wo.get("name"), o["name"]), gloss=o["gloss"], house=r["house"],
            myth=wo.get("myth", o["myth"]), verse=o["verse"], gods=o["gods"],
            finding=wo.get("finding"),
            center=lonlat(cra, cdec),
            lines=[[*lonlat(pos[a]["ra"], pos[a]["dec"]), *lonlat(pos[b]["ra"], pos[b]["dec"])] for a, b in o["lines"] if a in pos and b in pos],
            figure=[[lonlat(*to_sky(r, x, y)) for x, y in pl] for pl in o["figure"] if len(pl) >= 2],
            anatomy=[dict(p=lonlat(pos[a["star"]]["ra"], pos[a["star"]]["dec"]), part=a["part"]) for a in o["anatomy"] if a["star"] in pos],
            stars=[dict(p=lonlat(pos[s["star"]]["ra"], pos[s["star"]]["dec"]), name=s["name"], gloss=s["gloss"],
                        mag=pos[s["star"]]["mag"], iau=pos[s["star"]]["iau"]) for s in o["star_names"] if s["star"] in pos],
            neighbours=r["neighbours"], blind=last_guess, redrawn=len(d.get("history", []))))
    from collections import Counter
    dup = [n for n, k in Counter(c["name"].lower() for c in cons).items() if k > 1]
    print("bad names:", [c["name"] for c in cons if not NAME_RE.match(c["name"])], "duplicates:", dup)
    epic = json.load(open("work/epic.json")) if pathlib.Path("work/epic.json").exists() else None
    calls = cost = secs = 0
    for p in pathlib.Path("work").glob("out*/*.json"):
        d = json.load(open(p)); calls += 1; cost += d.get("cost", 0); secs += d.get("secs", 0)
    match = [json.load(open(p))["match"] for p in pathlib.Path("work").glob("final/*.json")]
    first = [json.load(open(p))["output"]["match"] for p in pathlib.Path("work").glob("out2j_1/*.json")]
    rate = lambda m: f"{sum(x >= 4 for x in m)} of {len(m)}" if m else "n/a"
    redrawn = sum(c["redrawn"] > 0 for c in cons)
    import datetime
    meta = dict(agents=f"{calls:,}", date=datetime.date.today().isoformat(), rows=[
        ("140", "patches of sky, one model each, from the 2,061 stars brighter than magnitude 5.2"),
        (f"{calls:,}", "separate model calls, every one a fresh Claude Haiku 5.5 with no memory of the others"),
        (f"${cost:.2f}", "total model cost"),
        (f"{secs/3600:.1f} h", "of model time, run up to 28 at a time"),
        (rate(first), "first drawings a blind reader identified (judge score 4+ of 5)"),
        (str(redrawn), "figures redrawn after critique (two rounds, then three fresh attempts each for the hardest)"),
        (rate(match), "final drawings a blind reader identified, keeping the best-judged version of each"),
        ("6 → 1", "competing epics, picked by a jury of 7")])
    sky = dict(stars=stars, milky=milky, cons=cons, epic=epic, meta=meta)
    json.dump(sky, open("work/sky.json", "w"), separators=(",", ":"), ensure_ascii=False)
    print(len(cons), "constellations;", round(pathlib.Path("work/sky.json").stat().st_size / 1e6, 2), "MB")

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "work/final")
