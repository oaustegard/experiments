import json, sys, statistics as st
from pathlib import Path
R = Path("/home/user/experiments/image-to-wgsl/results")
raw = json.loads((R/"bit_allocation_raw.json").read_text())
S = json.loads((R/"summary.json").read_text())["images"]
ims = list(raw)
def pt(im, n):
    return next(p for p in S[im]["prefix1024"]["points"] if p["n"] == n)
slope = {}
for lo, hi in ((256, 512), (512, 1024)):
    v = [(pt(i,hi)["psnr"]-pt(i,lo)["psnr"])/(pt(i,hi)["gz_packed"]-pt(i,lo)["gz_packed"]) for i in ims]
    slope[f"{lo}-{hi}"] = {"mean_db_per_byte": st.mean(v), "per_image": dict(zip(ims, v))}
# marginal slope at each n: use 256->512 for n=256 comparison, 512->1024 for n=1024
ref = {256: slope["256-512"]["mean_db_per_byte"], 1024: slope["512-1024"]["mean_db_per_byte"]}
out = {"slope": slope, "fields": {}}
keys = [k for k in raw[ims[0]] if k != "base"]
for key in keys:
    row = {}
    for n in (256, 1024):
        dl = [raw[i]["base"][f"psnr{n}"] - raw[i][key][f"psnr{n}"] for i in ims]
        sv = [raw[i]["base"][f"gz{n}"] - raw[i][key][f"gz{n}"] for i in ims]
        row[str(n)] = {"bits": raw[ims[0]][key]["bits"], "psnr_loss_db": st.mean(dl), "bytes_saved": st.mean(sv),
                       "loss_db_per_byte": st.mean(dl)/st.mean(sv) if st.mean(sv) > 0 else None,
                       "splat_slope_db_per_byte": ref[n],
                       "net_gain_db_if_bytes_respent": st.mean(sv)*ref[n]-st.mean(dl),
                       "over_provisioned": (st.mean(dl)/st.mean(sv) < ref[n]) if st.mean(sv) > 0 else None,
                       "worst_image_loss_db": max(dl)}
    out["fields"][key] = row
out["base"] = {n: {"psnr": st.mean(raw[i]["base"][f"psnr{n}"] for i in ims), "gz": st.mean(raw[i]["base"][f"gz{n}"] for i in ims)} for n in (256, 1024)}
(R/"bit_allocation.json").write_text(json.dumps({"raw": raw, **out}, indent=1))
print("slopes", {k: v["mean_db_per_byte"] for k, v in slope.items()}, "per KB", {k: 1000*v for k,v in ref.items()})
for key, r in out["fields"].items():
    print(f"{key:9s}", *[f"n={n}: bits{r[str(n)]['bits']} loss {r[str(n)]['psnr_loss_db']:.3f}dB save {r[str(n)]['bytes_saved']:.0f}B  {1000*(r[str(n)]['loss_db_per_byte'] or 0):.3f}dB/KB net {r[str(n)]['net_gain_db_if_bytes_respent']:+.3f}" for n in (256,1024)])

raw2 = json.loads((R/"bit_allocation_raw2.json").read_text())
comb = {}
for key in ["shape-drop", "colour-3", "rec_A", "rec_B"]:
    row = {}
    for n in (256, 1024):
        dl = [raw[i]["base"][f"psnr{n}"] - raw2[i][key][f"psnr{n}"] for i in ims]
        sv = [raw[i]["base"][f"gz{n}"] - raw2[i][key][f"gz{n}"] for i in ims]
        row[str(n)] = {"bits": raw2[ims[0]][key]["bits"], "psnr_loss_db": st.mean(dl), "bytes_saved": st.mean(sv),
                       "loss_db_per_byte": st.mean(dl)/st.mean(sv), "splat_slope_db_per_byte": ref[n],
                       "net_gain_db_if_bytes_respent": st.mean(sv)*ref[n]-st.mean(dl),
                       "per_image_net": {i: (raw[i]["base"][f"gz{n}"]-raw2[i][key][f"gz{n}"])*ref[n]-(raw[i]["base"][f"psnr{n}"]-raw2[i][key][f"psnr{n}"]) for i in ims}}
    comb[key] = row
    print(key, *[f"n={n}: bits{row[str(n)]['bits']} loss {row[str(n)]['psnr_loss_db']:.3f} save {row[str(n)]['bytes_saved']:.0f}B net {row[str(n)]['net_gain_db_if_bytes_respent']:+.3f} ({row[str(n)]['bytes_saved']/out['base'][n]['gz']*100:.1f}% of {out['base'][n]['gz']:.0f}B)" for n in (256,1024)])
    print("   per-image net@1024", {i[-2:]: round(v,3) for i, v in row["1024"]["per_image_net"].items()})
(R/"bit_allocation.json").write_text(json.dumps({"raw": raw, "raw_extra": raw2, **out, "combined": comb}, indent=1))
