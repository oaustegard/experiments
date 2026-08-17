#!/usr/bin/env python3
"""Single-GPU, per-hour cost model: one consumer card vs the same tokens on the API.

The sibling of `model.py`, at 1/1000th the scale and asking a different
question. `model.py` asks "do we buy a cluster for 800 seats"; this asks "while
this card is generating, is it cheaper than the API". Shared `params.json`, so
the price book and the electricity rates cannot drift between them.

Three things this models that the fleet version does not need to:

  1. THE DECODE ROOFLINE. A dense model read once per token from VRAM cannot
     exceed bandwidth / weight_bytes tokens/s single-stream. Quoted TPS figures
     routinely exceed it, which means they are batched, speculative, or short-
     context bursts -- all fine, all different from what the quote implies.
  2. PREFILL AND DECODE SHARE ONE GPU. There is no separate prefill pool on a
     desktop. At a 40:1 input:output ratio the card spends most of its time on
     prefill, so the advertised decode rate is not the sustained rate.
  3. CACHE WRITE PREMIUM. Luna bills cache writes at 1.25x uncached input and
     reads at 0.1x, so caching only pays above a ~22% hit rate. Below it,
     turning caching on costs money.

    python3 hourly.py                     # the table, both decode branches
    python3 hourly.py --decode 120        # override the premise
    python3 hourly.py --ac                # add summer air conditioning
    python3 hourly.py --capex 3299        # the box at MSRP-era card prices
    python3 hourly.py --json              # for recheck.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())
H = PARAMS["hourly"]
MTOK = 1e6

# A desktop is billed at one of two rates depending on where it lives. The
# other four rows in params.json are datacenter framings, and one of them
# (SOS price-to-compare) is supply-only and would understate by ~40%.
REPORT_RATES = ("md_residential_eia_corrected", "md_commercial_eia_aug2026")
LABEL = {"md_residential_eia_corrected": "residential",
         "md_commercial_eia_aug2026": "commercial"}


def wall_kw(hw: dict, ac: bool) -> float:
    """Wall draw. No PUE -- a desktop is not a datacenter -- but the heat still
    has to leave the room in a Maryland summer."""
    dc = hw["gpu_tgp_w"] + hw["host_w"] + hw["platform_w"]
    w = dc / hw["psu_efficiency"]
    if ac:
        w += w / H["ac_cop"]
    return w / 1000


def weights_gb(model: dict) -> float:
    return model["total_params"] * model["weight_bits_nvfp4"] / 8 / 1e9


def roofline(hw: dict, model: dict) -> dict:
    """Single-stream decode ceiling. Every weight is read once per token, so
    this is bandwidth / bytes, full stop -- no amount of tuning beats it, only
    a different decoding scheme (speculative, multi-token) or batching."""
    gb = weights_gb(model)
    ceil_ = hw["memory_bandwidth_gb_s"] / gb
    return {"weights_gb": gb, "ceiling_tok_s": ceil_,
            "realistic_tok_s": ceil_ * H["mbu"], "mbu": H["mbu"]}


def prefill_tok_s(hw: dict, model: dict) -> float:
    """Scaled from the one published measurement on this card, by parameter
    count. Dense prefill is compute-bound and linear in params, so this is a
    defensible scaling -- but it is a scaling, not a measurement."""
    return hw["measured_prefill_tok_s_llama31_8b"] * (8e9 / model["total_params"])


def effective_input_price(profile: dict, price: dict) -> float:
    """Blended $/M for billed input tokens. With caching on, misses are cache
    WRITES at 1.25x, not plain input -- which is why a low hit rate is worse
    than no caching at all."""
    if not profile["use_cache"]:
        return price["input"]
    h = profile["cache_hit_rate"]
    return h * price["cached_read"] + (1 - h) * price["cache_write"]


def cache_breakeven_hit_rate(price: dict) -> float:
    return (price["cache_write"] - price["input"]) / (price["cache_write"] - price["cached_read"])


def sustained(decode: float, prefill: float, profile: dict) -> dict:
    """Output tok/s once prefill takes its share of the card.

    Local prefix caching is the mirror of the API's cache reads: only the
    FRESH fraction of billed input actually has to be prefilled. Time per
    output token = 1/decode + fresh_ratio/prefill.
    """
    r = profile["billed_in_out_ratio"]
    fresh = (1 - profile["cache_hit_rate"]) * r if profile["use_cache"] else r
    tok_s = 1.0 / (1.0 / decode + fresh / prefill)
    return {"billed_ratio": r, "fresh_ratio": fresh, "sustained_tok_s": tok_s,
            "decode_share": (1 / decode) / (1 / decode + fresh / prefill)}


def run(a) -> dict:
    hw = PARAMS["hardware"][a.hardware]
    model = PARAMS["models"][a.model]
    price = PARAMS["api_prices_usd_per_mtok"][a.price]
    rates = PARAMS["electricity"]["rates_usd_per_kwh"]
    kw = wall_kw(hw, a.ac)
    pre = prefill_tok_s(hw, model)
    capex = a.capex if a.capex is not None else hw["capex_usd"]

    out = {
        "config": {"hardware": a.hardware, "model": a.model, "price_book": a.price,
                   "wall_kw": kw, "ac": a.ac, "capex_usd": capex,
                   "capex_life_years": PARAMS["selfhost_opex"]["capex_life_years"]},
        "roofline": roofline(hw, model),
        "prefill_tok_s": pre,
        "cache_breakeven_hit_rate": cache_breakeven_hit_rate(price),
        "electricity_usd_per_hour": {k: kw * rates[k]["value"] for k in REPORT_RATES},
        "branches": {},
    }
    out["roofline"]["premise_tok_s"] = a.decode
    out["roofline"]["premise_over_ceiling"] = a.decode / out["roofline"]["ceiling_tok_s"]
    out["roofline"]["premise_over_realistic"] = a.decode / out["roofline"]["realistic_tok_s"]

    life = out["config"]["capex_life_years"]
    res_rate = rates[H["residential_rate_key"]]["value"]
    pwr_hr = kw * res_rate

    for label, dec in (("premise", a.decode), ("conservative", a.conservative)):
        branch = {"decode_tok_s": dec, "profiles": {}}
        for key, p in H["profiles"].items():
            s = sustained(dec, pre, p)
            out_hr = s["sustained_tok_s"] * 3600
            eff = effective_input_price(p, price)
            api = out_hr / MTOK * price["output"] + out_hr * p["billed_in_out_ratio"] / MTOK * eff
            budget = api - pwr_hr
            branch["profiles"][key] = {
                "label": p["label"], **s,
                "output_tokens_per_hour": out_hr,
                "effective_input_usd_per_mtok": eff,
                "api_usd_per_hour": api,
                "multiple_of_electricity": {k: api / (kw * rates[k]["value"]) for k in REPORT_RATES},
                "breakeven_hours_per_year": capex / life / budget if budget > 0 else None,
                "breakeven_hours_per_day": capex / life / budget / 365 if budget > 0 else None,
            }
        branch["usd_per_mtok_output_electricity_only"] = {
            k: kw * rates[k]["value"] / (dec * 3600) * MTOK for k in REPORT_RATES}
        out["branches"][label] = branch

    out["duty_cycle"] = {}
    for hpd in H["duty_cycle_hours_per_day"]:
        hrs = hpd * 365
        out["duty_cycle"][str(hpd)] = {
            "hours_per_year": hrs,
            "capex_usd_per_hour": capex / life / hrs,
            "power_usd_per_hour": pwr_hr,
            "all_in_usd_per_hour": capex / life / hrs + pwr_hr,
        }
    return out


def fmt(o: dict) -> str:
    L = []
    c, r = o["config"], o["roofline"]
    L.append(f"{c['hardware']} running {c['model']}  |  wall {c['wall_kw']*1000:.0f} W"
             f"{' (incl. summer AC)' if c['ac'] else ''}  |  ${c['capex_usd']:,} box over {c['capex_life_years']}y")
    L.append(f"\nDECODE ROOFLINE  weights {r['weights_gb']:.1f} GB / {PARAMS['hardware'][c['hardware']]['memory_bandwidth_gb_s']:,} GB/s")
    L.append(f"  ceiling {r['ceiling_tok_s']:.0f} tok/s @100% MBU, {r['realistic_tok_s']:.0f} @{r['mbu']:.0%}")
    L.append(f"  premise {r['premise_tok_s']:.0f} tok/s is {r['premise_over_ceiling']:.2f}x the hard ceiling, "
             f"{r['premise_over_realistic']:.2f}x the {r['mbu']:.0%}-MBU figure"
             + ("\n    -> needs MTP / speculative decoding / batching; plain decode cannot reach it"
                if r["premise_over_ceiling"] > 1 else ""))
    L.append(f"\nPREFILL {o['prefill_tok_s']:,.0f} tok/s (scaled from the 8B measurement) "
             f"-> {o['prefill_tok_s']/r['premise_tok_s']:.0f}:1 vs decode")
    L.append(f"CACHING pays above a {o['cache_breakeven_hit_rate']:.1%} hit rate")
    L.append("\nELECTRICITY  " + "   ".join(
        f"{LABEL[k]}: ${v:.3f}/hr" for k, v in o["electricity_usd_per_hour"].items()))

    for label, b in o["branches"].items():
        L.append(f"\n===== decode {b['decode_tok_s']:.0f} tok/s ({label}) =====")
        L.append(f"{'profile':<20}{'billed':>9}{'fresh':>7}{'sustained':>11}{'out/hr':>9}"
                 f"{'eff $/Min':>11}{'Luna $/hr':>11}{'x res':>8}{'x comm':>8}{'BE h/day':>10}")
        for k, p in b["profiles"].items():
            m = p["multiple_of_electricity"]
            res = m["md_residential_eia_corrected"]
            com = m["md_commercial_eia_aug2026"]
            be = p["breakeven_hours_per_day"]
            L.append(f"{p['label']:<20}{p['billed_ratio']:>8.0f}:1{p['fresh_ratio']:>7.1f}"
                     f"{p['sustained_tok_s']:>10.0f}/s{p['output_tokens_per_hour']/1000:>8.0f}k"
                     f"{p['effective_input_usd_per_mtok']:>11.4f}{p['api_usd_per_hour']:>11.3f}"
                     f"{res:>7.1f}x{com:>7.1f}x"
                     + (f"{be:>10.1f}" if be else f"{'never':>10}"))
        e = b["usd_per_mtok_output_electricity_only"]
        L.append(f"  $/M output tokens, electricity only: "
                 + ", ".join(f"${v:.3f} {LABEL[k]}" for k, v in e.items()))

    L.append(f"\n===== cost per productive hour, ${c['capex_usd']:,} box =====")
    for hpd, d in o["duty_cycle"].items():
        L.append(f"  {int(hpd):>2} h/day flat out ({d['hours_per_year']:>5,} h/yr): "
                 f"capex ${d['capex_usd_per_hour']:>5.2f} + power ${d['power_usd_per_hour']:.2f} "
                 f"= ${d['all_in_usd_per_hour']:>5.2f}/hr")
    L.append("\n  'h/day' means FLAT-OUT GENERATION, not hours with the tool open.")
    L.append("  Interactive chat generates tokens ~5-10% of wall-clock time.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default=H["default_hardware"])
    ap.add_argument("--model", default=H["default_model"])
    ap.add_argument("--price", default="luna_direct")
    ap.add_argument("--decode", type=float, default=H["nominal_decode_tok_s"])
    ap.add_argument("--conservative", type=float, default=H["conservative_decode_tok_s"])
    ap.add_argument("--capex", type=float, default=None)
    ap.add_argument("--ac", action="store_true", help="add summer air conditioning load")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    o = run(a)
    print(json.dumps(o, indent=2) if a.json else fmt(o))


if __name__ == "__main__":
    main()
