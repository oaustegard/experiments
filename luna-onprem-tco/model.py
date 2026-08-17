#!/usr/bin/env python3
"""On-prem vs API cost model for a Luna-capability LLM deployment.

The question this answers: for N office seats, is it cheaper to buy the
hardware that matches GPT-5.6 Luna's capability, or to keep paying Luna's
per-token price?

The question it was ORIGINALLY asked, and why that was the wrong one: "how
does raw electricity cost compare to $0.22/M input?" Electricity wins that
comparison by ~60x and it does not matter, because electricity is 3-6% of
the cost of owning the box. The binding constraint is a MEMORY FLOOR --
matching Luna means running DeepSeek V4 Pro, which means ~800 GB of weights,
which means a ~$450k node whether you use 5% of it or 85%. This model exists
to make that floor visible and to let you find the seat count and usage
intensity where it finally pays.

Stdlib only. No network. Every constant lives in params.json with a source.

    python3 model.py                          # defaults: 800 seats, scenario B
    python3 model.py --scenario C --seats 2000
    python3 model.py --all-scenarios          # the comparison table
    python3 model.py --sweep-mfu              # sensitivity on the weakest input
    python3 model.py --first-pass             # the GB200 NVL72 energy-only framing
    python3 model.py --json                   # machine-readable, for recheck.py

Every dollar figure is per year unless the key says otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())

SEC_PER_HOUR = 3600
HOURS_PER_YEAR = 8760
MTOK = 1e6


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------

def prefill_tok_s(hw: dict, model: dict, mfu: float) -> float:
    """Aggregate input-token throughput of the box.

    MODELLED, not measured -- see params.json:prefill. Prefill is compute
    bound, so it is (peak FLOPs x MFU) / (FLOPs per token), and FLOPs per
    token is 2 x active params.
    """
    per_tok = PARAMS["prefill"]["flops_per_token_multiplier"] * model["active_params"]
    return hw["gpu_fp4_dense_flops"] * mfu / per_tok * hw["gpus"]


def decode_tok_s(hw: dict, alt: bool = False) -> float:
    """Aggregate output-token throughput. MEASURED (InferenceX), at a stated
    interactivity target. `alt` selects the higher-throughput / lower-
    interactivity operating point where one is published."""
    key = "measured_decode_tok_s_per_gpu_alt" if alt else "measured_decode_tok_s_per_gpu"
    per_gpu = hw.get(key)
    if per_gpu is None:
        raise ValueError("no measured decode throughput for this hardware profile")
    return per_gpu * hw["gpus"]


def fits(hw: dict, model: dict, weight_key: str) -> bool:
    """Does the model's weight footprint fit this box's VRAM? Falls back to
    total_params x weight_bits for entries that carry a bit-width instead of a
    byte count (the consumer-scale models, which `hourly.py` owns)."""
    if weight_key in model:
        need = model[weight_key]
    elif "weight_bits_nvfp4" in model:
        need = model["total_params"] * model["weight_bits_nvfp4"] / 8
    else:
        return False
    return hw["vram_bytes"] >= need


def power_kw(hw: dict, util: float, serving: bool = True) -> float:
    """Wall power at a given utilisation, including PUE.

    Two regimes, and conflating them was a real bug in the first draft of this
    file. A GPU that is *serving* does not scale linearly down to idle -- it
    sits around a third of the load delta even when barely busy, so serving
    power is idle + (load - idle) * (0.35 + 0.65 * util). A GPU that is
    *parked* -- no requests in flight, weights resident -- draws idle. Applying
    the 0.35 serving floor to parked hours inflated annual kWh by ~35%, which
    made the electricity line look more important than it is. Since electricity
    turned out to be the thing this experiment was asked about, that error was
    pointed the wrong way.
    """
    idle, load = hw["it_idle_kw"], hw["it_load_kw"]
    if not serving:
        return idle * hw["pue"]
    return (idle + (load - idle) * (0.35 + 0.65 * util)) * hw["pue"]


# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------

def workload(scenario_key: str, seats: int, w: dict) -> dict:
    s = w["scenarios"][scenario_key]
    active = seats * w["daily_active_fraction"]
    reqs_day = active * s["requests_per_active_user_per_day"]
    tin_day = reqs_day * s["input_tokens_per_request"]
    tout_day = reqs_day * s["output_tokens_per_request"]
    return {
        "scenario": scenario_key,
        "label": s["label"],
        "active_users": active,
        "requests_per_day": reqs_day,
        "input_tokens_per_day": tin_day,
        "output_tokens_per_day": tout_day,
        "input_tokens_per_year": tin_day * w["workdays_per_year"],
        "output_tokens_per_year": tout_day * w["workdays_per_year"],
        "output_tokens_per_request": s["output_tokens_per_request"],
    }


def peak(wl: dict, hw_decode_tok_s: float, w: dict, interactivity: float) -> dict:
    """Busiest-hour load. This is where the power-curve activity distribution
    enters: peak_hour_request_share concentrates the day's requests."""
    reqs_s = wl["requests_per_day"] * w["peak_hour_request_share"] / SEC_PER_HOUR
    occupancy_s = wl["output_tokens_per_request"] / interactivity
    return {
        "peak_requests_per_s": reqs_s,
        "peak_concurrency": reqs_s * occupancy_s,
        "peak_decode_tok_s": reqs_s * wl["output_tokens_per_request"],
        "peak_utilisation": reqs_s * wl["output_tokens_per_request"] / hw_decode_tok_s,
    }


# --------------------------------------------------------------------------
# money
# --------------------------------------------------------------------------

def api_cost(wl: dict, price: dict) -> float:
    return (wl["input_tokens_per_year"] / MTOK * price["input"]
            + wl["output_tokens_per_year"] / MTOK * price["output"])


def annual_kwh(hw: dict, peak_util: float, w: dict) -> dict:
    """Three power states: covered workday hours, the nightly batch window,
    and everything else idle. Average daytime utilisation is well below peak
    utilisation -- the peak hour is one hour of a coverage window many hours
    long -- so it is derived rather than assumed equal."""
    share = w["peak_hour_request_share"]
    coverage = w["coverage_hours_per_workday"]
    mean_day_util = min(peak_util * (1.0 / share) / coverage, 1.0)

    day_h = coverage * w["workdays_per_year"]
    bat_h = w["batch_hours_per_night"] * w["batch_nights_per_year"]
    idle_h = HOURS_PER_YEAR - day_h - bat_h

    p_day = power_kw(hw, mean_day_util)
    p_bat = power_kw(hw, 0.95)
    p_idle = power_kw(hw, 0.0, serving=False)
    return {
        "mean_daytime_utilisation": mean_day_util,
        "kwh": day_h * p_day + bat_h * p_bat + idle_h * p_idle,
        "kw_daytime": p_day, "kw_batch": p_bat, "kw_idle": p_idle,
        "duty_cycle": (day_h * mean_day_util + bat_h * 0.95) / HOURS_PER_YEAR,
    }


def selfhost(hw: dict, energy: dict, rate: float, ops: float, life: int,
             nodes: int = 1) -> dict:
    power = energy["kwh"] * rate * nodes
    capex = hw["capex_usd"] * nodes / life
    return {
        "kwh_per_year": energy["kwh"] * nodes,
        "power_usd": power,
        "capex_amortised_usd": capex,
        "ops_usd": ops,
        "hw_plus_power_usd": capex + power,
        "total_usd": capex + power + ops,
        "power_share_of_hw_tco": power / (capex + power),
        "power_share_of_total": power / (capex + power + ops),
    }


def energy_per_mtok(hw: dict, tok_s: float, util: float, rate: float) -> dict:
    """Marginal energy to push a million tokens, at a stated utilisation.
    This is the FIRST-PASS framing -- true, and nearly irrelevant to the
    decision, which is the finding."""
    kw = power_kw(hw, util)
    kwh = kw * (MTOK / tok_s) / SEC_PER_HOUR
    return {"tok_s": tok_s, "kwh_per_mtok": kwh, "usd_per_mtok": kwh * rate}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def run(args) -> dict:
    w = dict(PARAMS["workload"])
    w["seats"] = args.seats
    if args.batch_hours is not None:
        w["batch_hours_per_night"] = args.batch_hours

    hw = PARAMS["hardware"][args.hardware]
    if "it_load_kw" not in hw:
        raise SystemExit(
            f"{args.hardware} is a single-GPU desktop profile, not a datacenter node.\n"
            f"The fleet model assumes rack power states, a measured aggregate decode\n"
            f"figure, and no prefill/decode contention -- none of which hold on one card.\n"
            f"Use:  python3 hourly.py --hardware {args.hardware}")
    model = PARAMS["models"][args.model]
    rate = args.rate if args.rate is not None else \
        PARAMS["electricity"]["rates_usd_per_kwh"][PARAMS["electricity"]["default_rate"]]["value"]
    ops = args.ops if args.ops is not None else PARAMS["selfhost_opex"]["ops_usd_per_year"]
    life = PARAMS["selfhost_opex"]["capex_life_years"]

    prices = PARAMS["api_prices_usd_per_mtok"]
    p_int = prices[args.interactive_price]
    p_bat = prices[args.batch_price]

    pre = prefill_tok_s(hw, model, args.mfu)
    dec = decode_tok_s(hw)
    interactivity = hw["measured_decode_at_tok_s_per_user"]

    out = {
        "config": {
            "seats": args.seats, "hardware": args.hardware, "model": args.model,
            "mfu": args.mfu, "rate_usd_per_kwh": rate, "ops_usd_per_year": ops,
            "capex_usd": hw["capex_usd"], "capex_life_years": life,
            "interactive_price": args.interactive_price,
            "batch_price": args.batch_price,
            "weights_fit": fits(hw, model, args.weight_key),
        },
        "hardware": {
            "prefill_input_tok_s": pre,
            "decode_output_tok_s": dec,
            "interactivity_tok_s_per_user": interactivity,
            "kw_loaded": power_kw(hw, 1.0),
            "kw_idle": power_kw(hw, 0.0, serving=False),
        },
        "scenarios": {},
    }

    # what the batch window can physically absorb, saturated
    bat_night = pre * w["batch_hours_per_night"] * SEC_PER_HOUR
    out["batch"] = {
        "hours_per_night": w["batch_hours_per_night"],
        "saturated_input_tokens_per_night": bat_night,
        "saturated_input_tokens_per_year": bat_night * w["batch_nights_per_year"],
        "value_at_batch_price_usd": bat_night * w["batch_nights_per_year"] / MTOK * p_bat["input"],
    }

    keys = list(w["scenarios"]) if args.all_scenarios else [args.scenario]
    for k in keys:
        wl = workload(k, args.seats, w)
        pk = peak(wl, dec, w, interactivity)

        # A box cannot serve more than 100% of itself. Without this the model
        # happily returned "self-host" at 845% peak utilisation -- comparing
        # one node's cost against a token bill that needs nine. Every such
        # error flatters self-hosting, so it has to be structural, not a note.
        nodes_required = max(1, math.ceil(pk["peak_utilisation"] - 1e-9))
        nodes = max(args.nodes, nodes_required)
        pk["nodes_required_for_peak"] = nodes_required
        pk["nodes_costed"] = nodes
        # per-node utilisation once the fleet is sized
        per_node_util = pk["peak_utilisation"] / nodes

        en = annual_kwh(hw, per_node_util, w)
        sh = selfhost(hw, en, rate, ops, life, nodes)
        api = api_cost(wl, p_int)

        batch_value = out["batch"]["value_at_batch_price_usd"] * nodes
        blended = ((wl["input_tokens_per_year"] * p_int["input"]
                    + wl["output_tokens_per_year"] * p_int["output"])
                   / max(wl["input_tokens_per_year"] + wl["output_tokens_per_year"], 1))
        out["scenarios"][k] = {
            **wl, **pk, **en, "selfhost": sh,
            "api_interactive_usd": api,
            "api_plus_saturated_batch_usd": api + batch_value,
            "saturated_batch_value_usd": batch_value,
            "blended_price_usd_per_mtok": blended,
            "breakeven": {
                "vs_hw_plus_power": {
                    "blended_tokens_per_year": sh["hw_plus_power_usd"] / blended * MTOK,
                    "blended_tokens_per_workday": sh["hw_plus_power_usd"] / blended * MTOK / w["workdays_per_year"],
                    "batch_input_tokens_per_night": sh["hw_plus_power_usd"] / p_bat["input"] * MTOK / w["batch_nights_per_year"],
                },
                "vs_total": {
                    "blended_tokens_per_year": sh["total_usd"] / blended * MTOK,
                    "blended_tokens_per_workday": sh["total_usd"] / blended * MTOK / w["workdays_per_year"],
                    "batch_input_tokens_per_night": sh["total_usd"] / p_bat["input"] * MTOK / w["batch_nights_per_year"],
                },
            },
            "verdict": "self-host" if api + batch_value > sh["total_usd"] else "API",
        }

    # first-pass framing: marginal energy per Mtok, both token kinds
    if args.first_pass:
        fp = {}
        for hwname in ("GB200-NVL72", "8xB200-HGX"):
            h = PARAMS["hardware"][hwname]
            p = prefill_tok_s(h, model, args.mfu)
            d = decode_tok_s(h)
            fp[hwname] = {
                "input": energy_per_mtok(h, p, 1.0, rate),
                "output": energy_per_mtok(h, d, 1.0, rate),
            }
            fp[hwname]["input"]["api_multiple_vs_luna_direct"] = \
                prices["luna_direct"]["input"] / fp[hwname]["input"]["usd_per_mtok"]
            fp[hwname]["input"]["api_multiple_vs_luna_batch"] = \
                prices["luna_batch"]["input"] / fp[hwname]["input"]["usd_per_mtok"]
            fp[hwname]["output"]["api_multiple_vs_luna_direct"] = \
                prices["luna_direct"]["output"] / fp[hwname]["output"]["usd_per_mtok"]
        out["first_pass_energy_only"] = fp

    if args.sweep_mfu:
        lo, hi = PARAMS["prefill"]["mfu_range"]
        sweep = {}
        for m in (lo, (lo + hi) / 2, args.mfu, hi):
            p = prefill_tok_s(hw, model, m)
            n = p * w["batch_hours_per_night"] * SEC_PER_HOUR
            sweep[f"{m:.2f}"] = {
                "prefill_input_tok_s": p,
                "saturated_batch_input_tokens_per_night": n,
                "value_at_batch_price_usd_per_year": n * w["batch_nights_per_year"] / MTOK * p_bat["input"],
                "energy_usd_per_mtok_input": energy_per_mtok(hw, p, 1.0, rate)["usd_per_mtok"],
            }
        out["mfu_sweep"] = sweep

    return out


def fmt(out: dict) -> str:
    L = []
    c, h = out["config"], out["hardware"]
    L.append(f"{c['hardware']} running {c['model']}  |  weights fit: {c['weights_fit']}")
    L.append(f"  prefill {h['prefill_input_tok_s']:>12,.0f} input tok/s   (MODELLED at {c['mfu']:.0%} MFU)")
    L.append(f"  decode  {h['decode_output_tok_s']:>12,.0f} output tok/s  (measured @{h['interactivity_tok_s_per_user']} tok/s/user)")
    L.append(f"  power   {h['kw_loaded']:>12,.1f} kW loaded / {h['kw_idle']:.1f} kW idle")
    L.append(f"  cost    ${c['capex_usd']:>11,.0f} capex over {c['capex_life_years']}y, "
             f"electricity @ ${c['rate_usd_per_kwh']:.4f}/kWh, ops ${c['ops_usd_per_year']:,.0f}/yr")

    b = out["batch"]
    L.append(f"\nbatch window {b['hours_per_night']} h/night, SATURATED: "
             f"{b['saturated_input_tokens_per_night']/1e9:.2f}B input tok/night "
             f"= ${b['value_at_batch_price_usd']:,.0f}/yr of avoided batch-price spend")

    L.append(f"\n{'sc':<3}{'label':<26}{'in/day':>9}{'out/day':>9}{'peak':>7}{'n':>3}{'duty':>7}"
             f"{'kWh/yr':>10}{'power$':>9}{'self $/yr':>11}{'API $/yr':>10}{'+batch':>10}  verdict")
    for k, s in out["scenarios"].items():
        sh = s["selfhost"]
        L.append(f"{k:<3}{s['label']:<26}"
                 f"{s['input_tokens_per_day']/1e9:>8.2f}B{s['output_tokens_per_day']/1e9:>8.2f}B"
                 f"{s['peak_utilisation']:>7.0%}{s['nodes_costed']:>3}{s['duty_cycle']:>7.1%}"
                 f"{sh['kwh_per_year']:>10,.0f}{sh['power_usd']:>9,.0f}"
                 f"{sh['total_usd']:>11,.0f}{s['api_interactive_usd']:>10,.0f}"
                 f"{s['api_plus_saturated_batch_usd']:>10,.0f}  {s['verdict']}")

    for k, s in out["scenarios"].items():
        sh, be = s["selfhost"], s["breakeven"]
        L.append(f"\n[{k}] power is {sh['power_share_of_hw_tco']:.1%} of hardware TCO, "
                 f"{sh['power_share_of_total']:.1%} of all-in; "
                 f"${sh['power_usd']/out['config']['seats']:.0f}/seat/yr")
        L.append(f"     break-even vs hw+power (${sh['hw_plus_power_usd']:,.0f}): "
                 f"{be['vs_hw_plus_power']['blended_tokens_per_workday']/1e9:.2f}B blended tok/workday, "
                 f"or {be['vs_hw_plus_power']['batch_input_tokens_per_night']/1e9:.2f}B batch tok/night")
        L.append(f"     break-even vs all-in  (${sh['total_usd']:,.0f}): "
                 f"{be['vs_total']['blended_tokens_per_workday']/1e9:.2f}B blended tok/workday, "
                 f"or {be['vs_total']['batch_input_tokens_per_night']/1e9:.2f}B batch tok/night")
        cap = b["saturated_input_tokens_per_night"] * s["nodes_costed"]
        need = be["vs_total"]["batch_input_tokens_per_night"]
        if need > cap:
            L.append(f"     -> batch route IMPOSSIBLE: needs {need/1e9:.2f}B/night, "
                     f"box tops out at {cap/1e9:.2f}B/night even at 100% saturation")

    if "first_pass_energy_only" in out:
        L.append("\nfirst-pass framing (marginal energy only, box assumed 100% busy):")
        for name, d in out["first_pass_energy_only"].items():
            L.append(f"  {name:<14} input  {d['input']['kwh_per_mtok']*1000:>7.1f} Wh/Mtok  "
                     f"${d['input']['usd_per_mtok']:.5f}/M  -> Luna direct is "
                     f"{d['input']['api_multiple_vs_luna_direct']:.0f}x, batch "
                     f"{d['input']['api_multiple_vs_luna_batch']:.0f}x")
            L.append(f"  {'':<14} output {d['output']['kwh_per_mtok']*1000:>7.1f} Wh/Mtok  "
                     f"${d['output']['usd_per_mtok']:.5f}/M  -> Luna direct is "
                     f"{d['output']['api_multiple_vs_luna_direct']:.0f}x")

    if "mfu_sweep" in out:
        L.append("\nMFU sensitivity (the weakest input in the model):")
        for m, d in out["mfu_sweep"].items():
            L.append(f"  {float(m):.0%}  prefill {d['prefill_input_tok_s']:>10,.0f} tok/s   "
                     f"batch {d['saturated_batch_input_tokens_per_night']/1e9:>5.2f}B/night   "
                     f"worth ${d['value_at_batch_price_usd_per_year']:>9,.0f}/yr   "
                     f"energy ${d['energy_usd_per_mtok_input']:.5f}/Mtok")
    return "\n".join(L)


def main():
    w = PARAMS["workload"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seats", type=int, default=w["seats"])
    ap.add_argument("--scenario", default=w["default_scenario"], choices=list(w["scenarios"]))
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--hardware", default=PARAMS["hardware"]["default_hardware"],
                    choices=[k for k in PARAMS["hardware"] if not k.startswith("_")
                             and k != "default_hardware"])
    ap.add_argument("--model", default="deepseek-v4-pro-0813",
                    choices=[k for k in PARAMS["models"]])
    ap.add_argument("--weight-key", default="weight_bytes_nvfp4")
    ap.add_argument("--mfu", type=float, default=PARAMS["prefill"]["mfu"])
    ap.add_argument("--rate", type=float, default=None, help="USD/kWh override")
    ap.add_argument("--ops", type=float, default=None, help="USD/yr override; 0 for the floor")
    ap.add_argument("--nodes", type=int, default=1, help="2 for N+1 redundancy")
    ap.add_argument("--batch-hours", type=float, default=None)
    ap.add_argument("--interactive-price", default="luna_direct")
    ap.add_argument("--batch-price", default="luna_batch")
    ap.add_argument("--first-pass", action="store_true",
                    help="also print the energy-only framing this experiment started from")
    ap.add_argument("--sweep-mfu", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = run(a)
    print(json.dumps(out, indent=2) if a.json else fmt(out))


if __name__ == "__main__":
    main()
