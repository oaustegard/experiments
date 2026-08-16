#!/usr/bin/env python3
"""Sub-5-second re-verification. Run on any touch of this directory.

This experiment has no expensive artifact to rebuild -- it is a closed-form
model over ~40 constants. What decays instead is the coupling between three
things that can each be edited alone: params.json, model.py, and the prose in
RESULTS.md. Two of the four errors in ERRORS.md were exactly that: arithmetic
that was right and a sentence about it that was wrong, or a formula that was
defensible in one place and applied in another where it was not.

Five phases, cheapest first:

  1. PARAMS HYGIENE      every numeric constant carries a source and a
                         confidence tag; nothing sourced to "I remember"
  2. INDEPENDENT RECOMPUTE  the headline numbers, recomputed here with inline
                         arithmetic that imports NOTHING from model.py, must
                         agree with model.py --json
  3. PROSE vs MODEL      the numbers written into RESULTS.md are parsed back
                         out and checked against the model, so the writeup
                         cannot drift from the code
  4. STRUCTURAL INVARIANTS  the things that, if they break, silently flatter
                         self-hosting -- node sizing, batch capacity, the
                         serving/parked power split
  5. NEGATIVE CONTROLS   a check that cannot go red is not evidence; these
                         perturb inputs and assert the conclusion moves

Run:  python3 recheck.py       (non-zero exit on any failure)
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "params.json").read_text())
RESULTS = (HERE / "RESULTS.md").read_text() if (HERE / "RESULTS.md").exists() else ""

FAILS: list[str] = []
N = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global N
    N += 1
    if not ok:
        FAILS.append(f"{label}{(' -- ' + detail) if detail else ''}")


def close(a: float, b: float, tol: float = 0.005) -> bool:
    """Relative agreement. Prose rounds; the model does not."""
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tol


def _run(script: str, *args: str) -> dict:
    r = subprocess.run([sys.executable, str(HERE / script), "--json", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        FAILS.append(f"{script} {' '.join(args)} exited {r.returncode}: {r.stderr[-400:]}")
        return {}
    return json.loads(r.stdout)


def model(*args: str) -> dict:
    return _run("model.py", *args)


def hourly(*args: str) -> dict:
    return _run("hourly.py", *args)


# --------------------------------------------------------------------------
print("1. params hygiene")

for name, row in P["electricity"]["rates_usd_per_kwh"].items():
    if name.startswith("_"):
        continue
    check(f"rate {name} has source", "source" in row)
    check(f"rate {name} has confidence", "confidence" in row)
    check(f"rate {name} plausible", 0.05 <= row["value"] <= 0.40, str(row["value"]))

for name, row in P["api_prices_usd_per_mtok"].items():
    if name.startswith("_") or name.startswith("default"):
        continue
    check(f"price {name} has input+output", "input" in row and "output" in row)
    check(f"price {name} output >= input", row["output"] >= row["input"])
    check(f"price {name} has confidence", "confidence" in row)

for name, m in P["models"].items():
    check(f"model {name} declares open_weights", "open_weights" in m)
    check(f"model {name} has source", "source" in m)

for name, hw in P["hardware"].items():
    if name == "default_hardware":
        continue
    check(f"hw {name} has sources", "sources" in hw)
    check(f"hw {name} has capex", "capex_usd" in hw)
    if "it_load_kw" in hw:                      # datacenter node/rack profile
        check(f"hw {name} idle < load", hw["it_idle_kw"] < hw["it_load_kw"])
        check(f"hw {name} pue >= 1", hw["pue"] >= 1.0)
    else:                                       # desktop profile: no PUE, PSU instead
        check(f"hw {name} has psu efficiency", 0.7 < hw["psu_efficiency"] <= 1.0)
        check(f"hw {name} has bandwidth", hw["memory_bandwidth_gb_s"] > 0)
        check(f"hw {name} declares no PUE", "pue" not in hw)

# the one thing that must never quietly become an all-in rate
sos = P["electricity"]["rates_usd_per_kwh"]["pepco_sos_price_to_compare_summer_2026"]
check("SOS rate flagged supply-only", "SUPPLY ONLY" in sos.get("note", ""))
check("default rate is not the SOS rate",
      P["electricity"]["default_rate"] != "pepco_sos_price_to_compare_summer_2026")

# --------------------------------------------------------------------------
print("2. independent recompute (shares no code with model.py)")

d = model("--all-scenarios", "--first-pass")
if not d:
    print("model.py failed; aborting")
    sys.exit(1)

hw = P["hardware"]["8xB200-HGX"]
mdl = P["models"]["deepseek-v4-pro-0813"]
rate = P["electricity"]["rates_usd_per_kwh"]["md_commercial_eia_aug2026"]["value"]
w = P["workload"]
mfu = P["prefill"]["mfu"]

# prefill, by hand
pre = 9e15 * mfu / (2 * 49e9) * 8
check("prefill tok/s", close(pre, d["hardware"]["prefill_input_tok_s"]),
      f"{pre:.0f} vs {d['hardware']['prefill_input_tok_s']:.0f}")

# decode, by hand
dec = 976 * 8
check("decode tok/s", close(dec, d["hardware"]["decode_output_tok_s"]))

# saturated batch, by hand
bat = pre * 2.5 * 3600
check("batch tok/night", close(bat, d["batch"]["saturated_input_tokens_per_night"]))
check("batch value", close(bat * 365 / 1e6 * 0.10, d["batch"]["value_at_batch_price_usd"]))

# per-scenario token volumes and API bills, by hand
for k, s in w["scenarios"].items():
    got = d["scenarios"][k]
    active = 800 * 0.55
    reqs = active * s["requests_per_active_user_per_day"]
    tin = reqs * s["input_tokens_per_request"]
    tout = reqs * s["output_tokens_per_request"]
    check(f"[{k}] input/day", close(tin, got["input_tokens_per_day"]))
    check(f"[{k}] output/day", close(tout, got["output_tokens_per_day"]))
    api = tin * 250 / 1e6 * 0.20 + tout * 250 / 1e6 * 1.20
    check(f"[{k}] API $/yr", close(api, got["api_interactive_usd"]), f"{api:.0f}")
    util = (reqs * 0.15 / 3600) * s["output_tokens_per_request"] / dec
    check(f"[{k}] peak util", close(util, got["peak_utilisation"]))

    # energy, by hand, with the serving/parked split spelled out
    mean_util = min(util * (1 / 0.15) / 11.5, 1.0)
    p_day = (hw["it_idle_kw"] + (hw["it_load_kw"] - hw["it_idle_kw"]) * (0.35 + 0.65 * mean_util)) * hw["pue"]
    p_bat = (hw["it_idle_kw"] + (hw["it_load_kw"] - hw["it_idle_kw"]) * (0.35 + 0.65 * 0.95)) * hw["pue"]
    p_idle = hw["it_idle_kw"] * hw["pue"]
    day_h, bat_h = 11.5 * 250, 2.5 * 365
    kwh = day_h * p_day + bat_h * p_bat + (8760 - day_h - bat_h) * p_idle
    check(f"[{k}] kWh/yr", close(kwh, got["selfhost"]["kwh_per_year"]), f"{kwh:.0f}")
    check(f"[{k}] power $/yr", close(kwh * rate, got["selfhost"]["power_usd"]))
    check(f"[{k}] total $/yr", close(450000 / 3 + kwh * rate + 120000, got["selfhost"]["total_usd"]))

# first-pass energy per Mtok on the rack, by hand
r = P["hardware"]["GB200-NVL72"]
rack_pre = 9e15 * mfu / (2 * 49e9) * 72
rack_kw = (r["it_idle_kw"] + (r["it_load_kw"] - r["it_idle_kw"]) * 1.0) * r["pue"]
usd = rack_kw * (1e6 / rack_pre) / 3600 * rate
check("NVL72 input $/Mtok",
      close(usd, d["first_pass_energy_only"]["GB200-NVL72"]["input"]["usd_per_mtok"]),
      f"{usd:.5f}")

# --------------------------------------------------------------------------
print("2b. independent recompute -- hourly / single-GPU")

hr = hourly()
if hr:
    ghw = P["hardware"]["RTX-5090-desktop"]
    q = P["models"]["qwen3.8-27b"]
    luna = P["api_prices_usd_per_mtok"]["luna_direct"]

    kw = (600 + 90 + 40) / 0.90 / 1000
    check("wall kW", close(kw, hr["config"]["wall_kw"]), f"{kw*1000:.0f} W")

    gb = 27.78e9 * 4.25 / 8 / 1e9
    check("weights GB", close(gb, hr["roofline"]["weights_gb"]))
    check("decode ceiling", close(1792 / gb, hr["roofline"]["ceiling_tok_s"]))
    # the premise MUST be flagged as above the ceiling -- if a param edit ever
    # makes it fit, the roofline warning silently stops firing
    check("premise is above the single-stream ceiling",
          hr["roofline"]["premise_over_ceiling"] > 1.0,
          f"{hr['roofline']['premise_over_ceiling']:.2f}x")

    check("prefill scaling", close(7200 * (8e9 / 27.78e9), hr["prefill_tok_s"]))
    be = (0.25 - 0.20) / (0.25 - 0.02)
    check("cache break-even hit rate", close(be, hr["cache_breakeven_hit_rate"]),
          f"{be:.3f}")

    for k, p in P["hourly"]["profiles"].items():
        got = hr["branches"]["premise"]["profiles"][k]
        R, h = p["billed_in_out_ratio"], p["cache_hit_rate"]
        fresh = (1 - h) * R if p["use_cache"] else R
        check(f"[{k}] fresh ratio", close(fresh, got["fresh_ratio"]))
        sus = 1 / (1 / 190 + fresh / (7200 * 8e9 / 27.78e9))
        check(f"[{k}] sustained tok/s", close(sus, got["sustained_tok_s"]), f"{sus:.1f}")
        eff = h * 0.02 + (1 - h) * 0.25 if p["use_cache"] else 0.20
        check(f"[{k}] effective input price", close(eff, got["effective_input_usd_per_mtok"]))
        api = sus * 3600 / 1e6 * 1.20 + sus * 3600 * R / 1e6 * eff
        check(f"[{k}] Luna $/hr", close(api, got["api_usd_per_hour"]), f"{api:.4f}")
        # prefill must actually cost something -- if this ever equals the
        # nominal decode rate, the contention term has been dropped
        check(f"[{k}] sustained < nominal decode", got["sustained_tok_s"] < 190)

    # the profile below the caching break-even must NOT be using the cache
    bulk = P["hourly"]["profiles"]["bulk"]
    check("sub-break-even profile has caching off",
          bulk["cache_hit_rate"] < be and not bulk["use_cache"])
else:
    check("hourly.py runs", False)

# --------------------------------------------------------------------------
print("3. prose vs model")

if RESULTS:
    def prose(pattern: str):
        m = re.search(pattern, RESULTS)
        return float(m.group(1).replace(",", "")) if m else None

    B = d["scenarios"]["B"]
    claims = {
        r"ANCHOR-input-usd-per-mtok:\s*([\d.]+)":
            d["first_pass_energy_only"]["GB200-NVL72"]["input"]["usd_per_mtok"],
        r"ANCHOR-power-usd-per-year:\s*([\d,]+)": B["selfhost"]["power_usd"],
        r"ANCHOR-selfhost-total-usd:\s*([\d,]+)": B["selfhost"]["total_usd"],
        r"ANCHOR-api-b-usd:\s*([\d,]+)": B["api_interactive_usd"],
        r"ANCHOR-batch-tok-night:\s*([\d.]+)":
            d["batch"]["saturated_input_tokens_per_night"] / 1e9,
        r"ANCHOR-batch-breakeven-tok-night:\s*([\d.]+)":
            B["breakeven"]["vs_total"]["batch_input_tokens_per_night"] / 1e9,
        r"ANCHOR-power-share-total:\s*([\d.]+)": B["selfhost"]["power_share_of_total"] * 100,
        r"ANCHOR-peak-util-b:\s*([\d.]+)": B["peak_utilisation"] * 100,
    }
    if hr:
        pr = hr["branches"]["premise"]["profiles"]
        claims.update({
            r"ANCHOR-5090-wall-w:\s*([\d,]+)": hr["config"]["wall_kw"] * 1000,
            r"ANCHOR-5090-ceiling:\s*([\d.]+)": hr["roofline"]["ceiling_tok_s"],
            r"ANCHOR-5090-power-res:\s*([\d.]+)":
                hr["electricity_usd_per_hour"]["md_residential_eia_corrected"],
            r"ANCHOR-cache-breakeven:\s*([\d.]+)": hr["cache_breakeven_hit_rate"] * 100,
            r"ANCHOR-agentic-sustained:\s*([\d.]+)": pr["agentic"]["sustained_tok_s"],
            r"ANCHOR-agentic-luna-hr:\s*([\d.]+)": pr["agentic"]["api_usd_per_hour"],
            r"ANCHOR-agentic-be-hday:\s*([\d.]+)": pr["agentic"]["breakeven_hours_per_day"],
            r"ANCHOR-chat-be-hday:\s*([\d.]+)": pr["chat"]["breakeven_hours_per_day"],
        })
    for pat, want in claims.items():
        got = prose(pat)
        check(f"prose {pat.split(':')[0][7:]}", got is not None and close(got, want, 0.02),
              f"RESULTS.md says {got}, model says {want:.4f}")
else:
    check("RESULTS.md exists", False, "not written yet")

# --------------------------------------------------------------------------
print("4. structural invariants")

for k, s in d["scenarios"].items():
    check(f"[{k}] nodes >= ceil(peak)",
          s["nodes_costed"] >= max(1, math.ceil(s["peak_utilisation"] - 1e-9)))
    check(f"[{k}] per-node peak <= 100%", s["peak_utilisation"] / s["nodes_costed"] <= 1.0)
    check(f"[{k}] power is a minority of TCO", s["selfhost"]["power_share_of_total"] < 0.15)
    check(f"[{k}] duty cycle < 100%", 0 < s["duty_cycle"] < 1.0)

# the headline structural claim: at 800 seats the batch route cannot close
cap = d["batch"]["saturated_input_tokens_per_night"]
need = d["scenarios"]["B"]["breakeven"]["vs_total"]["batch_input_tokens_per_night"]
check("batch break-even exceeds batch capacity at 800 seats", need > cap,
      f"need {need/1e9:.2f}B, capacity {cap/1e9:.2f}B")

# parked power must be strictly below serving-at-zero power, or the fix regressed
check("parked power < serving floor",
      d["hardware"]["kw_idle"] < (hw["it_idle_kw"] + (hw["it_load_kw"] - hw["it_idle_kw"]) * 0.35) * hw["pue"])

# an 8xB200 node is a bad decode engine vs the rack -- if this inverts, a
# throughput constant was edited without thinking
n8 = d["first_pass_energy_only"]["8xB200-HGX"]["output"]["usd_per_mtok"]
n72 = d["first_pass_energy_only"]["GB200-NVL72"]["output"]["usd_per_mtok"]
check("NVL72 beats 8xB200 on decode energy", n72 < n8, f"{n72:.5f} vs {n8:.5f}")

# --------------------------------------------------------------------------
print("5. negative controls")

# a 10x cheaper electricity rate must NOT flip the verdict at 800 seats --
# that is the whole finding, and if it flips, the model has lost the plot
cheap = model("--scenario", "B", "--rate", "0.0164")
check("10x cheaper power does not flip the verdict",
      cheap["scenarios"]["B"]["verdict"] == "API",
      f"got {cheap['scenarios']['B']['verdict']}")

# free electricity must not flip it either
free = model("--scenario", "B", "--rate", "0.0")
check("free power does not flip the verdict",
      free["scenarios"]["B"]["verdict"] == "API")

# but enough token volume MUST flip it, or nothing in the model has teeth
big = model("--scenario", "C", "--seats", "4000")
check("4000 seats at scenario C flips to self-host",
      big["scenarios"]["C"]["verdict"] == "self-host",
      f"got {big['scenarios']['C']['verdict']}")

# hourly: a cheaper box must lower the break-even duty cycle, and the street-vs-MSRP
# gap is the experiment's claim that the GPU shortage roughly doubles it
msrp = hourly("--capex", "3299")
if hr and msrp:
    a_street = hr["branches"]["premise"]["profiles"]["agentic"]["breakeven_hours_per_day"]
    a_msrp = msrp["branches"]["premise"]["profiles"]["agentic"]["breakeven_hours_per_day"]
    check("cheaper box lowers break-even duty cycle", a_msrp < a_street)
    check("street price ~doubles break-even vs MSRP-era box",
          1.6 <= a_street / a_msrp <= 2.2, f"ratio {a_street/a_msrp:.2f}")

# hourly: the conservative decode branch must raise break-even, never lower it --
# fewer tokens per hour buys less API spend for the same power
if hr:
    for k in P["hourly"]["profiles"]:
        p_hi = hr["branches"]["premise"]["profiles"][k]["breakeven_hours_per_day"]
        p_lo = hr["branches"]["conservative"]["profiles"][k]["breakeven_hours_per_day"]
        check(f"[{k}] conservative decode raises break-even", p_lo > p_hi)

# hourly: summer AC must raise cost per hour, never lower it
ac = hourly("--ac")
if hr and ac:
    check("AC raises wall draw", ac["config"]["wall_kw"] > hr["config"]["wall_kw"])
    check("AC raises break-even",
          ac["branches"]["premise"]["profiles"]["chat"]["breakeven_hours_per_day"]
          > hr["branches"]["premise"]["profiles"]["chat"]["breakeven_hours_per_day"])

# pointing the fleet model at a desktop profile must fail with guidance, not a
# KeyError traceback -- the two scales share params.json and will be confused
_r = subprocess.run([sys.executable, str(HERE / "model.py"),
                     "--hardware", "RTX-5090-desktop"], capture_output=True, text=True)
check("fleet model rejects desktop hardware cleanly",
      _r.returncode != 0 and "hourly.py" in (_r.stdout + _r.stderr)
      and "Traceback" not in _r.stderr,
      f"rc={_r.returncode}")

# and dropping ops to zero must lower, never raise, the self-host total
floor = model("--scenario", "B", "--ops", "0")
check("ops=0 lowers self-host total",
      floor["scenarios"]["B"]["selfhost"]["total_usd"]
      < d["scenarios"]["B"]["selfhost"]["total_usd"])

# N+1 must roughly double capex
n2 = model("--scenario", "B", "--nodes", "2")
check("N+1 doubles amortised capex",
      close(n2["scenarios"]["B"]["selfhost"]["capex_amortised_usd"],
            2 * d["scenarios"]["B"]["selfhost"]["capex_amortised_usd"]))

# --------------------------------------------------------------------------
print()
if FAILS:
    print(f"FAIL {len(FAILS)}/{N}")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print(f"OK {N}/{N} checks")
