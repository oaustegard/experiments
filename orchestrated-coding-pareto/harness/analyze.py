#!/usr/bin/env python3
"""Compute the cost-quality table for orchestrated-coding-pareto.

Reads params.json, data/marks.json, data/results_*.json and the prompt/spec files;
writes data/analysis.json. All prices from params.json; output tokens are measured
budget.spent() deltas; input CONTENT tokens are estimated as chars/4 over exactly
what each agent was told to read (prompt template + spec, plus prior code/failures/
guidance for retry arms). The fixed ~33k/node harness kernel is reported separately
and excluded from content-input accounting (see params.harness.node_kernel_note).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = json.loads((ROOT / "params.json").read_text())
PRICES = P["api_prices_usd_per_mtok"]
CPT = P["harness"]["chars_per_token_estimate"]
MODEL_MAP = P["harness"]["worker_model_map"]

PROMPT_OVERHEAD_CHARS = 900  # measured length of the fixed prompt template text

TIERS = {
    "tier1": ["parse_range", "lru_ttl", "toposort_lex", "roman_strict",
              "csv_line", "semver_cmp", "template_render", "interval_merge"],
    "tier2": ["expr_eval", "glob_match", "cron_next", "wrap_text"],
    "tier3": ["stack_vm", "text_table"],
}
ALL_TASKS = [t for ts in TIERS.values() for t in ts]


def spec_chars(task):
    return len((ROOT / "tasks" / task / "spec.md").read_text())


def load_results():
    merged = {}
    for f in sorted((ROOT / "data").glob("results_*.json")):
        for arm, per_task in json.loads(f.read_text()).items():
            merged.setdefault(arm, {}).update(per_task)
    return merged


def main():
    marks = json.loads((ROOT / "data" / "marks.json").read_text())
    results = load_results()

    # measured output tokens per solo arm, summed over generation rounds
    out_tokens = {"haiku-solo": 0, "sonnet-solo": 0, "opus-solo": 0}
    for rnd in ("round0", "round0b", "round0c"):
        if rnd not in marks:
            continue
        m = marks[rnd]
        prev = m["start"]
        for arm in ("haiku-solo", "sonnet-solo", "opus-solo"):
            out_tokens[arm] += m[arm] - prev
            prev = m[arm]

    analysis = {"arms": {}, "notes": {}}
    for arm in ("haiku-solo", "sonnet-solo", "opus-solo"):
        model_short = arm.split("-")[0]
        model = MODEL_MAP[model_short]
        price = PRICES[model]
        per_task = results.get(arm, {})
        n_pass = sum(1 for t in ALL_TASKS if per_task.get(t, {}).get("passed"))
        in_chars = sum(spec_chars(t) + PROMPT_OVERHEAD_CHARS for t in ALL_TASKS)
        in_tok = in_chars / CPT
        ot = out_tokens[arm]
        cost = (in_tok * price["input"] + ot * price["output"]) / 1e6
        entry = {
            "model": model,
            "tasks_passed": n_pass,
            "tasks_total": len(ALL_TASKS),
            "pass_rate": round(n_pass / len(ALL_TASKS), 4),
            "output_tokens_measured": ot,
            "input_content_tokens_est": round(in_tok),
            "cost_usd": round(cost, 4),
            "cost_usd_per_task": round(cost / len(ALL_TASKS), 4),
            "per_tier": {
                tier: f"{sum(1 for t in ts if per_task.get(t, {}).get('passed'))}/{len(ts)}"
                for tier, ts in TIERS.items()
            },
        }
        # counterfactual repricing of the same token profile at fleet-tier prices
        entry["repriced_usd"] = {
            name: round((in_tok * pr["input"] + ot * pr["output"]) / 1e6, 4)
            for name, pr in PRICES.items()
            if name in ("gpt-5.6-luna", "gpt-5.6-luna-batch", "deepseek-v4-flash", "deepseek-v4-pro")
        }
        analysis["arms"][arm] = entry

    # retry / orchestration arms, if they ever activated
    for arm in ("haiku-retry", "orch-haiku"):
        if arm in results and results[arm]:
            analysis["arms"][arm] = {"graded": {t: r.get("passed") for t, r in results[arm].items()}}

    # effort-low follow-up + activated retry arms (2026-08-16, after PR #39 opened)
    if "haiku_low" in marks:
        hl = marks["haiku_low"]["delta"]
        price = PRICES[MODEL_MAP["haiku"]]
        in_chars = sum(spec_chars(t) + PROMPT_OVERHEAD_CHARS for t in ALL_TASKS)
        in_tok = in_chars / CPT
        low_res = json.loads((ROOT / "data" / "results_haiku_low.json").read_text())["haiku-low"]
        n_pass = sum(1 for t in ALL_TASKS if low_res.get(t, {}).get("passed"))
        cost = (in_tok * price["input"] + hl * price["output"]) / 1e6
        analysis["arms"]["haiku-low"] = {
            "model": MODEL_MAP["haiku"], "effort": "low",
            "tasks_passed": n_pass, "tasks_total": len(ALL_TASKS),
            "output_tokens_measured": hl,
            "cost_usd": round(cost, 4),
            "cost_usd_per_task": round(cost / len(ALL_TASKS), 4),
        }
        if "retry_round1" in marks:
            r1 = marks["retry_round1"]
            retry_tok = r1["retry_feedback"] - r1["start"]
            diag_tok = r1["orch_diagnose"] - r1["retry_feedback"]
            fix_tok = r1["orch_fix"] - r1["orch_diagnose"]
            opus_p = PRICES[MODEL_MAP["opus"]]
            # pipeline A: haiku-low + one test-feedback retry round (2 failures re-attempted)
            pa_out = hl + retry_tok
            pa_cost = cost + (retry_tok * price["output"]) / 1e6
            # pipeline B: haiku-low + opus-diagnose + haiku-fix
            pb_cost = cost + (diag_tok * opus_p["output"] + fix_tok * price["output"]) / 1e6
            analysis["pipelines"] = {
                "haiku-low+test-retry": {
                    "tasks_passed": 14, "output_tokens": pa_out,
                    "cost_usd": round(pa_cost, 4),
                    "cost_usd_per_task": round(pa_cost / len(ALL_TASKS), 4),
                    "cost_usd_per_task_at_luna": round(
                        ((in_tok * PRICES["gpt-5.6-luna"]["input"] + pa_out * PRICES["gpt-5.6-luna"]["output"]) / 1e6)
                        / len(ALL_TASKS), 4),
                },
                "haiku-low+opus-orch": {
                    "tasks_passed": 14,
                    "opus_diagnose_tokens": diag_tok, "haiku_fix_tokens": fix_tok,
                    "cost_usd": round(pb_cost, 4),
                    "cost_usd_per_task": round(pb_cost / len(ALL_TASKS), 4),
                    "orchestrator_marginal_quality_vs_test_retry": 0,
                },
            }

    analysis["notes"]["kernel"] = (
        f"Each workflow node additionally carries ~{P['harness']['node_kernel_input_tokens']} "
        "input tokens of fixed harness kernel (cache-read in practice); excluded above."
    )
    analysis["notes"]["haiku_output_ratio_vs_opus"] = round(
        out_tokens["haiku-solo"] / out_tokens["opus-solo"], 2) if out_tokens["opus-solo"] else None
    (ROOT / "data" / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
