"""Check the numbers RESULTS.md states against results/summary.json. Runs in well under a second."""
import json
from pathlib import Path

HERE = Path(__file__).parent
s = json.loads((HERE / "results/summary.json").read_text())
text = (HERE / "RESULTS.md").read_text()
r = s["replay"]
old, new = r["rubric_2026_09_28"], r["rubric_2026_10_07"]
late = s["archive_verdicts"]["2026-10-03..10-08"]
early = s["archive_verdicts"]["2026-09-28..10-02"]
up = r["mid_session_upgrades"]

checks = {
    "74 sessions": r["sessions"] == 74,
    "early row": f"| 2026-09-28 to 10-02 | {early.get('model-known', 0)} | {early.get('blind', 0)} | "
                 f"{early.get('error', 0)} | {early.get('no-note', 0)} |" in text,
    "late row": f"| 2026-10-03 to 10-08 | {late['model-known']} | {late['blind']} | {late['error']} | "
                f"{late['no-note']} |" in text,
    "1 of 22": f"{late['model-known']}\nof {late['model-known'] + late['blind']} classified" in text,
    "suggests Opus": f"| suggests Opus | {old['decisions']['opus/tier']} | {new['decisions']['opus/tier']} |" in text,
    "opaque": f"| {old['decisions']['silent/opaque']} | {new['decisions']['silent/opaque']} |" in text,
    "unsure": f"below 0.75 | {old['decisions']['silent/unsure']} | {new['decisions']['silent/unsure']} |" in text,
    "agrees": f"Sonnet is enough | {old['decisions']['silent/agrees']} | {new['decisions']['silent/agrees']} |" in text,
    "cheaper": f"Haiku looks enough | {old['decisions']['silent/cheaper']} | {new['decisions']['silent/cheaper']} |" in text,
    "top opus": f"is Opus | {old['top_tier']['opus']} | {new['top_tier']['opus']} |" in text,
    "top sonnet": f"is Sonnet | {old['top_tier']['sonnet']} | {new['top_tier']['sonnet']} |" in text,
    "shift": f"{r['p_opus_or_fable_shift']['mean']:.2f} on average (median {r['p_opus_or_fable_shift']['median']:.2f})" in text,
    "one upgrade": len(up) == 1 and up[0]["session"] == "7c199534",
    "7c199534": f"from {up[0]['p_opus_or_fable_0928']:.2f} to {up[0]['p_opus_or_fable_1007']:.2f}" in text
                and up[0]["rule_1007"] == "unsure",
    "start models": all(f"{n}{' of the 74 sessions' if m == 'opus' else ''} on {m.capitalize()}"
                        in text.replace("\n", " ") for m, n in r["start_model_actually_used"].items()),
    "fires by model": [new["fires_by_model_actually_used"].get(f"opus on {m}", 0) for m in ("opus", "sonnet", "fable", "haiku")]
                      == [14, 5, 2, 1] and old["fires_by_model_actually_used"]["opus on sonnet"] == 4,
}
bad = [k for k, ok in checks.items() if not ok]
print("recheck:", "OK" if not bad else f"FAILED {bad}", f"({len(checks)} checks)")
raise SystemExit(1 if bad else 0)
