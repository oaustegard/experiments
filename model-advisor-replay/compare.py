"""Aggregate the advisor's archived verdicts and the two rubric replays into results/summary.json.

    python3 compare.py notes.tsv rows.jsonl replay_0928.jsonl replay_1007.jsonl > results/summary.json

replay_0928 is `model_advisor.py` at claude-workspace bb35fbb (the 2026-09-28 rubric), replay_1007 the
2026-10-07 rubric (72b146d); both run as `replay rows.jsonl --from sonnet`. Only counts leave this
script: the prompts are private.
"""
import json
import statistics
import sys
from collections import Counter

notes_path, rows_path, old_path, new_path = sys.argv[1:5]
TIERS = ("haiku", "sonnet", "opus", "fable")


def tier(model):
    return next((t for t in TIERS if t in (model or "")), None)


def kind(note):
    if not note:
        return "no-note"
    if "classifier failed" in note:
        return "error"
    if "unknown model" in note:
        return "blind"
    return "model-known"


notes = [line.rstrip("\n").split("\t") for line in open(notes_path)]
by_era = {}
for _, date, note in notes:
    era = "2026-10-03..10-08" if date >= "2026-10-03" else "2026-09-28..10-02"
    if date < "2026-09-28":
        continue
    by_era.setdefault(era, Counter())[kind(note)] += 1

rows = {r["session"]: r for r in map(json.loads, open(rows_path))}
old = {r["session"]: r for r in map(json.loads, open(old_path))}
new = {r["session"]: r for r in map(json.loads, open(new_path))}
assert old.keys() == new.keys()


def p_opus_up(r):
    return r["tier"]["opus"] + r["tier"]["fable"]


def tally(rep):
    return {
        "decisions": dict(Counter(f"{r['suggest'] or 'silent'}/{r['rule']}" for r in rep.values())),
        "top_tier": dict(Counter(max(r["tier"], key=r["tier"].get) for r in rep.values())),
        "fires_by_model_actually_used": dict(Counter(
            f"{r['suggest'] or 'silent'} on {tier((rows[s]['model_seq'] or [None])[0])}" for s, r in rep.items())),
    }


shift = [p_opus_up(new[s]) - p_opus_up(old[s]) for s in old]
upgrades = [s for s, r in rows.items() if s in old and len(r["model_seq"]) > 1
            and TIERS.index(tier(r["model_seq"][-1])) > TIERS.index(tier(r["model_seq"][0]))]
summary = {
    "archive_verdicts": {era: dict(c) for era, c in sorted(by_era.items())},
    "replay": {
        "sessions": len(old),
        "start_model_actually_used": dict(Counter(tier((rows[s]["model_seq"] or [None])[0]) for s in old)),
        "rubric_2026_09_28": tally(old),
        "rubric_2026_10_07": tally(new),
        "p_opus_or_fable_shift": {"mean": round(statistics.mean(shift), 3),
                                  "median": round(statistics.median(shift), 3)},
        "mid_session_upgrades": [{"session": s, "models": rows[s]["model_seq"],
                                  "p_opus_or_fable_0928": round(p_opus_up(old[s]), 2),
                                  "p_opus_or_fable_1007": round(p_opus_up(new[s]), 2),
                                  "rule_1007": new[s]["rule"]} for s in upgrades],
    },
}
print(json.dumps(summary, indent=2))
