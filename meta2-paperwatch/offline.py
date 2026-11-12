#!/usr/bin/env python3
"""Offline stage: Haiku reviews of the offline split, with and without verifier access, then one
Haiku call (the paper uses one base model throughout) that distills the contrasts into a frozen Meta-Skill (<= 400 words).

  plain    : the baseline reviewer prompt, two runs per paper (setting "a")
  informed : the same prompt plus the gold rating and key numbers (setting "c", verifier access)

Only offline-split papers and their gold labels reach the distiller. Writes meta_skill.md,
cache/offline/*.json and results/offline.json.
"""
import json, re
from concurrent.futures import ThreadPoolExecutor

import common as c

OUT = c.HERE / "cache" / "offline"
WORKERS = 8
MAX_WORDS = 400
DISTILL_MODEL = "haiku"  # paper-faithful: one base model for every stage (Opus hit the account weekly limit)


def review(pid, kind, k, gold):
    dst = OUT / f"{pid}.{kind}{k}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    user = c.paper_block(pid)
    if kind == "informed":
        g = gold[pid]
        user += "\n\n" + c.INFORMED_NOTE.format(rating=g["practical"], numbers=g["key_numbers"])
    r = None
    for _ in range(2):
        r = c.call("haiku", c.reviewer_system(), user)
        if r and c.parse_rating(r["text"]):
            break
    rec = {"pid": pid, "kind": kind, "run": k, "call": r, "rating": c.parse_rating(r["text"]) if r else None}
    c.save(dst, rec)
    print(pid, kind, k, rec["rating"], "gold", gold[pid]["practical"], flush=True)
    return rec


DISTILL_SYSTEM = """You build a **Meta-Skill**: a short, frozen library of guidance for a cheap reviewer model (and \
for a supervisor that checks its drafts) that rates arXiv papers for one reader's practical use. You learn it by \
contrasting reviews of training papers, following the Meta 2 method (arXiv 2611.05139), condensed to one call.

The reader's profile, which every review rates against:

{profile}

For each training paper you get the checked gold rating with the numbers it rests on, two reviews written with the \
plain reviewer prompt (setting **a**), and one review written by the same model given the gold rating and key \
numbers (setting **c**, verifier access that the deployed reviewer will never have).

Contrast directions (each is directional evidence, not a causal estimate):
- **act**: a missed the gold rating and c hit it. What did c check or weigh, visible in the paper itself, that a \
could have done without the gold label? A lesson whose only basis is knowing the answer is not allowed.
- **avoid**: a reasoning move that appears in reviews that missed (in a or c): a specific inference that led the \
rating astray, such as treating topic overlap with the profile as usefulness, or a model card as evidence.
- **hold**: a hit the gold in both runs. Name the situation where the plain reviewer is already right, so that \
no check pushes those papers off their rating.

Rules, from the Meta 2 aggregation step:
1. Every entry must be supported by at least two distinct training papers. Drop anything else.
2. `When` names one situation the reviewer can recognise from the paper text or its own draft. No paper title, \
model name, dataset, library or number in `When`, `Guidance` or `Why`; those may appear only in `Seen in`.
3. `Guidance` is one action: a check to run on the paper's text, or a rating move, and what evidence licenses it.
4. Name the rating errors explicitly (direction: inflation or deflation; which rating boundary).
5. The whole Meta-Skill is at most {max_words} words including headings.

Output exactly one JSON object and nothing else:
```json
{{"analysis": "<the error pattern across papers, with paper ids and the a/c/gold ratings>", "meta_skill": "<markdown: entries `### E-NN — <title>` with lines **Type.** act|avoid|hold, **When.**, **Guidance.**, **Why.**, **Seen in.** <paper ids>>", "dropped": ["<candidate lessons dropped, with reason>"]}}
```"""


def fmt_case(pid, gold, recs):
    g = gold[pid]
    parts = [f"## Paper {pid}: {g['title']}",
             f"Gold: **{g['practical']}**. Key numbers: {g['key_numbers']}. Code/weights: {g['code_weights']}. Why: {g['why']}"]
    for r in recs:
        tag = "a" if r["kind"] == "plain" else "c"
        parts.append(f"### Setting {tag} review (run {r['run']}), rated {r['rating']}\n{(r['call'] or {}).get('text', '(failed)')}")
    return "\n\n".join(parts)


def main():
    gold = json.load(open(c.HERE / "gold.json"))
    offline, test = c.split()
    offline = [p for p in offline if gold.get(p)]
    jobs = [(p, kind, k) for p in offline for kind, k in (("plain", 1), ("plain", 2), ("informed", 1))]
    with ThreadPoolExecutor(WORKERS) as ex:
        recs = list(ex.map(lambda a: review(*a, gold), jobs))
    by = {p: [r for r in recs if r["pid"] == p] for p in offline}

    summary = []
    for p in offline:
        g = gold[p]["practical"]
        summary.append({"pid": p, "gold": g, "plain": [r["rating"] for r in by[p] if r["kind"] == "plain"],
                        "informed": [r["rating"] for r in by[p] if r["kind"] == "informed"]})
    plain_hits = sum(r == s["gold"] for s in summary for r in s["plain"])
    inf_hits = sum(r == s["gold"] for s in summary for r in s["informed"])

    meta = c.HERE / "meta_skill.md"
    dcall_path = c.HERE / "cache" / "offline" / "distill.json"
    if not dcall_path.exists():
        user = "# Training papers\n\n" + "\n\n".join(fmt_case(p, gold, by[p]) for p in offline)
        system = DISTILL_SYSTEM.format(profile=c._section(c.PROCEDURE.read_text(), "Profile for the subagents"),
                                       max_words=MAX_WORDS)
        attempts = []
        for _ in range(3):
            r = c.call(DISTILL_MODEL, system, user, timeout=1800)
            j = c.parse_json(r["text"]) if r else None
            attempts.append({"call": r, "parsed": j})
            if j and j.get("meta_skill") and len(j["meta_skill"].split()) <= MAX_WORDS:
                break
            user += "\n\n(Your previous answer was not valid JSON or exceeded the word limit. Answer again within the rules.)"
        c.save(dcall_path, {"attempts": attempts, "system": system, "user_chars": len(user)})
    d = json.loads(dcall_path.read_text())
    j = d["attempts"][-1]["parsed"]
    meta.write_text(j["meta_skill"].strip() + "\n")
    out = {"offline_ids": offline, "test_ids": test, "per_paper": summary,
           "plain_exact": plain_hits / (2 * len(offline)), "informed_exact": inf_hits / len(offline),
           "meta_skill_words": len(j["meta_skill"].split()), "distill_attempts": len(d["attempts"]),
           "distill_analysis": j.get("analysis"), "dropped": j.get("dropped"),
           "tokens": {"reviews": sum(c.tokens(r["call"]) for r in recs),
                      "distill": sum(c.tokens(a["call"]) for a in d["attempts"])},
           "cost_usd": {"reviews": sum((r["call"] or {}).get("cost_usd") or 0 for r in recs),
                        "distill": sum((a["call"] or {}).get("cost_usd") or 0 for a in d["attempts"])}}
    c.save(c.HERE / "results" / "offline.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_paper", "distill_analysis", "dropped")}, indent=1))


if __name__ == "__main__":
    main()
