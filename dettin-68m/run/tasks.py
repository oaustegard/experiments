"""Build the headless-child task files (claude -p through headless_fanout.py) and collect their output.

    python3 run/tasks.py pick                      # choose gold, query-train and query-eval sections -> data/picks.json
    python3 run/tasks.py queries train|eval        # -> out/tasks/queries_<set>.jsonl
    python3 run/tasks.py gold sonnet|opus|haiku    # -> out/tasks/gold_<judge>.jsonl (section questions)
    python3 run/tasks.py goldrel sonnet|opus       # -> out/tasks/goldrel_<judge>.jsonl (query, section pairs)
    python3 run/tasks.py collect NAME              # out/fan/NAME/results/*.json -> data/NAME.jsonl

Every child gets its inputs inline, no tools, and returns structured output checked against a JSON
schema. The shared instructions sit in the system prompt, which is the part the children can cache.
"""
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import REL_Q, SECTION_QS, options

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "out" / "tasks"
STYLES = ("question", "keywords", "claim")
N_GOLD, N_EVAL_Q, PER_ARTICLE = 400, 1200, 4

QUERY_SYS = """You write search queries for a retrieval benchmark over sections of life-science research articles.

You get numbered sections, each with a style. For each one, write ONE query that the section answers and that a researcher who has not seen the section could plausibly type.

Styles:
- question: a natural-language question, 8 to 20 words.
- keywords: a search-box query of 3 to 8 words, no punctuation, not a sentence.
- claim: one factual statement that the section supports, 8 to 25 words.

Rules:
- Be specific. Name the organism, condition, molecule, method or population, so that this section is the best answer and not just any paper on the topic.
- Do not copy more than four consecutive words from the section.
- Do not refer to "the section", "the study", "the authors" or "this paper".
- Use only what the section says.
- Ask about what the section says, not about what kind of section it is."""


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16)


def sections():
    with gzip.open(ROOT / "out" / "sections.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def clip(text, words):
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " ..."


def pick():
    rows = sections()
    rng = random.Random(0)
    by_art = {}
    for r in rows:
        by_art.setdefault(r["pmcid"], []).append(r)
    picks = {"gold": [], "query_eval": [], "query_train": []}
    arts = sorted(by_art)
    rng.shuffle(arts)
    for a in arts:
        secs = by_art[a][:]
        rng.shuffle(secs)
        split = secs[0]["split"]
        if split == "train":
            picks["query_train"] += [s["sid"] for s in secs[:PER_ARTICLE]]
        else:
            picks["query_eval"] += [s["sid"] for s in secs[:PER_ARTICLE]]
            if split == "test":
                picks["gold"] += [s["sid"] for s in secs[:2]]
    picks["gold"] = picks["gold"][:N_GOLD]
    picks["query_eval"] = picks["query_eval"][:N_EVAL_Q]
    (ROOT / "data" / "picks.json").write_text(json.dumps(picks))
    print({k: len(v) for k, v in picks.items()})


def write_tasks(name, tasks):
    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / f"{name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (TASKS / f"{name}.map.json").write_text(json.dumps({t["id"]: t["_ids"] for t in tasks}))
    n = sum(len(t["_ids"]) for t in tasks)
    print(f"{name}: {len(tasks)} tasks, {n} items, about {sum(len(t['prompt']) for t in tasks) // 4:,} prompt tokens")


def queries(which, per_child=14, model=None):
    rows = {r["sid"]: r for r in sections()}
    ids = json.loads((ROOT / "data" / "picks.json").read_text())[f"query_{which}"]
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["n", "query"],
        "properties": {"n": {"type": "integer"}, "query": {"type": "string"}}}}}}
    tasks = []
    for i in range(0, len(ids), per_child):
        chunk = ids[i: i + per_child]
        parts = []
        for n, sid in enumerate(chunk, 1):
            parts.append(f"### Section {n} (style: {STYLES[h(sid) % 3]})\n{clip(rows[sid]['text'], 600)}")
        prompt = (f"Write one query for each of the {len(chunk)} sections below, in the style named for it. "
                  f"Return items with n = the section number.\n\n" + "\n\n".join(parts))
        tasks.append({"id": f"q{which}-{i // per_child:05d}", "prompt": prompt, "system_prompt": QUERY_SYS,
                      "json_schema": schema, "tools": "", "_ids": chunk})
    write_tasks(f"queries_{which}", tasks)


def rubric(qs):
    out = []
    for name, q in qs.items():
        if q["type"] == "noul":
            out.append(f"- {name} (true or false): {q['instructions']}")
        elif isinstance(q["criteria"], list):
            lv = "\n".join(f"    {i} = {d}" for i, d in enumerate(q["criteria"]))
            out.append(f"- {name} (integer level): {q['instructions']}\n{lv}")
        else:
            op = "\n".join(f"    {k} = {d}" for k, d in q["criteria"].items())
            out.append(f"- {name} (one option key): {q['instructions']}\n{op}")
    return "\n".join(out)


def item_schema(qs):
    props = {"n": {"type": "integer"}}
    for name, q in qs.items():
        if q["type"] == "noul":
            props[name] = {"type": "boolean"}
        elif isinstance(q["criteria"], list):
            props[name] = {"type": "integer", "enum": list(range(len(q["criteria"])))}
        else:
            props[name] = {"type": "string", "enum": options(q)}
    return {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": list(props), "properties": props}}}}


GOLD_SYS = ("You label sections of life-science research articles for a benchmark. Judge each section on its own "
            "text alone; its heading has been removed. Answer every question for every section. When two options "
            "both fit, choose the one that covers more of the section.\n\nQuestions:\n")


def gold(judge, per_child=5):
    rows = {r["sid"]: r for r in sections()}
    ids = json.loads((ROOT / "data" / "picks.json").read_text())["gold"]
    tasks = []
    for i in range(0, len(ids), per_child):
        chunk = ids[i: i + per_child]
        parts = [f"### Section {n}\n{rows[sid]['text']}" for n, sid in enumerate(chunk, 1)]
        prompt = (f"Label each of the {len(chunk)} sections below. Return items with n = the section number.\n\n"
                  + "\n\n".join(parts))
        tasks.append({"id": f"gold-{judge}-{i // per_child:04d}", "prompt": prompt,
                      "system_prompt": GOLD_SYS + rubric(SECTION_QS), "json_schema": item_schema(SECTION_QS),
                      "tools": "", "model": judge, "_ids": chunk})
    write_tasks(f"gold_{judge}", tasks)


def goldrel(judge, per_child=8):
    with open(ROOT / "data" / "goldrel_pairs.jsonl") as f:
        pairs = [json.loads(line) for line in f]
    sys_p = ("You grade search results for a benchmark over sections of life-science research articles. For each "
             "numbered pair, grade how well the section answers the query. Judge each pair on its own.\n\n"
             "Question:\n" + rubric(REL_Q))
    tasks = []
    for i in range(0, len(pairs), per_child):
        chunk = pairs[i: i + per_child]
        parts = [f"### Pair {n}\nQuery: {p['query']}\nSection: {p['text']}" for n, p in enumerate(chunk, 1)]
        prompt = (f"Grade each of the {len(chunk)} pairs below. Return items with n = the pair number.\n\n"
                  + "\n\n".join(parts))
        tasks.append({"id": f"goldrel-{judge}-{i // per_child:04d}", "prompt": prompt, "system_prompt": sys_p,
                      "json_schema": item_schema(REL_Q), "tools": "", "model": judge,
                      "_ids": [p["id"] for p in chunk]})
    write_tasks(f"goldrel_{judge}", tasks)


def collect(name):
    """Join each child's structured output back to the ids it was given; a child that returned the wrong
    number of items, or item numbers that are not 1..k, is reported and left out."""
    amap = json.loads((TASKS / f"{name}.map.json").read_text())
    out, bad = [], []
    for tid, ids in amap.items():
        f = ROOT / "out" / "fan" / name / "results" / f"{tid}.json"
        if not f.exists():
            bad.append((tid, "missing"))
            continue
        res = json.loads(f.read_text())
        items = (res.get("structured_output") or {}).get("items") or []
        ns = sorted(it.get("n") for it in items)
        if ns != list(range(1, len(ids) + 1)):
            bad.append((tid, f"items {ns} for {len(ids)} inputs"))
            continue
        for it in items:
            out.append({"id": ids[it["n"] - 1], **{k: v for k, v in it.items() if k != "n"}})
    (ROOT / "data" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{name}: {len(out)} items from {len(amap) - len(bad)} tasks; {len(bad)} tasks left out {bad[:5]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pick":
        pick()
    elif cmd == "queries":
        queries(sys.argv[2], per_child=int(sys.argv[3]) if len(sys.argv) > 3 else 14)
    elif cmd == "gold":
        gold(sys.argv[2])
    elif cmd == "goldrel":
        goldrel(sys.argv[2])
    elif cmd == "collect":
        collect(sys.argv[2])
