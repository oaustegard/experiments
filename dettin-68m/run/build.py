"""Turn labels into training files.

    python3 run/build.py cases     # -> out/upload/cases_{haiku,jev}.jsonl (train_enc.py --data), data/gold.json
    python3 run/build.py emb       # -> out/upload/emb_data.json (pairs, eval queries, eval pool, distill ids)

Cases. A training case is one section with every section question; its gold is the bulk teacher's answer
(Haiku one-hot labels, or Jev's distributions), except `role`, which comes from the article's own heading
(sections under a combined "Results and discussion" heading carry no role question). A relevance case is
one (query, candidate) pair with Jev's distribution over the four levels. Test cases are the gold
sections and gold pairs, with gold = the mean of Sonnet's and Opus's one-hot answers; data/gold.json
keeps both judges' labels so the scorer can restrict to items they agree on. N_SEC sections and N_REL
pairs are kept for training, by seeded sample.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import (
    REL_Q,
    SECTION_QS,
    options,
    relevance_state,
    section_state,
)
from tasks import ROOT, STYLES, h, sections

N_SEC, N_REL = 8000, 16000
ROLE = {"introduction": "introduction", "methods": "methods", "results": "results", "discussion": "discussion",
        "conclusion": "discussion", "other": "other"}
UP = ROOT / "out" / "upload"


def lab(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def onehot(q, key):
    return {o: float(o == key) for o in options(q)}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def jev_table(path):
    return {r["id"]: r["probs"] for r in read(path) if "probs" in r}


def cases():
    rows = {r["sid"]: r for r in sections()}
    picks = json.loads((ROOT / "data" / "picks.json").read_text())
    jev = jev_table(ROOT / "out" / "jev_sections.jsonl")
    haiku = {r["id"]: r for r in read(ROOT / "data" / "label_haiku.jsonl")}
    jrel = jev_table(ROOT / "out" / "jev_pairs.jsonl")
    gs = {r["id"]: r for r in read(ROOT / "data" / "gold_sonnet.jsonl")}
    go = {r["id"]: r for r in read(ROOT / "data" / "gold_opus.jsonl")}
    rng = random.Random(0)
    UP.mkdir(parents=True, exist_ok=True)

    def sec_case(sid, gold, split):
        r = rows[sid]
        qs = {k: v for k, v in SECTION_QS.items() if k in gold}
        return {"id": sid, "workflow": "ls_section", "account": r["pmcid"], "state": section_state(r["text"]),
                "questions": qs, "gold": {k: {"probabilities": gold[k]} for k in qs}, "split": split}

    def rel_case(pid, split, gold):
        _, qid, sid = pid.split("|")
        return {"id": pid, "workflow": "ls_relevance", "account": qid.split(":")[0],
                "state": relevance_state(QUERY[pid], rows[sid]["text"]), "questions": REL_Q,
                "gold": {"relevance": {"probabilities": gold}}, "split": split}

    QUERY = {r["id"]: r["state"]["query"] for r in read(ROOT / "out" / "jev_pairs_in.jsonl")}
    train_sids = [s for s in picks["query_train"] if s in haiku and s in jev]
    rng.shuffle(train_sids)
    train_sids = train_sids[:N_SEC]
    rel_ids = sorted(p for p in jrel if p.startswith("train|"))
    rng.shuffle(rel_ids)
    rel_ids = rel_ids[:N_REL]

    # test: gold sections and gold pairs
    test, gold_out = [], {"sections": {}, "pairs": {}}
    for sid in picks["gold"]:
        if sid not in gs or sid not in go:
            continue
        g = {}
        for k, q in SECTION_QS.items():
            a, b = onehot(q, lab(gs[sid][k])), onehot(q, lab(go[sid][k]))
            g[k] = {o: (a[o] + b[o]) / 2 for o in a}
        role = ROLE.get(rows[sid]["role"])
        gold_out["sections"][sid] = {"sonnet": {k: lab(gs[sid][k]) for k in SECTION_QS},
                                     "opus": {k: lab(go[sid][k]) for k in SECTION_QS}, "heading_role": role}
        if role:
            g["role"] = onehot(SECTION_QS["role"], role)
        else:
            g.pop("role")
        test.append(sec_case(sid, g, "test"))
    rs = {r["id"]: r for r in read(ROOT / "data" / "goldrel_sonnet.jsonl")}
    ro = {r["id"]: r for r in read(ROOT / "data" / "goldrel_opus.jsonl")}
    for pid in rs:
        if pid not in ro:
            continue
        a, b = onehot(REL_Q["relevance"], lab(rs[pid]["relevance"])), onehot(REL_Q["relevance"], lab(ro[pid]["relevance"]))
        test.append(rel_case(pid, "test", {o: (a[o] + b[o]) / 2 for o in a}))
        gold_out["pairs"][pid] = {"sonnet": lab(rs[pid]["relevance"]), "opus": lab(ro[pid]["relevance"])}
    (ROOT / "data" / "gold.json").write_text(json.dumps(gold_out))

    for teacher in ("haiku", "jev"):
        out = []
        for sid in train_sids:
            if teacher == "haiku":
                g = {k: onehot(q, lab(haiku[sid][k])) for k, q in SECTION_QS.items()}
            else:
                g = {k: dict(jev[sid][k]) for k in SECTION_QS}
            role = ROLE.get(rows[sid]["role"])
            if role:
                g["role"] = onehot(SECTION_QS["role"], role)
            else:
                g.pop("role")
            out.append(sec_case(sid, g, "train"))
        out += [rel_case(p, "train", jrel[p]["relevance"]) for p in rel_ids]
        rng2 = random.Random(1)
        rng2.shuffle(out)
        (UP / f"cases_{teacher}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in out + test))
        print(f"cases_{teacher}: {len(out)} train cases ({len(train_sids)} sections, {len(rel_ids)} pairs), {len(test)} test cases")


def emb():
    rows = {r["sid"]: r for r in sections()}
    jrel = jev_table(ROOT / "out" / "jev_pairs.jsonl")
    rng = random.Random(0)

    def exp_rel(pid):
        p = (jrel.get(pid) or {}).get("relevance")
        return sum(int(k) * v for k, v in p.items()) if p else None

    train_pairs, dropped, kept_neg = [], 0, 0
    for r in read(ROOT / "out" / "pairs_train.jsonl"):
        negs = []
        for c in r["cands"][1:]:
            e = exp_rel(f"train|{r['qid']}|{c['sid']}")
            if e is not None and e >= 1.5:      # Jev says this "negative" answers the query: not a negative
                dropped += 1
                continue
            negs.append(c["sid"])
        kept_neg += len(negs)
        train_pairs.append({"query": r["query"], "pos": r["qid"], "negs": negs, "style": STYLES[h(r["qid"]) % 3]})
    eval_q = [{"query": r["query"], "pos": r["qid"], "style": STYLES[h(r["qid"]) % 3]}
              for r in read(ROOT / "out" / "pairs_eval.jsonl")]
    train_sids = [s for s, r in rows.items() if r["split"] == "train"]
    used = {p["pos"] for p in train_pairs} | {n for p in train_pairs for n in p["negs"]}
    rest = [s for s in train_sids if s not in used]
    rng.shuffle(rest)
    pool = [s for s, r in rows.items() if r["split"] != "train"] + rest[:20000]
    distill = sorted(used) + rest[20000:45000]
    UP.mkdir(parents=True, exist_ok=True)
    (UP / "emb_data.json").write_text(json.dumps({"train_pairs": train_pairs, "eval_queries": eval_q,
                                                   "eval_pool": pool, "distill_sids": distill}))
    print(f"emb: {len(train_pairs)} train pairs, {kept_neg} negatives kept, {dropped} dropped as relevant; "
          f"{len(eval_q)} eval queries over a pool of {len(pool)}; {len(distill)} sections for the teacher pull")


if __name__ == "__main__":
    {"cases": cases, "emb": emb}[sys.argv[1]]()
