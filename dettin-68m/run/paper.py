"""Haiku at low effort as the decision model, reading a whole article: single call, stacked calls, worked examples.

    python3 run/paper.py pick                 # 150 test and 30 dev articles -> data/paper_picks.json
    python3 run/paper.py tasks                # every task file under out/tasks/paper_*.jsonl (not the ICL arm)
    python3 run/paper.py icl                  # dev disagreements -> data/paper_icl.json, out/tasks/paper_haiku_icl.jsonl
    python3 run/paper.py jev                  # Jev on the same articles -> data/paper_jev.jsonl
    python3 run/paper.py score                # -> results/paper.json

An article is rebuilt from its parsed body sections, in order, with their headings (sections under 120 words,
back matter, tables and figures are not in it). Four questions about the article as a whole. Gold is where
Sonnet 5.5 and Opus 5.5, each reading the full article, agree. Haiku arms, one article per call at low effort:
  plain x3       the same instruction three times (what resampling alone buys)
  methods        "decide from what the Methods and Results report was done"
  sceptic        "start from the least committal answer and move only on an explicit statement"
  icl            the plain instruction plus worked examples, drawn from dev articles where plain Haiku and Opus disagreed
A stack is a majority vote over three calls; a tie goes to the plain call.
"""
import glob
import gzip
import json
import random
import statistics as st
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T
from suite import FIELDS

ROOT = Path(__file__).resolve().parents[1]
N_TEST, N_DEV, MAX_WORDS = 150, 30, 9000

QS = {
    "study_type": {"type": "choice", "instructions": "What kind of study does this article report?", "criteria": {
        "in_vitro": "Experiments on cells, tissues, proteins or other biological material outside a living organism",
        "animal": "Experiments or observations on live non-human animals",
        "human_interventional": "People assigned to an intervention: a clinical trial or an experimental manipulation",
        "human_observational": "People observed without assignment: cohort, case-control, cross-sectional, case report, survey, qualitative study",
        "computational": "Analysis of existing datasets, modelling, simulation or software, with no new laboratory or participant data",
        "review": "A review or meta-analysis of published literature",
        "other": "Another kind of study: plants, field ecology, chemistry, engineering, method development"}},
    "field": {"type": "choice", "instructions": "Which field does this article belong to?", "criteria": FIELDS},
    "human_samples": {"type": "noul", "instructions":
                      "The study itself collected or analysed data from human participants or human-derived samples (patients, "
                      "volunteers, human tissue, blood, primary human cells). Established human cell lines alone do not count."},
    "immunoassay": {"type": "noul", "instructions":
                    "The study itself quantified an analyte in samples with a ligand-binding immunoassay: ELISA, "
                    "electrochemiluminescence (such as Meso Scale Discovery), a bead-based multiplex (such as Luminex), Simoa, "
                    "radioimmunoassay or a lateral-flow test. Western blots, immunohistochemistry, immunofluorescence and flow "
                    "cytometry do not count, and neither do assays only cited from other papers."},
}
BASE = ("You answer fixed questions about a life-science research article. Read the whole article and answer each "
        "question for the article as a whole.\n\nQuestions:\n")
ANGLE = {
    "plain": "",
    "methods": "\n\nDecide from what the Methods and Results sections report was actually done in this study. Do not "
               "count what the Introduction or Discussion says about other studies.",
    "sceptic": "\n\nFor each question start from the least committal answer (false for a yes/no question, the option "
               "that claims least for a choice) and move off it only when the article states the fact explicitly.",
}


def lab(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def articles():
    arts = {}
    with gzip.open(ROOT / "out" / "sections.jsonl.gz", "rt") as f:
        for r in map(json.loads, f):
            arts.setdefault(r["pmcid"], []).append(r)
    return arts


def text_of(units):
    units = sorted(units, key=lambda r: int(r["sid"].split(":")[1]))
    return f"# {units[0]['article_title']}\n\n" + "\n\n".join(f"## {u['heading']}\n{u['text']}" for u in units)


def pick():
    arts = articles()
    rng = random.Random(0)
    out, skipped = {}, 0
    for split, n in (("test", N_TEST), ("dev", N_DEV)):
        ids = sorted(a for a, u in arts.items() if u[0]["split"] == split and len(u) >= 4)
        rng.shuffle(ids)
        keep = []
        for a in ids:
            if sum(u["n_words"] for u in arts[a]) > MAX_WORDS:
                skipped += 1
                continue
            keep.append(a)
            if len(keep) == n:
                break
        out[split] = keep
    (ROOT / "data" / "paper_picks.json").write_text(json.dumps(out))
    w = [sum(u["n_words"] for u in arts[a]) for a in out["test"]]
    print({k: len(v) for k, v in out.items()}, "skipped over the word cap:", skipped, "test words median", st.median(w), "max", max(w))


def make(name, ids, system, model, schema, arts):
    out = [{"id": f"{name}-{i:04d}", "prompt": "Answer the questions for the article below. Return items with n = 1.\n\n" + text_of(arts[a]),
            "system_prompt": system, "json_schema": schema, "tools": "", "model": model, "_ids": [a]} for i, a in enumerate(ids)]
    T.write_tasks(name, out)


def tasks():
    arts = articles()
    p = json.loads((ROOT / "data" / "paper_picks.json").read_text())
    schema = T.item_schema(QS)
    sysp = BASE + T.rubric(QS)
    for j in ("sonnet", "opus"):
        make(f"paper_gold_{j}", p["test"], sysp, j, schema, arts)
    for k in (1, 2, 3):
        make(f"paper_haiku_plain{k}", p["test"], sysp, "haiku", schema, arts)
    for a in ("methods", "sceptic"):
        make(f"paper_haiku_{a}", p["test"], sysp + ANGLE[a], "haiku", schema, arts)
    make("paper_dev_haiku", p["dev"], sysp, "haiku", schema, arts)
    # dev gold from Opus carries, per question, the sentence that settles it: the raw material for worked examples
    ev = json.loads(json.dumps(schema))
    props = ev["properties"]["items"]["items"]["properties"]
    for q in QS:
        props[f"{q}_evidence"] = {"type": "string"}
    ev["properties"]["items"]["items"]["required"] = list(props)
    make("paper_dev_opus", p["dev"], sysp + "\n\nFor each question also give `<question>_evidence`: the one sentence from the "
         "article that settles your answer, quoted exactly, or an empty string when the answer rests on an absence.", "opus", ev, arts)


def read(name):
    f = ROOT / "data" / f"{name}.jsonl"
    with open(f) as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def icl():
    arts = articles()
    p = json.loads((ROOT / "data" / "paper_picks.json").read_text())
    hk, op = read("paper_dev_haiku"), read("paper_dev_opus")
    ex, used = [], Counter()
    for a in p["dev"]:
        if a not in hk or a not in op:
            continue
        wrong = [q for q in QS if lab(hk[a][q]) != lab(op[a][q])]
        for q in wrong:
            if used[q] >= 3 or not op[a].get(f"{q}_evidence"):
                continue
            used[q] += 1
            ex.append({"pmcid": a, "question": q, "title": arts[a][0]["article_title"], "evidence": op[a][f"{q}_evidence"],
                       "answer": lab(op[a][q]), "haiku_said": lab(hk[a][q])})
    agree = {q: sum(lab(hk[a][q]) == lab(op[a][q]) for a in hk if a in op) / max(1, sum(a in op for a in hk)) for q in QS}
    (ROOT / "data" / "paper_icl.json").write_text(json.dumps({"examples": ex, "dev_haiku_vs_opus": agree}, indent=1))
    block = "\n\nWorked examples from other articles. Each gives a question, the sentence that settled it, and the answer.\n" + "\n".join(
        f"- {e['question']}: \"{e['evidence']}\" -> {e['answer']}" for e in ex)
    make("paper_haiku_icl", p["test"], BASE + T.rubric(QS) + block, "haiku", T.item_schema(QS), arts)
    print(f"{len(ex)} worked examples {dict(used)}; dev Haiku against Opus {json.dumps({k: round(v, 3) for k, v in agree.items()})}")


def jev():
    arts = articles()
    p = json.loads((ROOT / "data" / "paper_picks.json").read_text())
    import suite
    suite.ALL_QS.update({f"paper_{k}": v for k, v in QS.items()})
    inp = ROOT / "out" / "paper_jev_in.jsonl"
    inp.write_text("".join(json.dumps({"id": a, "state": {"article": text_of(arts[a])}, "questions": [f"paper_{k}" for k in QS]}) + "\n"
                           for a in p["test"]))
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0, 'run'); import paper, suite, jev_label;"
                    "suite.ALL_QS.update({f'paper_{k}': v for k, v in paper.QS.items()});"
                    f"sys.argv = ['jev_label', '{inp}', '{ROOT / 'data' / 'paper_jev.jsonl'}', '--workers', '6']; jev_label.main()"],
                   cwd=ROOT, check=False)


def score():
    p = json.loads((ROOT / "data" / "paper_picks.json").read_text())
    gs, go = read("paper_gold_sonnet"), read("paper_gold_opus")
    ids = [a for a in p["test"] if a in gs and a in go]
    truth = {q: {a: lab(go[a][q]) for a in ids if lab(gs[a][q]) == lab(go[a][q])} for q in QS}
    res = {"n_articles": len(ids), "agreement": {q: round(len(truth[q]) / len(ids), 3) for q in QS},
           "prior": {q: round(Counter(truth[q].values()).most_common(1)[0][1] / len(truth[q]), 3) for q in QS}, "arms": {}}

    def acc(pred):
        out = {}
        for q in QS:
            keys = [a for a in truth[q] if a in pred]
            out[q] = round(sum(pred[a][q] == truth[q][a] for a in keys) / len(keys), 3)
        out["mean"] = round(st.mean(out[q] for q in QS), 3)
        out["n"] = min(sum(a in pred for a in truth[q]) for q in QS)
        return out

    runs = {}
    for name in ("plain1", "plain2", "plain3", "methods", "sceptic", "icl"):
        f = ROOT / "data" / f"paper_haiku_{name}.jsonl"
        if f.exists():
            runs[name] = {a: {q: lab(r[q]) for q in QS} for a, r in read(f"paper_haiku_{name}").items()}
            res["arms"][f"haiku {name}"] = acc(runs[name])

    def stack(names):
        common = set.intersection(*(set(runs[n]) for n in names))
        out = {}
        for a in common:
            out[a] = {}
            for q in QS:
                votes = Counter(runs[n][a][q] for n in names)
                top, c = votes.most_common(1)[0]
                out[a][q] = top if c > 1 else runs[names[0]][a][q]
        return out

    if all(k in runs for k in ("plain1", "plain2", "plain3")):
        res["arms"]["stack: plain x3"] = acc(stack(["plain1", "plain2", "plain3"]))
    if all(k in runs for k in ("plain1", "methods", "sceptic")):
        res["arms"]["stack: plain + methods + sceptic"] = acc(stack(["plain1", "methods", "sceptic"]))
        res["three_angles_unanimous_share"] = {q: round(st.mean(len({runs[n][a][q] for n in ("plain1", "methods", "sceptic")}) == 1
                                                                 for a in set.intersection(*(set(runs[n]) for n in ("plain1", "methods", "sceptic")))), 3) for q in QS}
    if all(k in runs for k in ("icl", "methods", "sceptic")):
        res["arms"]["stack: icl + methods + sceptic"] = acc(stack(["icl", "methods", "sceptic"]))
    f = ROOT / "data" / "paper_jev.jsonl"
    if f.exists():
        with open(f) as fh:
            jv = {r["id"]: {q: max(r["probs"][f"paper_{q}"], key=r["probs"][f"paper_{q}"].get) for q in QS}
                  for r in map(json.loads, fh) if "probs" in r}
        res["arms"]["jev"] = acc(jv)
    cost = {}
    for d in sorted(glob.glob(str(ROOT / "out/fan/paper_*"))):
        with open(Path(d) / "ledger.jsonl") as fh:
            led = [json.loads(line) for line in fh]
        secs = []
        for r in glob.glob(d + "/results/*.json"):
            secs.append(json.loads(Path(r).read_text()).get("duration_ms", 0) / 1000)
        cost[Path(d).name] = {"usd": round(sum(r.get("cost", 0) for r in led), 3), "calls": len(led),
                              "seconds_median": round(st.median(secs), 2) if secs else None}
    res["runs"] = cost
    (ROOT / "results" / "paper.json").write_text(json.dumps(res, indent=1) + "\n")
    print("agreement", res["agreement"], "prior", res["prior"], "n", res["n_articles"])
    for k, v in res["arms"].items():
        print(f"{k:36s}", {q: v[q] for q in list(QS) + ["mean"]}, "n", v["n"])
    print(json.dumps(cost))


if __name__ == "__main__":
    {"pick": pick, "tasks": tasks, "icl": icl, "jev": jev, "score": score}[sys.argv[1]]()
