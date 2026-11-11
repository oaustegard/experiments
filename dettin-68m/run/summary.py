"""Collect the headline numbers from results/ into results/summary.json (what RESULTS.md cites).

    python3 run/summary.py
"""
import glob
import json
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def load(p):
    return json.loads((R / p).read_text())


def r3(x):
    return round(x, 3)


s = {"td": {}, "ls": {}, "emb": {}, "speed": {}, "runtime": {}}
lr = load("a/td/a_lr.json")
for size in ("17", "32", "68"):
    acc = [load(f"a/td/a-{size}-lr{lr[size]}-s{k}.json")["test"]["accuracy"] for k in range(3)]
    q = load(f"a/td/quant_levels-a-{size}-lr{lr[size]}-s0.json")["variants"]
    s["td"][size] = {"acc_mean": r3(st.mean(acc)), "acc_sd": r3(st.stdev(acc)), "lr": lr[size],
                     "fp32": r3(q["fp32"]["test"]["accuracy"]), "packed63": r3(q["t4-w8L63"]["test"]["accuracy"]),
                     "packed31": r3(q["t4-w8L31"]["test"]["accuracy"]),
                     "fp32_mb": round(q["fp32"]["file_bytes"] / 1e6, 1), "packed_mb": round(q["t4-w8L63"]["file_bytes"] / 1e6, 1),
                     "packed63_brotli_mb": round(q["t4-w8L63"]["file_brotli"] / 1e6, 1),
                     "packed31_brotli_mb": round(q["t4-w8L31"]["file_brotli"] / 1e6, 1),
                     "packed63_agree": r3(q["t4-w8L63"]["top_agree_with_fp32"])}
lowo = [json.loads(Path(f).read_text())["test"]["accuracy"] for f in sorted(glob.glob(str(R / "a/td/a-68-lowo-*.json"))) if "preds" not in f]
s["td"]["lowo68"] = r3(st.mean(lowo))
sp = {"lora": [], "full": []}
for f in glob.glob(str(R / "a/spikes/S-*.json")):
    r = json.loads(Path(f).read_text())
    sp[r["kind"]].append(r["test"]["accuracy"])
    if r["kind"] == "lora":
        s["td"]["spike_trainable"] = r["trainable"]
s["td"]["spike_lora68"] = r3(st.mean(sp["lora"]))
s["td"]["spike_full68"] = r3(st.mean(sp["full"]))
s["td"]["step_17_32"] = r3(s["td"]["32"]["acc_mean"] - s["td"]["17"]["acc_mean"])
s["td"]["step_32_68"] = r3(s["td"]["68"]["acc_mean"] - s["td"]["32"]["acc_mean"])

d = load("decisions.json")
for size in ("17", "32", "68"):
    s["ls"][size] = {"macro": r3(d["sizes"][size]["macro_all"]), **{q: r3(v) for q, v in d["sizes"][size]["per_question"].items()}}
    q = load(f"b/ls/quant_levels-b-{size}-haiku-s0.json")["variants"]
    s["ls"][size]["all_items_fp32"] = r3(q["fp32"]["test"]["accuracy"])
    s["ls"][size]["all_items_packed63"] = r3(q["t4-w8L63"]["test"]["accuracy"])
s["ls"]["spikes68"] = r3(st.mean(d["models"][f"spikes68-s{k}"]["macro_all"] for k in (0, 1)))
s["ls"]["spikes68_per_question"] = {q: r3(st.mean(d["models"][f"spikes68-s{k}"][q]["acc"] for k in (0, 1))) for q in d["prior"]}
s["ls"]["jev_teacher68"] = r3(d["models"]["b-68-jev-s0"]["macro_all"])
s["ls"]["haiku"] = {q: r3(v["acc"]) for q, v in d["models"]["labeller-haiku"].items() if isinstance(v, dict)}
s["ls"]["jev"] = {q: r3(v["acc"]) for q, v in d["models"]["labeller-jev"].items() if isinstance(v, dict)}
s["ls"]["haiku_macro8"] = r3(d["models"]["labeller-haiku"]["macro_all"])
s["ls"]["jev_macro8"] = r3(d["models"]["labeller-jev"]["macro_all"])
s["ls"]["macro8"] = {size: r3(st.mean(v for q, v in d["sizes"][size]["per_question"].items() if q != "relevance")) for size in ("17", "32", "68")}
s["ls"]["agreement"] = {q: r3(v) for q, v in d["agreement"].items()}
s["ls"]["prior"] = {q: r3(v) for q, v in d["prior"].items()}
s["ls"]["steps"] = {k: {"delta": r3(v["macro_delta"]), "lo": r3(v["ci95"][0]), "hi": r3(v["ci95"][1])} for k, v in d["size_steps"].items()}
s["ls"]["immunoassay_auroc68"] = r3(st.mean(d["immunoassay_auroc"][f"b-68-haiku-s{k}"] for k in range(3)))

E = {}
for f in glob.glob(str(R / "c/emb/*.json")):
    r = json.loads(Path(f).read_text())
    E[r["name"]] = r
E["base-bm25"], E["base-bm25-para"] = load("emb_base-bm25.json"), load("emb_base-bm25-para.json")
ORIG = {"base-gte": "base-gte-p2", "base-medcpt": "base-medcpt-p2", "e17-distill": "e17-distill-p2",
        "e32-distill": "e32-distill-p2", "e68-distill": "e68-distill-p2"}
for n in ("base-bm25", "base-gte", "base-bge-small", "base-medcpt", "base-capemox", "e17-distill", "e32-distill", "e68-distill",
          "e17-joint", "e32-joint", "e68-joint", "e68-joint-q", "e68-contrast"):
    row = {"orig": r3(E[ORIG.get(n, n)]["sections"]["ndcg@10"]), "para": r3(E[n + "-para"]["sections"]["ndcg@10"]),
           "para_recall10": r3(E[n + "-para"]["sections"]["recall@10"]),
           "para_within_article": r3(E[n + "-para"]["sections"]["within_article_top1"])}
    for b in ("scifact", "nfcorpus"):
        if b in E.get(n, {}) and "ndcg@10" in E[n][b]:
            row[b] = r3(E[n][b]["ndcg@10"])
    s["emb"][n] = row
s["emb"]["pool"] = E["e68-joint"]["sections"]["pool"]
s["emb"]["n_queries"] = E["e68-joint"]["sections"]["n_queries"]
s["emb"]["step_17_32_para"] = r3(s["emb"]["e32-joint"]["para"] - s["emb"]["e17-joint"]["para"])
s["emb"]["step_32_68_para"] = r3(s["emb"]["e68-joint"]["para"] - s["emb"]["e32-joint"]["para"])
s["emb"]["fold"] = load("fold_check.json")

for tag in ("cpu4", ):
    for r in load(f"speed_{tag}.json")["rows"]:
        if r["batch"] == 1:
            s["speed"][f"{tag}_{r['model']}_{r['tokens']}_ms"] = r["ms_per_batch"]
for r in load("a/speed/speed_gpu.json")["rows"]:
    s["speed"][f"a100_{r['model']}_{r['tokens']}_b{r['batch']}_per_s"] = r["seqs_per_s"]
    if r["batch"] == 1:
        s["speed"][f"a100_{r['model']}_{r['tokens']}_ms"] = r["ms_per_batch"]
chk = [json.loads(line) for line in (R / "dettin_check.jsonl").read_text().splitlines() if line.startswith("{")]
s["runtime"] = {"packed68_td_acc": chk[0]["accuracy"], "packed68_ms_per_case_cpu": round(chk[0]["ms_per_case_median"]),
                "spike_invoice_acc": chk[1]["accuracy"], "spike_customer_acc": chk[2]["accuracy"],
                "spike_invoice_fp32_base": load("a/spikes/S-lora-invoice_processing-s0.json")["test"]["accuracy"],
                "spike_customer_fp32_base": load("a/spikes/S-lora-customer_service-s0.json")["test"]["accuracy"]}
s["corpus"] = {k: v for k, v in json.loads((R.parent / "data/corpus_stats.json").read_text()).items() if k in ("articles", "rows_by_split")}
s["cost"] = {"colab_units": 70.3, "jev_input_tokens_m": 87.6, "headless_usd_list": {
    "queries_train_haiku": 3.90, "queries_eval_sonnet": 3.62, "gold_sonnet": 1.39, "gold_opus": 3.42, "gold_haiku": 0.13,
    "label_haiku": 6.18, "goldrel_opus": 4.65}}
(R / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
print(json.dumps({k: s[k] for k in ("td", "ls")}, indent=None)[:2500])
