"""Collect every scored run into results/summary.json, the file RESULTS.md cites.

    python3 make_summary.py

arms.<name>   GPU (fp32 HF) selection runs, one per checkpoint: held-out paper questions (judged =
              correct + half partial), the same questions' fully-correct share, post-cutoff papers,
              training questions asked back verbatim (seen), the 30 one-line facts, and where measured
              the dev fixture (loop rate, MMLU, GSM8K, IFEval loop rate)
final.<name>  the shipped 8-bit exports on onnxruntime CPU and on the Mac GPU bridge (WebGPU)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def flat(d):
    out = {}
    if "papers_all" in d:
        p = d["papers_all"]
        out.update({"papers_judged": p.get("judged"), "papers_correct": p.get("correct"),
                    "papers_key_recall": p.get("key_recall"), "papers_loop": p.get("loop"),
                    "papers_post_judged": d["papers_post"].get("judged"), "papers_pre_judged": d["papers_pre"].get("judged")})
    if "seen_all" in d:
        out["seen_judged"] = d["seen_all"].get("judged")
        out["seen_correct"] = d["seen_all"].get("correct")
    if "facts" in d:
        out["facts_ok"] = d["facts"]["ok"]
    if "dev" in d:
        v = d["dev"]
        out.update({"dev_loop": v["loop"], "dev_mmlu": v["mmlu_acc"], "dev_gsm8k": v["gsm8k_acc"],
                    "dev_ifeval_loop": v["ifeval_loop"]})
    return {k: v for k, v in out.items() if v is not None}


def main():
    sel, gpu, seen, dev = load("sel"), load("gpu_select"), load("seen"), load("dev")
    names = {"A8p": "A8p"} | {k: k for k in list(sel) + list(gpu)}
    arms = {}
    for n in names:
        a = {}
        for src in (gpu.get(n, {}), sel.get(n, {}), {"seen_all": seen[n]["seen_all"]} if n in seen else {},
                    {"dev": dev[n]["dev"]} if n in dev and "dev" in dev[n] else {}):
            a.update(flat(src))
        arms[n.replace("-", "_")] = a
    final = {}
    for rt in ("cpu", "bridge"):
        for k, v in load(rt).items():          # keys are run-file stems: <model>__<set or runtime>
            final.setdefault(rt, {}).setdefault(k.split("__")[0], {}).update(flat(v))
    papers = json.loads((HERE / "data/papers.json").read_text())
    ranked = json.loads((HERE / "data/papers_ranked.json").read_text())
    ho = [json.loads(l) for l in open(HERE / "data/heldout.jsonl")]
    para = sum(1 for _ in open(HERE / "data/train_para.jsonl"))
    tr = sum(1 for _ in open(HERE / "data/train_papers.jsonl"))
    data = {"papers_ranked": len(ranked), "papers_with_abstract": sum(bool(p["abstract"]) for p in papers),
            "papers_post_cutoff": len({h["paper"] for h in ho if h["post_cutoff"]}), "train_rows_r1": tr,
            "train_rows_r2": para, "paraphrases_kept": para - tr,
            "paraphrases_dropped": len(json.loads((HERE / "data/para_dropped.json").read_text())),
            "heldout": len(ho), "replay_pool": sum(1 for _ in open(HERE / "data/replay.jsonl"))}
    timing = {k: load(k) for k in ("bridge_timing_A8p_q8", "bridge_timing_skygest_e8") if load(k)}
    out = {"data": data, "arms": arms, "final": final, "timing": timing, "extra": load("extra")}
    (R / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
