"""Re-derive the numbers RESULTS.md's Answer and Findings quote from results/*.json. Exit 1 on drift."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def j(name):
    return json.loads((R / f"{name}.json").read_text())


s2, s3, s4, s5, s6 = (j(n) for n in ("step2_fit", "step3_classify", "step4_retrieve", "step5_phrasing",
                                     "step6_determinism"))
a = s3["arms"]
runs = s4["runs"]
checks = [
    ("zs micro-F1", a["jev_zs@0.5"]["micro_f1"], 0.636),
    ("dense@50 micro-F1", a["dense+probe@50"]["micro_f1"], 0.471),
    ("dense@200 micro-F1", a["dense+probe@200"]["micro_f1"], 0.720),
    ("dense@500 micro-F1", a["dense+probe@500"]["micro_f1"], 0.779),
    ("zs oracle-lab micro-F1", a["jev_zs@oracle-lab"]["micro_f1"], 0.705),
    ("jev256@500 micro-F1", a["jev256+probe@500"]["micro_f1"], 0.620),
    ("zs micro-AP", a["jev_zs@0.5"]["micro_ap"], 0.610),
    ("cs.LG precision", s3["per_label_zs"]["cs.LG"]["precision"], 0.29),
    ("rrf2", runs["rrf(bm25,dense)"]["ndcg10"], 0.774),
    ("rrf3 bernoulli", runs["rrf(bm25,dense,jev_about_bernoulli)"]["ndcg10"], 0.688),
    ("rrf3 bernoulli delta", -runs["rrf(bm25,dense,jev_about_bernoulli)"]["delta_vs_rrf2"], 0.087),
    ("jev bernoulli alone", runs["jev_about_bernoulli"]["ndcg10"], 0.337),
    ("jev dot alone", runs["jev_about_dot"]["ndcg10"], 0.184),
    ("bm25+jev vs bm25", -runs["rrf(bm25,jev_about_bernoulli)"]["delta_vs_bm25"], 0.072),
    ("dense+jev vs dense", -runs["rrf(dense,jev_about_bernoulli)"]["delta_vs_dense"], 0.325),
    ("active/doc mixed", s2["active_mean"], 7.53),
    ("dead all corpora", len(s2["dead_all_corpora"]), 27),
    ("calls ok", s2["cost"]["calls_ok"], 7449),
    ("latency p50", s2["cost"]["latency_p50"], 0.59),
    ("mentions delta", -s5["mentions"]["delta_micro_f1_vs_about"][0], 0.088),
    ("determinism flips", s6["bit_flips_total"], 5),
    ("determinism identical", s6["frac_identical"], 0.847),
]
bad = [(n, got, want) for n, got, want in checks if abs(got - want) > (0.0051 if abs(want) < 10 else 0.5)]
for n, got, want in checks:
    print(f"{'FAIL' if (n, got, want) in bad else 'ok  '} {n}: {got:.4f} (quoted {want})")
sys.exit(1 if bad else 0)
