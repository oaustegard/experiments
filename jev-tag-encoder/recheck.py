"""Re-derive the numbers RESULTS.md's Answer and Findings quote from results/*.json. Exit 1 on drift."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def j(name):
    return json.loads((R / f"{name}.json").read_text())


s2, s3, s4, s5, s6, qe, pc = (j(n) for n in ("step2_fit", "step3_classify", "step4_retrieve", "step5_phrasing",
                                             "step6_determinism", "quant_eval", "probe_codes"))
q3 = qe["step3"]["arms"]
q4 = qe["step4"]
a = s3["arms"]
runs = s4["runs"]
checks = [
    ("zs micro-F1", a["jev_zs@0.5"]["micro_f1"], 0.636),
    ("dense scaled@50 micro-F1", a["dense+probe@50"]["micro_f1"], 0.471),
    ("dense raw@50", pc["dense-noscale@50"]["micro_f1"], 0.517),
    ("dense raw@200", pc["dense-noscale@200"]["micro_f1"], 0.714),
    ("dense raw@500", pc["dense-noscale@500"]["micro_f1"], 0.778),
    ("dense raw@500 AP", pc["dense-noscale@500"]["micro_ap"], 0.855),
    ("jev raw@50", pc["float-noscale@50"]["micro_f1"], 0.566),
    ("jev raw@200", pc["float-noscale@200"]["micro_f1"], 0.671),
    ("jev raw@500", pc["float-noscale@500"]["micro_f1"], 0.697),
    ("jev raw@200 - zs", pc["float-noscale@200"]["minus_zs"][0][0], 0.035),
    ("jev scaled@50", pc["float@50"]["micro_f1"], 0.347),
    ("3bit-logit AP", q3["3bit-logit"]["point"][3], 0.567),
    ("2bit-logit AP", q3["2bit-logit"]["point"][3], 0.527),
    ("1bit AP", q3["1bit-sign"]["point"][3], 0.433),
    ("1bit zs F1 unchanged", q3["1bit-sign"]["point"][0], 0.636),
    ("3bit-logit bernoulli", q4["3bit-logit|about_bernoulli"]["alone"]["mean"], 0.325),
    ("3bit-logit bernoulli delta", q4["3bit-logit|about_bernoulli"]["alone"]["delta_vs_float"], -0.011),
    ("1bit bernoulli", q4["1bit-sign|about_bernoulli"]["alone"]["mean"], 0.137),
    ("2bit-logit bernoulli", q4["2bit-logit|about_bernoulli"]["alone"]["mean"], 0.276),
    ("zs oracle-lab micro-F1", a["jev_zs@oracle-lab"]["micro_f1"], 0.705),
    ("jev256 scaled@500 micro-F1", a["jev256+probe@500"]["micro_f1"], 0.620),
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
