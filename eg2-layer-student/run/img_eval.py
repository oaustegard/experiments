"""Score query encoders on text->image over the bsky image-search index (run/img_common.py).

Usage: out/imgenv/bin/python -I run/img_eval.py [--build k4_t3w=out/export/l2_mac_k4/model_t3w.onnx ...] [--name base]

EG2 = the teacher vectors from run/img_teacher.py. Each build is an exported ONNX with its
tokenizer.json beside it. Reports, on all 5,239 queries and on the 1,035 held-out-image ones:
hit@1, hit@10, top-10 overlap with EG2's own top 10, median cosine to EG2, paired differences
against EG2, and the share of alt-text queries the 32K cut tokenizes exactly as EG2 does.
Writes results/img_eval_<name>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_common import IMG, QUERY, ImageIndex, eval_queries, is_test, paired, rows, score  # noqa: E402
from export import ort_embed  # noqa: E402
from model import ROOT, tokenize  # noqa: E402
from common import Teacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--name", default="base")
    a = ap.parse_args()
    I = ImageIndex()
    texts, target = eval_queries()
    E = np.load(IMG / "teacher_eval_all.npy")
    assert json.loads((IMG / "teacher_eval_all.json").read_text()) == texts
    cid = {r["i"]: r["cid"] for r in rows()}
    test = np.array([is_test(cid[t]) for t in target])
    T = Teacher()
    splits = {"all": np.ones(len(texts), bool), "test": test}
    rank_e, top_e = I.rank_and_top(E, target)
    res = {"n_images": I.n, "n_queries": {k: int(m.sum()) for k, m in splits.items()}, "builds": {}}
    per = {}
    for sp, m in splits.items():
        r, p, _ = score(I, E[m], target[m])
        res.setdefault("eg2", {})[sp] = r
        per[("eg2", sp)] = p
    vocab_ids = None
    for spec in a.build:
        name, path = spec.split("=", 1)
        path = ROOT / path
        tok = Tokenizer.from_file(str(path.with_name("tokenizer.json")))
        enc = tokenize(T, texts, remap=tok)
        Q = ort_embed(path, enc)
        res["builds"][name] = str(path.relative_to(ROOT))
        for sp, m in splits.items():
            r, p, _ = score(I, Q[m], target[m], E[m], top_e[m])
            for k in ("hit1", "hit10"):
                r[f"{k}_minus_eg2"] = paired(p[k], per[("eg2", sp)][k])
            res.setdefault(name, {})[sp] = r
            per[(name, sp)] = p
        if vocab_ids is None:  # intact share: the cut's tokenization equals EG2's, token for token
            vocab = tok.get_vocab()
            inv_full = {i: t for t, i in T.tok.get_vocab().items()}
            full = [T.tok.encode(QUERY + T.clean(t)).ids[:128] for t in texts]
            vocab_ids = float(np.mean([[inv_full[i] for i in f] == [tok.id_to_token(i) for i in c]
                                       for f, c in zip(full, enc)]))
            res["intact_tokenization_share_32k"] = vocab_ids
        print(name, json.dumps({sp: {k: round(v[0], 4) if isinstance(v, list) else v for k, v in res[name][sp].items()
                                     if k in ("hit@1", "hit@10", "overlap@10_vs_eg2")} for sp in splits}), flush=True)
    print("eg2", json.dumps({sp: {k: round(v[0], 4) for k, v in res["eg2"][sp].items()} for sp in splits}))
    (ROOT / "results" / f"img_eval_{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
