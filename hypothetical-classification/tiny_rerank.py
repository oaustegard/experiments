"""Tiny model as a RERANKER over the embedder's top-k, not as a generator.

tiny.py showed both Pleias models scoring below the no-model control as generators:
they echo the query or corrupt it from few-shot bleed. Generation asks a 57M model for
format compliance, which is the thing it is worst at.

Scoring asks it for nothing but a likelihood. Take the embedder's top-k labels, compute
logP(label | few-shot preamble + "Query: q\\nCategory:") under the tiny model, rerank.
No parsing, no format, no way to emit an illegal label. Both halves run in a browser:
MiniLM-int8 is 23 MB and Monad-q4f16 is 35 MB.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, '/home/user/muninn-utilities')
import bench
from muninn_utils.hypothetical_classifier import Vocabulary

SHOTS = [("wood coffee table", "Coffee Tables"),
         ("navy throw pillow", "Throw Pillows"),
         ("counter height stool", "Bar Stools"),
         ("kids bunk bed", "Kids Beds"),
         ("round wall mirror", "Wall Mirrors"),
         ("outdoor patio umbrella", "Patio Umbrellas")]
PRE = ("A product taxonomy maps each search query to its category label.\n"
       "Labels are short plural noun phrases in plain retail wording.\n\n")
BODY = "".join(f"Query: {q}\nCategory: {l}\n\n" for q, l in SHOTS)


class Scorer:
    def __init__(self, mid):
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()

    @torch.no_grad()
    def score(self, query, candidates, length_norm=True):
        """Mean (or sum) logP of each candidate's tokens given the prompt."""
        prompt = PRE + BODY + f"Query: {query}\nCategory:"
        pids = self.tok(prompt, return_tensors="pt")["input_ids"][0]
        out = []
        for c in candidates:
            cids = self.tok(" " + c, add_special_tokens=False,
                            return_tensors="pt")["input_ids"][0]
            ids = torch.cat([pids, cids]).unsqueeze(0)
            logits = self.m(ids).logits[0, :-1].log_softmax(-1)
            tgt = ids[0, 1:]
            lp = logits[torch.arange(len(tgt)), tgt][len(pids) - 1:]
            out.append(float(lp.mean() if length_norm else lp.sum()))
        return out


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    D = json.load(open("h40.json"))
    queries, goldn = D["q"], D["g"]
    vocab, _ = bench.load()
    gold = np.array([vocab.index(c) for c in goldn])

    res = {}
    for bk in ("minilm", "tfidf"):
        V = Vocabulary(vocab, backend=bk)
        cand = V.snap(queries, k=K)                        # embedder does recall
        base1 = float(np.mean([vocab.index(c[0][0]) == g for c, g in zip(cand, gold)]))
        baseK = float(np.mean([g in [vocab.index(l) for l, _ in c] for c, g in zip(cand, gold)]))
        print(f"\n[{bk}] embedder alone: acc@1={base1:.3f}  ceiling recall@{K}={baseK:.3f}", flush=True)
        res[f"{bk}/embedder acc@1"] = base1
        res[f"{bk}/ceiling recall@{K}"] = baseK

        for mid in ("PleIAs/Monad", "PleIAs/Baguettotron"):
            sc = Scorer(mid)
            t = time.time()
            top1 = []
            for q, c in zip(queries, cand):
                labels = [l for l, _ in c]
                lp = sc.score(q, labels)
                top1.append(labels[int(np.argmax(lp))])
            dt = (time.time() - t) / len(queries)
            a1 = float(np.mean([vocab.index(l) == g for l, g in zip(top1, gold)]))
            name = f"{bk}/{mid.split('/')[1]} rerank acc@1"
            res[name] = a1
            print(f"  {mid.split('/')[1]:16} rerank acc@1={a1:.3f}   {dt*1000:6.0f} ms/query", flush=True)
            del sc
    json.dump(res, open("tiny_rerank_results.json", "w"), indent=1)
