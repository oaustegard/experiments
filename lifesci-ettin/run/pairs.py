"""Query/document pairs for the contrastive round, from the training shards rounds 1-2 used.

    python3 -I run/pairs.py   # .work/shards/pairs.npz

train.jsonl was written per PMID: the title (role q, with e17c's query prompt) and then the abstract's
passages (role d). The shards keep that row order, so a q row owns the d rows up to the next q row.
A q row is recognised by the query prompt's token prefix, in each arm's own tokenization, and the two
arms must agree on which rows are queries. Abstracts whose title failed to split have no q row; their
passages follow the previous title, and are dropped by requiring every d row of a group to come from the
same abstract run (here: at most 12 passages, the longest a <= 400-word abstract gives at 90 words each).

Output: q [N] (row index of each query), d0 [N] and dn [N] (its first passage row and passage count),
and a seeded order of pairs: per epoch a permutation cut into batches of P pairs, each batch's positive
drawn uniformly from the query's passages with a seeded generator, the same for every arm.
"""
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import QP  # noqa: E402

W = HERE / ".work"
S = W / "shards"
TOK = {"stock": W / "e17c/epoch1/tokenizer.json", "ls32": W / "tok/ls32/tokenizer.json"}
P, EPOCHS, MAXP = 64, 2, 12


def qrows(arm):
    lens = np.fromfile(S / f"{arm}.lens.u16", np.uint16).astype(np.int64)
    ids = np.fromfile(S / f"{arm}.ids.u16", np.uint16)
    offs = np.concatenate([[0], np.cumsum(lens)])
    if arm == "stock":
        pre = Tokenizer.from_file(str(TOK[arm])).encode(QP + "x").ids[:-2]  # [CLS] + prompt tokens, without "x" [SEP]
    else:
        pre = None
    return lens, ids, offs, pre


def find_q(ids, offs, pre):
    k = len(pre)
    pre = np.array(pre, np.uint16)
    return np.array([np.array_equal(ids[o:o + k], pre) for o in offs[:-1]])


def main():
    lens, ids, offs, pre = qrows("stock")
    isq = find_q(ids, offs, pre)
    # the 32K arm: its prefix is whatever its first q row starts with, cross-checked on every row
    l2 = np.fromfile(S / "ls32.lens.u16", np.uint16).astype(np.int64)
    i2 = np.fromfile(S / "ls32.ids.u16", np.uint16)
    o2 = np.concatenate([[0], np.cumsum(l2)])
    first = int(np.argmax(isq))
    k2 = 1
    while k2 < 64 and all(i2[o2[r] + k2] == i2[o2[first] + k2] for r in np.flatnonzero(isq)[:200]):
        k2 += 1
    pre2 = i2[o2[first]:o2[first] + k2 - 1]  # stop one short: the title's first token may coincide
    isq2 = find_q(i2, o2, list(pre2))
    agree = float((isq == isq2).mean())
    qs = np.flatnonzero(isq)
    nxt = np.append(qs[1:], len(lens))
    dn = nxt - qs - 1
    keep = (dn >= 1) & (dn <= MAXP)
    q, d0, dn = qs[keep], qs[keep] + 1, dn[keep]
    rng = np.random.default_rng(1)
    order = []
    for _ in range(EPOCHS):
        perm = rng.permutation(len(q))
        perm = perm[: len(perm) // P * P]
        pos = d0[perm] + (rng.random(len(perm)) * dn[perm]).astype(np.int64)
        order.append(np.stack([q[perm], pos], 1))
    order = np.concatenate(order).astype(np.uint32)  # [steps * P, 2]: (query row, positive row)
    np.savez(S / "pairs.npz", q=q, d0=d0, dn=dn, order=order)
    stats = {"rows": len(lens), "q_rows": int(isq.sum()), "arms_agree_on_q": agree, "pairs": len(q),
             "dropped_groups": int((~keep).sum()), "mean_passages": float(dn.mean()),
             "steps_per_epoch": len(order) // P // EPOCHS, "P": P, "epochs": EPOCHS}
    print(stats)
    tok = Tokenizer.from_file(str(TOK["stock"]))
    for a, b in order[:2]:
        print(" Q:", tok.decode(ids[offs[a]:offs[a + 1]].tolist())[:160])
        print(" D:", tok.decode(ids[offs[b]:offs[b + 1]].tolist())[:160])
    return stats


if __name__ == "__main__":
    import json
    (HERE / "results" / "pairs_stats.json").write_text(json.dumps(main(), indent=1))
