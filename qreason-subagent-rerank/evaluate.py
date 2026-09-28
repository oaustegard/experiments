"""NDCG@10 per arm, paired bootstrap CIs, run-to-run agreement, cross-batch calibration.
Merge rule for every arm: sort the 30 candidates by judge score desc, ties by retrieval rank."""
import json
import math
import random
import statistics as st

import numpy as np

K = json.load(open('keys.json')); sel = K['selected']; mp = K['mapping']
qrels = {q: set(v) for q, v in K['qrels'].items()}

def load(prefix):
    s = {}
    for p in (1, 2, 3):
        for q, v in json.load(open(f'outputs/{prefix}_{p}.json'))['scores'].items():
            s.setdefault(q, {}).update({pid: int(x) for pid, x in v.items()})
    assert all(len(s[q]) == 30 for q in sel), prefix
    return s

runs = {a: load(a) for a in ['A', 'A2', 'B', 'B2', 'C', 'C2', 'E', 'D']}

def ndcg(ranked, gold, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))

def ranked(s, q):
    return [mp[q][p] for p in sorted(mp[q], key=lambda p: (-s[q][p], int(p[1:])))]

res = {'R': {q: ndcg([mp[q][f'p{i:02d}'] for i in range(1, 31)], qrels[q]) for q in sel}}
for a, s in runs.items():
    res[a] = {q: ndcg(ranked(s, q), qrels[q]) for q in sel}
for a in 'ABC':
    res[a + 'x'] = {q: (res[a][q] + res[a + '2'][q]) / 2 for q in sel}

out = {'mean_ndcg10': {a: round(100 * st.mean(v.values()), 2) for a, v in res.items()}, 'paired': {}, 'agreement': {}, 'auc': {}}
random.seed(1)
def boot(x, y, B=20000):
    d = [res[x][q] - res[y][q] for q in sel]
    bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(B))
    return {'diff': round(100 * st.mean(d), 2), 'ci95': [round(100 * bs[int(.025 * B)], 2), round(100 * bs[int(.975 * B)], 2)],
            'wins': sum(v > 1e-9 for v in d), 'losses': sum(v < -1e-9 for v in d)}
for x, y in [('A', 'R'), ('B', 'A'), ('C', 'A'), ('E', 'A'), ('D', 'A'), ('C', 'B'), ('E', 'B'), ('D', 'B'),
             ('B2', 'A2'), ('C2', 'A2'), ('B2', 'C2'), ('Bx', 'Ax'), ('Cx', 'Ax'), ('Bx', 'Cx'), ('Ax', 'R'), ('E', 'Bx'), ('D', 'Bx')]:
    out['paired'][f'{x}-{y}'] = boot(x, y)
for a in 'ABC':
    xs = [(runs[a][q][p], runs[a + '2'][q][p]) for q in sel for p in mp[q]]
    out['agreement'][a] = {'exact': round(sum(x == y for x, y in xs) / len(xs), 3),
                           'pearson': round(float(np.corrcoef(*zip(*xs))[0, 1]), 3),
                           'per_query_ndcg_diff_sd': round(100 * st.pstdev([res[a][q] - res[a + '2'][q] for q in sel]), 1)}
def auc(pairs):
    g = [s for s, y in pairs if y]; n = [s for s, y in pairs if not y]
    return None if not g or not n else sum((a > b) + 0.5 * (a == b) for a in g for b in n) / (len(g) * len(n))
for a, s in runs.items():
    G, W = [], []
    for q in sel:
        pr = [(s[q][p], mp[q][p] in qrels[q]) for p in mp[q]]
        x = auc(pr); x is not None and G.append(x)
        for b in range(3):
            y = auc(pr[10 * b:10 * b + 10]); y is not None and W.append(y)
    out['auc'][a] = {'global': round(st.mean(G), 3), 'within_batch': round(st.mean(W), 3)}
out['per_query'] = {a: {q: round(v[q], 4) for q in sel} for a, v in res.items()}
json.dump(out, open('results/summary.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['mean_ndcg10', 'agreement', 'auc']}, indent=0))
for k, v in out['paired'].items(): print(k, v)
