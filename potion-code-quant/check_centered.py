"""Parent-session check on pass 4: is the 0.96 student/teacher cosine real?

SPECTER2 vectors share a large mean component (raw pairwise cosine among teacher
vectors is 0.85), so a raw cosine of 0.96 mostly measures the mean. This refits the
pass-4 ridge table at alpha=0.1 and reports the cosine after subtracting the corpus
mean, plus the student/student cell under centered cosine, to show whether the
0.002 in the raw-inner-product cell is a norm artifact. Runs in ~2 min on 4 vCPU.
"""
import json
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from tokenizers import Tokenizer

CACHE = "/home/user/oaustegard/remax/bench/.cache/SPECTER2"
E = np.load(f"{CACHE}/embeddings.npy")
T = json.load(open(f"{CACHE}/texts.json"))
rng = np.random.default_rng(99)
perm = rng.permutation(len(E))
q_idx, c_idx = perm[:100], perm[100:]
tok = Tokenizer.from_file("/home/user/models/specter2_base/tokenizer.json")
tok.enable_truncation(512)
ids = [tok.encode(t, add_special_tokens=False).ids for t in T]
cnt = Counter(i for j in c_idx for i in set(ids[j]))
keep = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= 2))}


def rows(idx):
    r, c, v = [], [], []
    for n, j in enumerate(idx):
        L = len(ids[j]) or 1
        for t in ids[j]:
            if t in keep:
                r.append(n); c.append(keep[t]); v.append(1 / L)
    return sp.csr_matrix((v, (r, c)), shape=(len(idx), len(keep)))


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


Xc, Xq = rows(c_idx), rows(q_idx)
Yc, Yq = E[c_idx], E[q_idx]
half = len(c_idx) // 2
oof = np.zeros_like(Yc)
for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
    oof[b] = Ridge(alpha=0.1, fit_intercept=False, solver="sparse_cg").fit(Xc[a], Yc[a]).predict(Xc[b])
m = Ridge(alpha=0.1, fit_intercept=False, solver="sparse_cg").fit(Xc, Yc)
Sq = m.predict(Xq)
mu = Yc.mean(0)
truth = np.argsort(-(Yq @ Yc.T), 1)[:, :10]


def r10(scores):
    p = np.argsort(-scores, 1)[:, :10]
    return float(np.mean([len(set(p[i]) & set(truth[i])) / 10 for i in range(100)]))


out = {
    "query_cos_raw": float((unit(Sq) * unit(Yq)).sum(1).mean()),
    "query_cos_centered": float((unit(Sq - mu) * unit(Yq - mu)).sum(1).mean()),
    "teacher_pairwise_cos_raw": float((unit(Yc[:2000]) @ unit(Yc[:2000]).T).mean()),
    "teacher_pairwise_cos_centered": float((unit(Yc[:2000] - mu) @ unit(Yc[:2000] - mu).T).mean()),
    "corpus_oof_cos_centered": float((unit(oof - mu) * unit(Yc - mu)).sum(1).mean()),
    "r10_student_q_teacher_idx_ip": r10(Sq @ Yc.T),
    "r10_student_q_teacher_idx_centered_cos": r10(unit(Sq - mu) @ unit(Yc - mu).T),
    "r10_student_q_student_oof_idx_ip": r10(Sq @ oof.T),
    "r10_student_q_student_oof_idx_centered_cos": r10(unit(Sq - mu) @ unit(oof - mu).T),
    "norm_teacher_q": float(np.linalg.norm(Yq, axis=1).mean()),
    "norm_student_q": float(np.linalg.norm(Sq, axis=1).mean()),
}
print(json.dumps(out, indent=1))
json.dump(out, open("results_check_centered.json", "w"), indent=1)
