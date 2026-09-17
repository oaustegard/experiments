#!/usr/bin/env python3
"""Commutator test of LLM knowledge composition in the residual stream.

experiments#97. For a word pair (A, B) read the residual stream at the final
`:` token of an ordered composition h(AB) and its reversal h(BA), and measure
the commutator K = h(AB) - h(BA) under carriers where English order is
semantically live (COMPOUND: "a milk chocolate" / "a chocolate milk") and
inert (CONJUNCTION: "milk and chocolate" / "chocolate and milk"). RANDOM
(unrelated nouns) and LIST (adjacent nouns, order-inert frame) are controls.

Every number in RESULTS.md comes from `results/<model>.json` written here.
Seeded, CPU-only, fp32. Run: `python3 run.py [--models smol qwen05 qwen15]`.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

torch.set_num_threads(os.cpu_count() or 4)
SEED = 20260917
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

MODELS = {
    "smol": "HuggingFaceTB/SmolLM2-135M",
    "qwen05": "Qwen/Qwen2.5-0.5B",
    "qwen15": "Qwen/Qwen2.5-1.5B",
}

# Both orders form a compound with a different meaning (issue list, verbatim).
PAIRS = [
    ("milk", "chocolate"), ("horse", "race"), ("house", "boat"), ("rock", "garden"),
    ("dog", "house"), ("fire", "wood"), ("salt", "water"), ("card", "game"),
    ("coffee", "shop"), ("guest", "book"), ("box", "gift"), ("tree", "house"),
    ("work", "life"), ("pot", "flower"), ("cup", "tea"), ("night", "club"),
    ("bag", "hand"), ("wall", "paper"), ("glass", "wine"), ("board", "key"),
]

# Consonant-initial (so "a {A} {B}" stays grammatical) concrete nouns that
# do not compound with each other in either order. Paired by seeded shuffle.
RANDOM_NOUNS = [
    "lamp", "pencil", "cloud", "spoon", "mirror", "ladder", "candle", "violin",
    "turtle", "cactus", "helmet", "bucket", "pillow", "hammer", "lemon", "jacket",
    "saddle", "rocket", "bridge", "carrot", "tunnel", "feather", "pumpkin", "whistle",
    "sponge", "magnet", "kettle", "ribbon", "lantern", "wagon", "tulip", "parrot",
    "blanket", "cabbage", "dolphin", "marble", "needle", "peach", "walnut", "zebra",
]

# Three-word compounds: every one of the 6 orders is readable as a compound.
TRIPLES = [
    ("glass", "wine", "bottle"), ("house", "boat", "party"), ("rock", "garden", "wall"),
    ("dog", "house", "key"), ("night", "club", "card"), ("coffee", "shop", "guest"),
    ("fire", "wood", "box"), ("card", "game", "night"), ("tree", "house", "paper"),
    ("work", "life", "board"), ("salt", "water", "cup"), ("race", "horse", "track"),
]

# Three paraphrase carriers per condition. Index t is matched across
# conditions so a per-template comparison holds the frame fixed.
TEMPLATES = {
    "COMPOUND": {
        "pair":   ["Here is a {A} {B}:", "I saw a {A} {B}:", "They talked about the {A} {B}:"],
        "spoke":  ["Here is a {A}:",     "I saw a {A}:",     "They talked about the {A}:"],
        "bare":   ["Here is a:",         "I saw a:",         "They talked about the:"],
        "triple": ["Here is a {A} {B} {C}:", "I saw a {A} {B} {C}:", "They talked about the {A} {B} {C}:"],
    },
    "CONJUNCTION": {
        "pair":   ["Here is {A} and {B}:", "I saw {A} and {B}:", "They talked about {A} and {B}:"],
        "spoke":  ["Here is {A}:",         "I saw {A}:",         "They talked about {A}:"],
        "bare":   ["Here is:",             "I saw:",             "They talked about:"],
        "triple": ["Here is {A}, {B} and {C}:", "I saw {A}, {B} and {C}:", "They talked about {A}, {B} and {C}:"],
    },
    # Positional control (falsification prong 1): the nouns are adjacent
    # exactly as in COMPOUND, but the frame announces a bag of words.
    "LIST": {
        "pair":   ["Here are two words, {A} {B}:", "Two random words, {A} {B}:", "Vocabulary items, {A} {B}:"],
        "spoke":  ["Here are two words, {A}:",     "Two random words, {A}:",     "Vocabulary items, {A}:"],
        "bare":   ["Here are two words:",          "Two random words:",          "Vocabulary items:"],
        "triple": ["Here are three words, {A} {B} {C}:", "Three random words, {A} {B} {C}:", "Vocabulary items, {A} {B} {C}:"],
    },
}
CONDS = ["COMPOUND", "CONJUNCTION", "LIST"]
NT = 3
N_PERM = 1000


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
def random_pairs():
    rng = random.Random(SEED)
    nouns = RANDOM_NOUNS[:]
    rng.shuffle(nouns)
    return [(nouns[2 * i], nouns[2 * i + 1]) for i in range(len(nouns) // 2)]


def random_triples(n=12):
    rng = random.Random(SEED + 1)
    nouns = RANDOM_NOUNS[:]
    rng.shuffle(nouns)
    return [tuple(nouns[3 * i:3 * i + 3]) for i in range(n)]


def build_prompts():
    """Every prompt string the experiment reads, deduplicated."""
    P = set()
    rp = random_pairs()
    rt = random_triples()
    for c in CONDS:
        T = TEMPLATES[c]
        for t in range(NT):
            for A, B in PAIRS + rp:
                P.add(T["pair"][t].format(A=A, B=B))
                P.add(T["pair"][t].format(A=B, B=A))
                P.add(T["spoke"][t].format(A=A))
                P.add(T["spoke"][t].format(A=B))
            P.add(T["bare"][t])
            for tri in TRIPLES + rt:
                for perm in _perms3(tri):
                    P.add(T["triple"][t].format(A=perm[0], B=perm[1], C=perm[2]))
                for X in tri:
                    P.add(T["spoke"][t].format(A=X))
                    for Y in tri:
                        if X != Y:
                            P.add(T["pair"][t].format(A=X, B=Y))
    return sorted(P), rp, rt


def _perms3(tri):
    a, b, c = tri
    return [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], torch_dtype=torch.float32)
    model.eval()
    return tok, model


def forward_all(tok, model, prompts, bs=16):
    """Residual stream at the last token, every layer (L+1, d), and the
    next-token log-probs there. Right padding + causal attention means the
    last real token's state is pad-independent."""
    H, LP = {}, {}
    t0 = time.time()
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last = enc["attention_mask"].sum(1) - 1
        hs = torch.stack(out.hidden_states, 0)  # (L+1, B, T, d)
        for j, p in enumerate(batch):
            H[p] = hs[:, j, last[j], :].numpy().astype(np.float32)
            LP[p] = torch.log_softmax(out.logits[j, last[j]].float(), -1).numpy()
        if (i // bs) % 20 == 0:
            print(f"  {i + len(batch)}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    return H, LP


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def frac_outside(v, basis):
    """||v_perp|| / ||v|| after least-squares projection of v onto span(basis)."""
    B = np.stack(basis, 1)  # (d, k)
    coef, *_ = np.linalg.lstsq(B, v, rcond=None)
    r = v - B @ coef
    n = np.linalg.norm(v)
    return float(np.linalg.norm(r) / n) if n > 0 else float("nan")


def additive_r2(h, h0, hA, hB):
    """R^2 of h ~ h0 + a(hA-h0) + b(hB-h0), a,b free per observation."""
    y = h - h0
    B = np.stack([hA - h0, hB - h0], 1)
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ coef
    ss = float(np.dot(y, y))
    return 1.0 - float(np.dot(r, r)) / ss if ss > 0 else float("nan")


def additive_r2_frame(h, h0, hA, hB, f):
    """Same model plus a free-coefficient frame term f: the mean over random-
    noun pairs of h(XY) - h0 in the same carrier. Absorbs tokens ('and', ',')
    that neither spoke nor bare template carries. Post-hoc; see RESULTS."""
    y = h - h0
    B = np.stack([hA - h0, hB - h0, f], 1)
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ coef
    ss = float(np.dot(y, y))
    return 1.0 - float(np.dot(r, r)) / ss if ss > 0 else float("nan")


def kl(lp_p, lp_q):
    p = np.exp(lp_p)
    return float(np.sum(p * (lp_p - lp_q)))


def top_pc_ratio(V):
    """Explained-variance ratio of PC1 of the rows of V (centred)."""
    X = V - V.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    var = s ** 2
    return (var / var.sum()).tolist()


def mean_abs_cos(V):
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = U @ U.T
    iu = np.triu_indices(len(U), 1)
    return float(np.abs(C[iu]).mean())


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def analyse(name, tok, H, LP, rp, rt):
    L = next(iter(H.values())).shape[0] - 1  # transformer layers
    focus = {"third": round(L / 3), "half": round(L / 2), "twothirds": round(2 * L / 3), "last2": L - 2}
    layers = list(range(L + 1))
    single = {w: len(tok.encode(" " + w)) == 1 for w in
              set(sum(PAIRS, ()) + sum(rp, ()) + sum(TRIPLES, ()) + sum(rt, ()))}

    # per (condition, pairset, pair, template): per-layer arrays ----------------
    per = {}  # key -> dict of per-layer lists
    rng = np.random.default_rng(SEED)
    # frame vectors: per (cond, template, random pair index) h(XY)-h0 both orders
    frame = {}
    for c in CONDS:
        T = TEMPLATES[c]
        for t in range(NT):
            h0 = H[T["bare"][t]]
            frame[(c, t)] = [0.5 * ((H[T["pair"][t].format(A=X, B=Y)] - h0) + (H[T["pair"][t].format(A=Y, B=X)] - h0))
                             for X, Y in rp]
    for c in CONDS:
        T = TEMPLATES[c]
        for setname, pairs in (("real", PAIRS), ("random", rp)):
            for pi, (A, B) in enumerate(pairs):
                for t in range(NT):
                    hAB = H[T["pair"][t].format(A=A, B=B)]
                    hBA = H[T["pair"][t].format(A=B, B=A)]
                    hA = H[T["spoke"][t].format(A=A)]
                    hB = H[T["spoke"][t].format(A=B)]
                    h0 = H[T["bare"][t]]
                    # a different pair's spokes: baseline for P3
                    oj = int(rng.integers(len(pairs)))
                    while oj == pi:
                        oj = int(rng.integers(len(pairs)))
                    oA, oB = pairs[oj]
                    ohA = H[T["spoke"][t].format(A=oA)]
                    ohB = H[T["spoke"][t].format(A=oB)]
                    K = hAB - hBA
                    fv = [f for j, f in enumerate(frame[(c, t)]) if not (setname == "random" and j == pi)]
                    fmean = np.mean(fv, 0)  # leave-one-out for the random set
                    rec = {"relK": [], "absK": [], "cosAB": [], "outside": [], "outside_other": [],
                           "r2_AB": [], "r2_BA": [], "r2f_AB": [], "r2f_BA": []}
                    for ly in layers:
                        k = K[ly]
                        dAB = hAB[ly] - h0[ly]
                        rec["absK"].append(float(np.linalg.norm(k)))
                        rec["relK"].append(float(np.linalg.norm(k) / np.linalg.norm(dAB)))
                        dBA = hBA[ly] - h0[ly]
                        rec["cosAB"].append(float(np.dot(dAB, dBA) / (np.linalg.norm(dAB) * np.linalg.norm(dBA))))
                        rec["outside"].append(frac_outside(k, [h0[ly], hA[ly], hB[ly]]))
                        rec["outside_other"].append(frac_outside(k, [h0[ly], ohA[ly], ohB[ly]]))
                        rec["r2_AB"].append(additive_r2(hAB[ly], h0[ly], hA[ly], hB[ly]))
                        rec["r2_BA"].append(additive_r2(hBA[ly], h0[ly], hA[ly], hB[ly]))
                        rec["r2f_AB"].append(additive_r2_frame(hAB[ly], h0[ly], hA[ly], hB[ly], fmean[ly]))
                        rec["r2f_BA"].append(additive_r2_frame(hBA[ly], h0[ly], hA[ly], hB[ly], fmean[ly]))
                    pAB, pBA = T["pair"][t].format(A=A, B=B), T["pair"][t].format(A=B, B=A)
                    rec["kl"] = kl(LP[pAB], LP[pBA])
                    rec["kl_sym"] = 0.5 * (kl(LP[pAB], LP[pBA]) + kl(LP[pBA], LP[pAB]))
                    rec["single_token"] = bool(single[A] and single[B])
                    rec["K"] = K  # kept in memory only, for PCA
                    per[(c, setname, pi, t)] = rec

    def med(c, setname, key, ly, t=None, only_single=False):
        vals = [r[key][ly] if isinstance(r[key], list) else r[key]
                for (cc, ss, pi, tt), r in per.items()
                if cc == c and ss == setname and (t is None or tt == t) and (not only_single or r["single_token"])]
        return float(np.median(vals)), len(vals)

    def pair_means(c, setname, key, ly, only_single=False):
        """Template-averaged value per pair (the permutation-test unit)."""
        out = []
        for pi in range(len(PAIRS if setname == "real" else rp)):
            v = [per[(c, setname, pi, t)][key][ly] if isinstance(per[(c, setname, pi, t)][key], list)
                 else per[(c, setname, pi, t)][key] for t in range(NT)]
            if only_single and not per[(c, setname, pi, 0)]["single_token"]:
                continue
            out.append(float(np.mean(v)))
        return np.array(out)

    def interaction_p(key, ly):
        """Unpaired permutation test of the real-vs-random interaction:
        log(C_real/J_real) - log(C_rand/J_rand), pair-level medians, the
        real/random label shuffled across the 40 pairs (each pair keeps its
        own COMPOUND and CONJUNCTION values together)."""
        cr, jr = pair_means("COMPOUND", "real", key, ly), pair_means("CONJUNCTION", "real", key, ly)
        cn, jn = pair_means("COMPOUND", "random", key, ly), pair_means("CONJUNCTION", "random", key, ly)
        C = np.concatenate([cr, cn]); J = np.concatenate([jr, jn])
        lab = np.array([1] * len(cr) + [0] * len(cn))
        def S(lab):
            return (math.log(np.median(C[lab == 1]) / np.median(J[lab == 1]))
                    - math.log(np.median(C[lab == 0]) / np.median(J[lab == 0])))
        obs = S(lab)
        prng = np.random.default_rng(SEED)
        cnt = sum(abs(S(prng.permutation(lab))) >= abs(obs) - 1e-12 for _ in range(N_PERM))
        return {"log_interaction": obs, "ratio_of_ratios": math.exp(obs), "p": (cnt + 1) / (N_PERM + 1)}

    def perm_p(x, y, stat="logratio"):
        """Paired label-swap permutation test on pair-level values x (COMPOUND)
        and y (CONJUNCTION). stat: log(median x / median y) or median(x - y)."""
        def S(a, b):
            if stat == "logratio":
                return math.log(np.median(a) / np.median(b))
            return float(np.median(a - b))
        obs = S(x, y)
        prng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(N_PERM):
            flip = prng.random(len(x)) < 0.5
            xa = np.where(flip, y, x)
            ya = np.where(flip, x, y)
            if abs(S(xa, ya)) >= abs(obs) - 1e-12:
                cnt += 1
        return obs, (cnt + 1) / (N_PERM + 1)

    R = {"model": MODELS[name], "n_layers": L, "d": int(next(iter(H.values())).shape[1]),
         "focus_layers": focus, "n_pairs": len(PAIRS), "n_random_pairs": len(rp),
         "single_token_pairs": int(sum(per[("COMPOUND", "real", pi, 0)]["single_token"] for pi in range(len(PAIRS)))),
         "layer_sweep": {}, "focus": {}, "P": {}, "prongs": {}}

    # layer sweep: medians of every metric, every condition, every layer -------
    for c in CONDS:
        for setname in ("real", "random"):
            R["layer_sweep"][f"{c}/{setname}"] = {
                key: [med(c, setname, key, ly)[0] for ly in layers]
                for key in ("relK", "absK", "cosAB", "outside", "outside_other", "r2_AB")}
    R["layer_sweep"]["ratio_relK_real"] = [
        med("COMPOUND", "real", "relK", ly)[0] / med("CONJUNCTION", "real", "relK", ly)[0] for ly in layers]
    R["layer_sweep"]["ratio_relK_random"] = [
        med("COMPOUND", "random", "relK", ly)[0] / med("CONJUNCTION", "random", "relK", ly)[0] for ly in layers]

    # focus-layer tables ---------------------------------------------------------
    for fname, ly in focus.items():
        F = {"layer": ly}
        for c in CONDS:
            for setname in ("real", "random"):
                F[f"{c}/{setname}"] = {
                    "relK": med(c, setname, "relK", ly)[0],
                    "relK_by_template": [med(c, setname, "relK", ly, t=t)[0] for t in range(NT)],
                    "relK_single_token": med(c, setname, "relK", ly, only_single=True)[0],
                    "cosAB": med(c, setname, "cosAB", ly)[0],
                    "outside": med(c, setname, "outside", ly)[0],
                    "outside_other": med(c, setname, "outside_other", ly)[0],
                    "r2": float(np.median([per[k]["r2_AB"][ly] for k in per if k[0] == c and k[1] == setname]
                                          + [per[k]["r2_BA"][ly] for k in per if k[0] == c and k[1] == setname])),
                    "r2_frame": float(np.median([per[k]["r2f_AB"][ly] for k in per if k[0] == c and k[1] == setname]
                                                + [per[k]["r2f_BA"][ly] for k in per if k[0] == c and k[1] == setname])),
                    "kl": med(c, setname, "kl", ly)[0],
                    "kl_sym": med(c, setname, "kl_sym", ly)[0],
                }
        # PCA of commutators (COMPOUND real, all pairs x templates) and cosine structure
        for c in ("COMPOUND", "CONJUNCTION"):
            Ks = np.stack([per[k]["K"][ly] for k in sorted(per) if k[0] == c and k[1] == "real"])
            Kn = Ks / np.linalg.norm(Ks, axis=1, keepdims=True)
            F[f"{c}/pca"] = {"pc1_raw": top_pc_ratio(Ks)[0], "pc1_unit": top_pc_ratio(Kn)[0],
                             "spectrum_raw_top10": top_pc_ratio(Ks)[:10],
                             "mean_abs_cos": mean_abs_cos(Ks), "n": len(Ks)}
            # per-template PCA (n=20 each) so the template count does not inflate the spread
            F[f"{c}/pca"]["pc1_raw_by_template"] = [
                top_pc_ratio(np.stack([per[k]["K"][ly] for k in sorted(per) if k[0] == c and k[1] == "real" and k[3] == t]))[0]
                for t in range(NT)]
        # cross-condition: does the COMPOUND commutator direction match the CONJUNCTION one for the same pair?
        cs = []
        for pi in range(len(PAIRS)):
            for t in range(NT):
                a = per[("COMPOUND", "real", pi, t)]["K"][ly]
                b = per[("CONJUNCTION", "real", pi, t)]["K"][ly]
                cs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        F["cos_K_compound_vs_conjunction_same_pair"] = float(np.median(cs))
        # is the COMPOUND commutator just the positional one? cos with the LIST
        # commutator of the same pair, and the part of K_COMPOUND outside
        # span{K_CONJ, K_LIST} of the same pair vs of a different pair.
        for setname, n in (("real", len(PAIRS)), ("random", len(rp))):
            cl, cj, oo, oth = [], [], [], []
            for pi in range(n):
                oj = (pi + 7) % n
                for t in range(NT):
                    kc = per[("COMPOUND", setname, pi, t)]["K"][ly]
                    kj = per[("CONJUNCTION", setname, pi, t)]["K"][ly]
                    kl_ = per[("LIST", setname, pi, t)]["K"][ly]
                    cl.append(float(np.dot(kc, kl_) / (np.linalg.norm(kc) * np.linalg.norm(kl_))))
                    cj.append(float(np.dot(kj, kl_) / (np.linalg.norm(kj) * np.linalg.norm(kl_))))
                    oo.append(frac_outside(kc, [kj, kl_]))
                    oth.append(frac_outside(kc, [per[("CONJUNCTION", setname, oj, t)]["K"][ly],
                                                 per[("LIST", setname, oj, t)]["K"][ly]]))
            F[f"Kdir/{setname}"] = {"cos_Kcompound_Klist": float(np.median(cl)),
                                    "cos_Kconj_Klist": float(np.median(cj)),
                                    "Kcompound_outside_span_Kconj_Klist_own": float(np.median(oo)),
                                    "Kcompound_outside_span_Kconj_Klist_otherpair": float(np.median(oth))}
        F["interaction_relK"] = interaction_p("relK", ly)
        F["interaction_r2"] = interaction_p("r2_AB", ly)
        # permutation tests (pair-level, template-averaged)
        x, y = pair_means("COMPOUND", "real", "relK", ly), pair_means("CONJUNCTION", "real", "relK", ly)
        F["P1_perm"] = dict(zip(("log_ratio", "p"), perm_p(x, y)))
        F["P1_ratio_pairlevel"] = float(np.median(x) / np.median(y))
        F["P1_pairs_with_compound_larger"] = int((x > y).sum())
        xr, yr = pair_means("COMPOUND", "random", "relK", ly), pair_means("CONJUNCTION", "random", "relK", ly)
        F["P1_ratio_random_pairlevel"] = float(np.median(xr) / np.median(yr))
        xs, ys = pair_means("COMPOUND", "real", "relK", ly, True), pair_means("CONJUNCTION", "real", "relK", ly, True)
        F["P1_ratio_single_token"] = float(np.median(xs) / np.median(ys)) if len(xs) else None
        x5 = pair_means("COMPOUND", "real", "r2_AB", ly)
        y5 = pair_means("CONJUNCTION", "real", "r2_AB", ly)
        F["P5_perm"] = dict(zip(("median_diff_conj_minus_comp", "p"), perm_p(y5, x5, stat="diff")))
        F["P5_pairs_with_conj_higher"] = int((y5 > x5).sum())
        F["P5_frame_perm"] = dict(zip(("median_diff_conj_minus_comp", "p"),
                                      perm_p(pair_means("CONJUNCTION", "real", "r2f_AB", ly),
                                             pair_means("COMPOUND", "real", "r2f_AB", ly), stat="diff")))
        F["P5_perm_random"] = dict(zip(("median_diff_conj_minus_comp", "p"),
                                       perm_p(pair_means("CONJUNCTION", "random", "r2_AB", ly),
                                              pair_means("COMPOUND", "random", "r2_AB", ly), stat="diff")))
        # P3 permutation: own-spoke outside vs other-pair-spoke outside (paired)
        o_own = pair_means("COMPOUND", "real", "outside", ly)
        o_oth = pair_means("COMPOUND", "real", "outside_other", ly)
        F["P3_own_vs_other_perm"] = dict(zip(("median_diff_own_minus_other", "p"), perm_p(o_own, o_oth, stat="diff")))
        R["focus"][fname] = F

    # P6 behavioural (layer-free) --------------------------------------------------
    k6 = {}
    for c in CONDS:
        for setname in ("real", "random"):
            k6[f"{c}/{setname}"] = med(c, setname, "kl", 0)[0]
            k6[f"{c}/{setname}/sym"] = med(c, setname, "kl_sym", 0)[0]
    kx, ky = pair_means("COMPOUND", "real", "kl", 0), pair_means("CONJUNCTION", "real", "kl", 0)
    k6["ratio_real"] = float(np.median(kx) / np.median(ky))
    k6["ratio_random"] = float(np.median(pair_means("COMPOUND", "random", "kl", 0)) /
                               np.median(pair_means("CONJUNCTION", "random", "kl", 0)))
    k6["perm"] = dict(zip(("log_ratio", "p"), perm_p(kx, ky)))
    k6["interaction"] = interaction_p("kl", 0)
    k6["pairs_with_compound_larger"] = int((kx > ky).sum())
    R["P6"] = k6

    # P4 triples -------------------------------------------------------------------
    P4 = {}
    tri_frame = {}  # (c, t) -> per random triple, mean over its 6 perms of h(XYZ)-h0
    for c in CONDS:
        T = TEMPLATES[c]
        for t in range(NT):
            h0 = H[T["bare"][t]]
            tri_frame[(c, t)] = [np.mean([H[T["triple"][t].format(A=p[0], B=p[1], C=p[2])] - h0 for p in _perms3(tri)], 0)
                                 for tri in rt]

    def tri_vals(c, tris, setname, ly, with_frame):
        """Per triple: mean over templates x 6 orders of the fraction of
        h(XYZ)-h0 outside span{spokes, 6 ordered pairs[, frame mean]}."""
        T = TEMPLATES[c]
        out = []
        for ti, tri in enumerate(tris):
            v = []
            for t in range(NT):
                h0 = H[T["bare"][t]][ly]
                basis = [H[T["spoke"][t].format(A=X)][ly] - h0 for X in tri]
                basis += [H[T["pair"][t].format(A=X, B=Y)][ly] - h0 for X in tri for Y in tri if X != Y]
                if with_frame:
                    fv = [f[ly] for j, f in enumerate(tri_frame[(c, t)]) if not (setname == "random" and j == ti)]
                    basis.append(np.mean(fv, 0))
                for perm in _perms3(tri):
                    h = H[T["triple"][t].format(A=perm[0], B=perm[1], C=perm[2])][ly] - h0
                    v.append(frac_outside(h, basis))
            out.append(float(np.mean(v)))
        return np.array(out)

    for fname, ly in focus.items():
        row = {}
        for c in CONDS:
            for setname, tris in (("real", TRIPLES), ("random", rt)):
                row[f"{c}/{setname}"] = float(np.median(tri_vals(c, tris, setname, ly, False)))
                row[f"{c}/{setname}/frame"] = float(np.median(tri_vals(c, tris, setname, ly, True)))
        row["perm"] = dict(zip(("median_diff_comp_minus_conj", "p"),
                               perm_p(tri_vals("COMPOUND", TRIPLES, "real", ly, False),
                                      tri_vals("CONJUNCTION", TRIPLES, "real", ly, False), stat="diff")))
        row["perm_frame"] = dict(zip(("median_diff_comp_minus_conj", "p"),
                                     perm_p(tri_vals("COMPOUND", TRIPLES, "real", ly, True),
                                            tri_vals("CONJUNCTION", TRIPLES, "real", ly, True), stat="diff")))
        row["perm_random"] = dict(zip(("median_diff_comp_minus_conj", "p"),
                                      perm_p(tri_vals("COMPOUND", rt, "random", ly, False),
                                             tri_vals("CONJUNCTION", rt, "random", ly, False), stat="diff")))
        row["perm_random_frame"] = dict(zip(("median_diff_comp_minus_conj", "p"),
                                            perm_p(tri_vals("COMPOUND", rt, "random", ly, True),
                                                   tri_vals("CONJUNCTION", rt, "random", ly, True), stat="diff")))
        P4[fname] = row
    R["P4"] = P4

    # scoring ----------------------------------------------------------------------
    late = ["half", "twothirds", "last2"]
    p1 = [R["focus"][f]["P1_ratio_pairlevel"] for f in late]
    R["P"]["P1"] = {"ratios_mid_late": p1, "verdict": _verdict(sum(r >= 1.5 for r in p1), len(late)),
                    "p_values": [R["focus"][f]["P1_perm"]["p"] for f in late],
                    "random_control_ratios": [R["focus"][f]["P1_ratio_random_pairlevel"] for f in late],
                    "interaction": [R["focus"][f]["interaction_relK"] for f in late]}
    p2 = [R["focus"][f]["COMPOUND/pca"]["pc1_raw"] for f in late]
    R["P"]["P2"] = {"pc1_mid_late": p2, "verdict": _verdict(sum(v < 0.5 for v in p2), len(late)),
                    "pc1_unit_mid_late": [R["focus"][f]["COMPOUND/pca"]["pc1_unit"] for f in late]}
    p3 = [R["focus"][f]["COMPOUND/real"]["outside"] for f in late]
    R["P"]["P3"] = {"outside_mid_late": p3, "verdict": _verdict(sum(v > 0.7 for v in p3), len(late)),
                    "other_pair_baseline": [R["focus"][f]["COMPOUND/real"]["outside_other"] for f in late]}
    p4 = [(P4[f]["COMPOUND/real"] > P4[f]["CONJUNCTION/real"]) and P4[f]["perm"]["p"] < 0.05 for f in late]
    R["P"]["P4"] = {"compound_gt_conj_sig_mid_late": p4, "verdict": _verdict(sum(p4), len(late)),
                    "p_values": [P4[f]["perm"]["p"] for f in late],
                    "outside_compound": [P4[f]["COMPOUND/real"] for f in late],
                    "outside_conjunction": [P4[f]["CONJUNCTION/real"] for f in late],
                    "frame_corrected": {"outside_compound": [P4[f]["COMPOUND/real/frame"] for f in late],
                                        "outside_conjunction": [P4[f]["CONJUNCTION/real/frame"] for f in late],
                                        "p_values": [P4[f]["perm_frame"]["p"] for f in late]}}
    p5 = [(R["focus"][f]["CONJUNCTION/real"]["r2"] > R["focus"][f]["COMPOUND/real"]["r2"]) and R["focus"][f]["P5_perm"]["p"] < 0.05 for f in late]
    R["P"]["P5"] = {"conj_gt_comp_sig_mid_late": p5, "verdict": _verdict(sum(p5), len(late)),
                    "r2_compound": [R["focus"][f]["COMPOUND/real"]["r2"] for f in late],
                    "r2_conjunction": [R["focus"][f]["CONJUNCTION/real"]["r2"] for f in late],
                    "p_values": [R["focus"][f]["P5_perm"]["p"] for f in late],
                    "frame_corrected": {"r2_compound": [R["focus"][f]["COMPOUND/real"]["r2_frame"] for f in late],
                                        "r2_conjunction": [R["focus"][f]["CONJUNCTION/real"]["r2_frame"] for f in late],
                                        "p_values": [R["focus"][f]["P5_frame_perm"]["p"] for f in late]}}
    R["P"]["P6"] = {"ratio": k6["ratio_real"], "ratio_random": k6["ratio_random"],
                    # "by >=2x, controlling for RANDOM": the ratio must clear 2 AND exceed
                    # the same ratio on random nouns (interaction > 1, p < .05)
                    "verdict": ("RIGHT" if k6["ratio_real"] >= 2 and k6["interaction"]["ratio_of_ratios"] > 1 and k6["interaction"]["p"] < 0.05
                                else ("PARTIAL" if k6["ratio_real"] >= 2 else "WRONG")),
                    "p": k6["perm"]["p"], "interaction": k6["interaction"]}
    # prongs
    R["prongs"]["1_positional_LIST_relK_mid_late"] = {c: [R["focus"][f][f"{c}/real"]["relK"] for f in late] for c in CONDS}
    R["prongs"]["2_single_token_ratio_mid_late"] = [R["focus"][f]["P1_ratio_single_token"] for f in late]
    sweep = R["layer_sweep"]["ratio_relK_real"]
    R["prongs"]["3_layer_sweep_ratio"] = sweep
    R["prongs"]["3_gap_only_at_last_layer"] = bool(max(sweep[1:-1]) < 1.5 <= sweep[-1])
    R["prongs"]["4_perm_p_P1_mid_late"] = R["P"]["P1"]["p_values"]
    R["prongs"]["4_perm_p_P5_mid_late"] = R["P"]["P5"]["p_values"]
    return R, per


def _verdict(hits, n):
    return "RIGHT" if hits == n else ("WRONG" if hits == 0 else "PARTIAL")


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def plots(name, R, per):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = R["n_layers"]
    xs = list(range(L + 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, ls in (("COMPOUND", "-"), ("CONJUNCTION", "--"), ("LIST", ":")):
        ax[0].plot(xs, R["layer_sweep"][f"{c}/real"]["relK"], ls, label=f"{c} (real pairs)")
        ax[0].plot(xs, R["layer_sweep"][f"{c}/random"]["relK"], ls, alpha=0.4, label=f"{c} (random nouns)")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("median |K| / |h(AB) - h0|"); ax[0].set_title(f"{R['model']}: commutator size by layer")
    ax[0].legend(fontsize=7)
    ax[1].plot(xs, R["layer_sweep"]["ratio_relK_real"], "-", label="real pairs")
    ax[1].plot(xs, R["layer_sweep"]["ratio_relK_random"], "-", alpha=0.5, label="random nouns")
    ax[1].axhline(1.5, color="k", lw=0.5, ls="--"); ax[1].axhline(1.0, color="k", lw=0.5)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("COMPOUND / CONJUNCTION median relK"); ax[1].set_title("P1 ratio by layer (1.5 = threshold)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}_relK_by_layer.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, len(R["focus"]), figsize=(3.2 * len(R["focus"]), 3.4), sharey=True)
    for i, (fname, F) in enumerate(R["focus"].items()):
        for c, m in (("COMPOUND", "o-"), ("CONJUNCTION", "s--")):
            spec = F[f"{c}/pca"]["spectrum_raw_top10"]
            ax[i].plot(range(1, len(spec) + 1), spec, m, ms=3, label=c)
        ax[i].set_title(f"{fname} (L{F['layer']})"); ax[i].set_xlabel("PC"); ax[i].axhline(0.5, color="k", lw=0.5, ls="--")
    ax[0].set_ylabel("explained variance ratio"); ax[0].legend(fontsize=8)
    fig.suptitle(f"{R['model']}: PC spectrum of the 60 commutators {{K_i}}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}_K_pc_spectrum.png"), dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# report: markdown tables from results/*.json (what RESULTS.md quotes)
# ----------------------------------------------------------------------------
def report(names):
    Rs = {}
    for n in names:
        path = os.path.join(OUT, f"{n}.json")
        if os.path.exists(path):
            with open(path) as f:
                Rs[n] = json.load(f)
    late = ["half", "twothirds", "last2"]
    f3 = lambda v: "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    f2 = lambda v: f"{v:.2f}"
    out = []
    out.append("### Scorecard\n")
    out.append("| prediction | " + " | ".join(R["model"].split("/")[-1] for R in Rs.values()) + " |")
    out.append("|---|" + "---|" * len(Rs))
    for p in ("P1", "P2", "P3", "P4", "P5", "P6"):
        out.append(f"| {p} | " + " | ".join(f"**{Rs[n]['P'][p]['verdict']}**" for n in Rs) + " |")
    out.append("")
    for n, R in Rs.items():
        m = R["model"].split("/")[-1]
        L = R["n_layers"]
        out.append(f"### {m} ({L} layers, d={R['d']}, {R['n_prompts']} prompts, {R['seconds']} s)\n")
        out.append("Median |K| / |h(AB) − h0| over 20 pairs × 3 templates (real) and 20 random-noun pairs × 3 templates; layers = 1/2, 2/3, last−2.\n")
        out.append("| condition | " + " | ".join(f"L{R['focus'][f]['layer']}" for f in late) + " |")
        out.append("|---|---|---|---|")
        for c in CONDS:
            for s_ in ("real", "random"):
                out.append(f"| {c} / {s_} | " + " | ".join(f3(R["focus"][f][f"{c}/{s_}"]["relK"]) for f in late) + " |")
        out.append("| **P1 ratio COMPOUND/CONJUNCTION, real (pair-level)** | " + " | ".join(f2(R["focus"][f]["P1_ratio_pairlevel"]) for f in late) + " |")
        out.append("| ratio, random nouns | " + " | ".join(f2(R["focus"][f]["P1_ratio_random_pairlevel"]) for f in late) + " |")
        out.append("| ratio of ratios (real / random), p | " + " | ".join(f"{f2(R['focus'][f]['interaction_relK']['ratio_of_ratios'])} (p={R['focus'][f]['interaction_relK']['p']:.2f})" for f in late) + " |")
        out.append("| pairs with COMPOUND > CONJUNCTION (of 20), perm p | " + " | ".join(f"{R['focus'][f]['P1_pairs_with_compound_larger']} (p={R['focus'][f]['P1_perm']['p']:.3f})" for f in late) + " |")
        out.append("| P1 ratio by template (real) | " + " | ".join("/".join(f2(R["focus"][f]["COMPOUND/real"]["relK_by_template"][t] / R["focus"][f]["CONJUNCTION/real"]["relK_by_template"][t]) for t in range(NT)) for f in late) + " |")
        out.append("")
        out.append("Commutator direction (P2 and the positional prong): PC1 explained-variance of the 60 COMPOUND commutators, mean |cos| between them, and how much of K_COMPOUND is the same-pair CONJUNCTION / LIST commutator.\n")
        out.append("| quantity | " + " | ".join(f"L{R['focus'][f]['layer']}" for f in late) + " |")
        out.append("|---|---|---|---|")
        out.append("| PC1 ratio, COMPOUND K (raw / unit-normed) | " + " | ".join(f"{f2(R['focus'][f]['COMPOUND/pca']['pc1_raw'])} / {f2(R['focus'][f]['COMPOUND/pca']['pc1_unit'])}" for f in late) + " |")
        out.append("| PC1 ratio per template, n=20 each | " + " | ".join("/".join(f2(v) for v in R["focus"][f]["COMPOUND/pca"]["pc1_raw_by_template"]) for f in late) + " |")
        out.append("| PC1 ratio, CONJUNCTION K | " + " | ".join(f2(R["focus"][f]["CONJUNCTION/pca"]["pc1_raw"]) for f in late) + " |")
        out.append("| mean abs cos between COMPOUND commutators | " + " | ".join(f2(R["focus"][f]["COMPOUND/pca"]["mean_abs_cos"]) for f in late) + " |")
        out.append("| median cos(K_COMPOUND, K_CONJUNCTION) same pair | " + " | ".join(f2(R["focus"][f]["cos_K_compound_vs_conjunction_same_pair"]) for f in late) + " |")
        out.append("| median cos(K_COMPOUND, K_LIST) same pair, real / random | " + " | ".join(f"{f2(R['focus'][f]['Kdir/real']['cos_Kcompound_Klist'])} / {f2(R['focus'][f]['Kdir/random']['cos_Kcompound_Klist'])}" for f in late) + " |")
        out.append("| K_COMPOUND outside span{K_CONJ, K_LIST}: own pair / other pair | " + " | ".join(f"{f2(R['focus'][f]['Kdir/real']['Kcompound_outside_span_Kconj_Klist_own'])} / {f2(R['focus'][f]['Kdir/real']['Kcompound_outside_span_Kconj_Klist_otherpair'])}" for f in late) + " |")
        out.append("")
        out.append("P3 (K outside span{h0, hA, hB}) and P5 (additive fit R²), medians; `frame` adds a free frame-mean term (post-hoc, see text).\n")
        out.append("| quantity | " + " | ".join(f"L{R['focus'][f]['layer']}" for f in late) + " |")
        out.append("|---|---|---|---|")
        out.append("| P3 outside, own spokes / other pair's spokes | " + " | ".join(f"{f3(R['focus'][f]['COMPOUND/real']['outside'])} / {f3(R['focus'][f]['COMPOUND/real']['outside_other'])}" for f in late) + " |")
        out.append("| P3 own − other, perm p | " + " | ".join(f"{R['focus'][f]['P3_own_vs_other_perm']['median_diff_own_minus_other']:+.3f} (p={R['focus'][f]['P3_own_vs_other_perm']['p']:.3f})" for f in late) + " |")
        out.append("| P5 R² COMPOUND / CONJUNCTION | " + " | ".join(f"{f3(R['focus'][f]['COMPOUND/real']['r2'])} / {f3(R['focus'][f]['CONJUNCTION/real']['r2'])}" for f in late) + " |")
        out.append("| P5 pairs with CONJUNCTION higher (of 20), perm p | " + " | ".join(f"{R['focus'][f]['P5_pairs_with_conj_higher']} (p={R['focus'][f]['P5_perm']['p']:.3f})" for f in late) + " |")
        out.append("| P5 R² random nouns COMPOUND / CONJUNCTION | " + " | ".join(f"{f3(R['focus'][f]['COMPOUND/random']['r2'])} / {f3(R['focus'][f]['CONJUNCTION/random']['r2'])}" for f in late) + " |")
        out.append("| P5 R² with frame term, COMPOUND / CONJUNCTION, p | " + " | ".join(f"{f3(R['focus'][f]['COMPOUND/real']['r2_frame'])} / {f3(R['focus'][f]['CONJUNCTION/real']['r2_frame'])} (p={R['focus'][f]['P5_frame_perm']['p']:.3f})" for f in late) + " |")
        out.append("")
        out.append("P4 (three-word compositions): median fraction of h(ABC) − h0 outside span{3 spokes, 6 ordered pairs}, 12 triples × 6 orders × 3 templates.\n")
        out.append("| condition | " + " | ".join(f"L{R['focus'][f]['layer']}" for f in late) + " |")
        out.append("|---|---|---|---|")
        for c in CONDS:
            for s_ in ("real", "random"):
                out.append(f"| {c} / {s_} | " + " | ".join(f3(R["P4"][f][f"{c}/{s_}"]) for f in late) + " |")
        out.append("| COMPOUND − CONJUNCTION real, perm p | " + " | ".join(f"{R['P4'][f]['perm']['median_diff_comp_minus_conj']:+.3f} (p={R['P4'][f]['perm']['p']:.3f})" for f in late) + " |")
        out.append("| same with frame term | " + " | ".join(f"{R['P4'][f]['perm_frame']['median_diff_comp_minus_conj']:+.3f} (p={R['P4'][f]['perm_frame']['p']:.3f})" for f in late) + " |")
        out.append("")
        k = R["P6"]
        out.append("P6 (behaviour): median KL(next-token | AB ∥ BA) at the `:` token.\n")
        out.append("| condition | real pairs | random nouns |")
        out.append("|---|---|---|")
        for c in CONDS:
            out.append(f"| {c} | {k[f'{c}/real']:.4f} | {k[f'{c}/random']:.4f} |")
        out.append(f"| COMPOUND / CONJUNCTION | {k['ratio_real']:.2f} (perm p={k['perm']['p']:.3f}) | {k['ratio_random']:.2f} |")
        out.append(f"| ratio of ratios | {k['interaction']['ratio_of_ratios']:.2f} (p={k['interaction']['p']:.2f}) | |")
        out.append("")
        sw = R["layer_sweep"]["ratio_relK_real"]
        swr = R["layer_sweep"]["ratio_relK_random"]
        out.append("Layer sweep of the P1 ratio (real / random nouns): " + ", ".join(f"L{ly}: {sw[ly]:.2f}/{swr[ly]:.2f}" for ly in range(1, L + 1)) + "\n")
        out.append(f"Single-token pairs: {R['single_token_pairs']}/20. Gap only at the last layer: {R['prongs']['3_gap_only_at_last_layer']}.\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--report", action="store_true", help="print markdown tables from results/*.json and exit")
    args = ap.parse_args()
    if args.report:
        report(args.models)
        return
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    prompts, rp, rt = build_prompts()
    print(f"{len(prompts)} distinct prompts; random pairs {rp}", flush=True)
    for name in args.models:
        t0 = time.time()
        print(f"== {name} ({MODELS[name]})", flush=True)
        tok, model = load(name)
        H, LP = forward_all(tok, model, prompts, args.bs)
        del model
        R, per = analyse(name, tok, H, LP, rp, rt)
        R["random_pairs"] = rp
        R["random_triples"] = rt
        R["n_prompts"] = len(prompts)
        R["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(R, f, indent=1)
        plots(name, R, per)
        print(json.dumps(R["P"], indent=1), flush=True)
        print(f"   done in {R['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
