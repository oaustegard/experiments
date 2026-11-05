"""Arms A0 (zero-shot cosine), A1 (frozen head, specialist) and A2 (frozen head, leave one workflow out).

    python3 -I run/arms.py a0            # choose prompt/template/τ on TD train, score TD test + unrelated set
    python3 -I run/arms.py a1 a2         # heads on the A0 choice (needs results/a0.json)

Everything reads cached EG2 vectors (run/embed_states.py fills the state cache first).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "browser-decider" / "run"))
from common import NOUL_DEFAULT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402
from enc import EG2, encoder, render, results_dir  # noqa: E402

OOD = ROOT.parent / "browser-decider" / "data" / "decider_eval.jsonl"
RES = results_dir()
TYPES = ["choice", "score", "noul"]
TEMPLATES = {
    "desc": "{desc}",
    "instr_desc": "{instr} {desc}",
    "name_desc": "{name}: {desc}",
    "instr_name_desc": "{instr} Answer: {name}. {desc}",
}
STATE_PROMPTS = ["doc", "cls"]
OPT_PROMPTS = ["query", "cls", "sts"]


def convert_ood(row):
    """Same conversion as browser-decider/run/eval_ood.py."""
    q = row["questions"][0]
    lab = row["labels"][q["name"]]
    if q["type"] == "choice":
        opts = [(c["value"], c["description"]) for c in q["choices"]]
        gi = [o for o, _ in opts].index(lab)
        typ = "choice"
    elif q["type"] == "predicate":
        opts = list(NOUL_DEFAULT.items())
        gi = 0 if lab else 1
        typ = "noul"
    else:
        opts = [(str(i), f"{lv['label']}: {lv['description']}") for i, lv in enumerate(q["levels"])]
        gi = int(lab)
        typ = "score"
    gold = [0.0] * len(opts)
    gold[gi] = 1.0
    return {"case": row["id"], "workflow": row["source"].split(" ")[0], "qname": q["name"], "type": typ,
            "question": f"{typ}: {q['instructions']}", "instructions": q["instructions"], "options": opts,
            "state": row["input"], "gold": gold, "subset": {"choice": "banking", "predicate": "predicates",
                                                          "score": "rubric"}[q["type"]]}


def load_all():
    td = {}
    for w in WORKFLOWS:
        for split in ("train", "test"):
            td[(w, split)] = examples(load_cases(w, split))
    for exs in td.values():
        for e in exs:
            e["state_text"] = render(e["state"])
    ood = [convert_ood(json.loads(l)) for l in open(OOD)]
    for e in ood:
        e["state_text"] = e["state"]
    return td, ood


def opt_text(e, name, desc, tmpl):
    name = name.replace("_", " ")
    if desc in ("", name) or desc == name.replace(" ", "_"):  # bare label (the mix): don't say it twice
        tmpl = {"desc": "name_desc", "instr_desc": "instr_name_desc"}.get(tmpl, tmpl)
        return TEMPLATES[tmpl].replace(": {desc}", "").replace(" {desc}", "").format(
            instr=e["instructions"], name=name).strip()
    return TEMPLATES[tmpl].format(instr=e["instructions"], name=name, desc=desc)


class Vecs:
    """Looks up state, question and option vectors for a list of examples under one configuration."""

    _enc = {}

    def __init__(self, sp, op, tmpl):
        self.sp, self.op, self.tmpl = sp, op, tmpl
        if not Vecs._enc:
            Vecs._enc["states"] = encoder()  # filled by embed_states.py
            Vecs._enc["small"] = encoder(tag="small")  # questions and options, its own writer
        self.states, self.small = Vecs._enc["states"], Vecs._enc["small"]

    def __call__(self, exs):
        S = self.states.embed([e["state_text"] for e in exs], prompt=self.sp)
        Q = self.small.embed([e["question"] for e in exs], prompt=self.op)
        flat = [opt_text(e, n, d, self.tmpl) for e in exs for n, d in e["options"]]
        F = self.small.embed(flat, prompt=self.op)
        O, i = [], 0
        for e in exs:
            k = len(e["options"])
            O.append(F[i:i + k])
            i += k
        return S, Q, O


def zs_probs(S, O, exs, tau):
    out = []
    for s, o, e in zip(S, O, exs):
        z = tau[e["type"]] * (o @ s)
        z = np.exp(z - z.max())
        out.append((z / z.sum()).tolist())
    return out


def fit_tau(S, O, exs):
    """Per-type temperature minimising KL to gold on the given (training) examples."""
    grid = np.concatenate([np.linspace(1, 20, 39), np.linspace(25, 200, 36)])
    tau = {}
    for t in TYPES:
        idx = [i for i, e in enumerate(exs) if e["type"] == t]
        if not idx:
            tau[t] = 20.0
            continue
        best = None
        for g in grid:
            m, _ = score_examples([exs[i] for i in idx], zs_probs([S[i] for i in idx], [O[i] for i in idx],
                                                                  [exs[i] for i in idx], {t: g}))
            if best is None or m["kl_from_gold"] < best[0]:
                best = (m["kl_from_gold"], float(g))
        tau[t] = best[1]
    return tau


def by_type(exs, probs):
    out = {}
    for t in TYPES:
        idx = [i for i, e in enumerate(exs) if e["type"] == t]
        if idx:
            out[t] = score_examples([exs[i] for i in idx], [probs[i] for i in idx])[0]["accuracy"]
    return out


def report(exs, probs, key="subset"):
    m, _ = score_examples(exs, probs)
    m["by_type"] = by_type(exs, probs)
    if key and key in exs[0]:
        m["by_subset"] = {}
        for sub in sorted({e[key] for e in exs}):
            idx = [i for i, e in enumerate(exs) if e[key] == sub]
            m["by_subset"][sub] = score_examples([exs[i] for i in idx], [probs[i] for i in idx])[0]["accuracy"]
    return m


def a0():
    td, ood = load_all()
    train = [e for w in WORKFLOWS for e in td[(w, "train")]]
    test = [e for w in WORKFLOWS for e in td[(w, "test")]]
    sweep = []
    for sp in STATE_PROMPTS:
        for op in OPT_PROMPTS:
            for tmpl in TEMPLATES:
                V = Vecs(sp, op, tmpl)
                S, _, O = V(train)
                tau = fit_tau(S, O, train)
                m, _ = score_examples(train, zs_probs(S, O, train, tau))
                sweep.append({"state_prompt": sp, "opt_prompt": op, "template": tmpl, "tau": tau,
                              "train_accuracy": m["accuracy"], "train_kl": m["kl_from_gold"],
                              "train_by_type": by_type(train, zs_probs(S, O, train, tau))})
                print(f"{sp:4s} {op:6s} {tmpl:16s} train acc {m['accuracy']:.4f} kl {m['kl_from_gold']:.3f}",
                      flush=True)
    best = max(sweep, key=lambda r: r["train_accuracy"])
    V = Vecs(best["state_prompt"], best["opt_prompt"], best["template"])
    tau = best["tau"]
    S, _, O = V(test)
    test_m = report(test, zs_probs(S, O, test, tau), key="workflow")
    S2, _, O2 = V(ood)
    ood_m = report(ood, zs_probs(S2, O2, ood, tau))
    # same choice with τ refit per type is not allowed on test; the oracle config is reported for scale only
    out = {"choice": {k: best[k] for k in ("state_prompt", "opt_prompt", "template", "tau")},
           "td_test": test_m, "ood": ood_m, "sweep": sweep}
    RES.mkdir(exist_ok=True)
    (RES / "a0.json").write_text(json.dumps(out, indent=1))
    print("A0 choice", out["choice"])
    print("A0 TD test", json.dumps({k: test_m[k] for k in ("accuracy", "kl_from_gold", "brier", "ece")}),
          test_m["by_type"], test_m.get("by_subset"))
    print("A0 OOD", ood_m["accuracy"], ood_m["by_subset"])


# --- frozen-encoder head -------------------------------------------------------------------------

def head_module(tau0, hidden=256, drop=0.1):
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        """field = norm(s + MLP([s, q, s*q])); logit_i = tau_type * cos(field, o_i) + r([s*o_i, q*o_i]).

        Both MLPs end in a zero-initialised layer, so the untrained head is exactly A0.
        """

        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop)
            self.f1, self.f2 = nn.Linear(3 * 768, hidden), nn.Linear(hidden, 768)
            self.r1, self.r2 = nn.Linear(2 * 768, 128), nn.Linear(128, 1)
            for lin in (self.f2, self.r2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.log_tau = nn.Parameter(torch.log(torch.tensor([tau0[t] for t in TYPES], dtype=torch.float32)))

        def forward(self, s, q, o, omask, t):
            # s, q: [B,768]; o: [B,K,768]; omask: [B,K] bool; t: [B] type index
            x = self.drop(torch.cat([s, q, s * q], -1))
            field = nn.functional.normalize(s + self.f2(torch.relu(self.f1(x))), dim=-1)
            cos = torch.einsum("bd,bkd->bk", field, o)
            so = torch.cat([s[:, None, :] * o, q[:, None, :] * o], -1)
            r = self.r2(torch.relu(self.r1(self.drop(so)))).squeeze(-1)
            z = self.log_tau.exp()[t][:, None] * cos + r
            return z.masked_fill(~omask, -1e9)

    return Head()


def tensors(V, exs):
    import torch
    S, Q, O = V(exs)
    K = max(len(o) for o in O)
    Ob = np.zeros((len(exs), K, 768), np.float32)
    M = np.zeros((len(exs), K), bool)
    G = np.zeros((len(exs), K), np.float32)
    for i, (o, e) in enumerate(zip(O, exs)):
        Ob[i, :len(o)] = o
        M[i, :len(o)] = True
        G[i, :len(o)] = e["gold"]
    T = np.array([TYPES.index(e["type"]) for e in exs])
    return [torch.tensor(a) for a in (S, Q, Ob, M, G, T)]


def predict(head, tt, temp=None):
    import torch
    head.eval()
    with torch.no_grad():
        z = head(*tt[:4], tt[5])
        if temp is not None:
            z = z / temp[tt[5]][:, None]
        p = torch.softmax(z, -1).numpy()
    m = tt[3].numpy()
    return [row[mm].tolist() for row, mm in zip(p, m)]


def train_head(V, tau0, train, seed, epochs=40, lr=1e-3, wd=1e-2, bs=64, val_frac=0.1, init=None):
    """Soft CE vs gold; best epoch by validation KL on held-out training cases; per-type temperature on val."""
    import torch
    rng = np.random.default_rng(seed)
    cases = sorted({e["case"] for e in train})
    val_cases = set(rng.choice(cases, max(1, int(len(cases) * val_frac)), replace=False).tolist())
    tr = [e for e in train if e["case"] not in val_cases]
    va = [e for e in train if e["case"] in val_cases]
    torch.manual_seed(seed)
    head = head_module(tau0)
    if init is not None:
        head.load_state_dict(init)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    ttr, tva = tensors(V, tr), tensors(V, va)
    n = len(tr)
    best = (float("inf"), None, -1)
    for ep in range(epochs):
        head.train()
        perm = torch.tensor(rng.permutation(n))
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            S, Q, O, M, G, T = (x[idx] for x in ttr)
            z = head(S, Q, O, M, T)
            loss = -(G * torch.log_softmax(z, -1).masked_fill(~M, 0)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        kl = score_examples(va, predict(head, tva))[0]["kl_from_gold"]
        if kl < best[0]:
            best = (kl, {k: v.clone() for k, v in head.state_dict().items()}, ep)
    head.load_state_dict(best[1])
    # per-type temperature on the validation slice, fitted to KL (browser-decider's recipe)
    temp = torch.ones(len(TYPES))
    grid = np.linspace(0.5, 3.0, 26)
    for ti, t in enumerate(TYPES):
        idx = [i for i, e in enumerate(va) if e["type"] == t]
        if not idx:
            continue
        sub = [x[idx] for x in tva]
        scores = []
        for g in grid:
            tt = temp.clone()
            tt[ti] = g
            scores.append(score_examples([va[i] for i in idx], predict(head, sub, tt))[0]["kl_from_gold"])
        temp[ti] = float(grid[int(np.argmin(scores))])
    return head, temp, best[2]


def a0_choice():
    c = json.loads((RES / "a0.json").read_text())["choice"]
    return Vecs(c["state_prompt"], c["opt_prompt"], c["template"]), c["tau"]


def a1(seeds=(0, 1, 2)):
    td, ood = load_all()
    V, tau0 = a0_choice()
    train = [e for w in WORKFLOWS for e in td[(w, "train")]]
    test = [e for w in WORKFLOWS for e in td[(w, "test")]]
    tte, tood = tensors(V, test), tensors(V, ood)
    runs = []
    for s in seeds:
        head, temp, ep = train_head(V, tau0, train, s)
        r = {"seed": s, "best_epoch": ep, "temp": temp.tolist(),
             "td_test": report(test, predict(head, tte, temp), key="workflow"),
             "ood": report(ood, predict(head, tood, temp))}
        print(f"A1 seed {s}: TD test {r['td_test']['accuracy']:.4f} {r['td_test']['by_type']} "
              f"OOD {r['ood']['accuracy']:.3f} {r['ood']['by_subset']} (best ep {ep})", flush=True)
        runs.append(r)
    out = {"runs": runs, "mean_td_test_accuracy": float(np.mean([r["td_test"]["accuracy"] for r in runs])),
           "sd_td_test_accuracy": float(np.std([r["td_test"]["accuracy"] for r in runs])),
           "mean_ood_accuracy": float(np.mean([r["ood"]["accuracy"] for r in runs]))}
    (RES / "a1.json").write_text(json.dumps(out, indent=1))
    print("A1", {k: v for k, v in out.items() if k != "runs"})


def a2(seeds=(0, 1, 2)):
    td, ood = load_all()
    V, tau0 = a0_choice()
    runs = []
    for held in WORKFLOWS:
        train = [e for w in WORKFLOWS if w != held for e in td[(w, "train")]]
        test = td[(held, "test")]
        tte = tensors(V, test)
        for s in seeds:
            head, temp, ep = train_head(V, tau0, train, s)
            r = {"held_out": held, "seed": s, "best_epoch": ep, "td_test": report(test, predict(head, tte, temp))}
            print(f"A2 {held} seed {s}: {r['td_test']['accuracy']:.4f} {r['td_test']['by_type']} (best ep {ep})",
                  flush=True)
            runs.append(r)
    per = {w: float(np.mean([r["td_test"]["accuracy"] for r in runs if r["held_out"] == w])) for w in WORKFLOWS}
    out = {"runs": runs, "per_workflow": per, "mean_accuracy": float(np.mean(list(per.values())))}
    (RES / "a2.json").write_text(json.dumps(out, indent=1))
    print("A2", per, out["mean_accuracy"])


class MixVecs(Vecs):
    """The mix's own cache and token caps (states 512, questions/options 256), as run/mix.py embedded them."""

    def __init__(self, sp, op, tmpl):
        super().__init__(sp, op, tmpl)
        if "mix" not in Vecs._enc:
            Vecs._enc["mix"] = encoder(tag="mix")
        self.mix = Vecs._enc["mix"]

    def __call__(self, exs):
        S = self.mix.embed([e["state_text"] for e in exs], prompt=self.sp, max_tokens=512)
        Q = self.mix.embed([e["question"] for e in exs], prompt=self.op, max_tokens=256)
        flat = [opt_text(e, n, d, self.tmpl) for e in exs for n, d in e["options"]]
        F = self.mix.embed(flat, prompt=self.op, max_tokens=256)
        O, i = [], 0
        for e in exs:
            k = len(e["options"])
            O.append(F[i:i + k])
            i += k
        return S, Q, O


class Joined:
    """Routes TD/OOD examples to the TD caches and mix rows to the mix cache, keeping order."""

    def __init__(self, td_v, mix_v):
        self.td_v, self.mix_v = td_v, mix_v

    def __call__(self, exs):
        mix_idx = [i for i, e in enumerate(exs) if e.get("_mix")]
        td_idx = [i for i, e in enumerate(exs) if not e.get("_mix")]
        S = np.zeros((len(exs), 768), np.float32)
        Q = np.zeros((len(exs), 768), np.float32)
        O = [None] * len(exs)
        for idx, v in ((td_idx, self.td_v), (mix_idx, self.mix_v)):
            if not idx:
                continue
            s, q, o = v([exs[i] for i in idx])
            S[idx], Q[idx] = s, q
            for j, i in enumerate(idx):
                O[i] = o[j]
        return S, Q, O


def mix_rows():
    import mix
    rows = mix.load()
    for e in rows:
        e["_mix"] = True
    return rows


def a3(seeds=(0, 1, 2)):
    import torch
    td, ood = load_all()
    c = json.loads((RES / "a0.json").read_text())["choice"]
    tdv = Vecs(c["state_prompt"], c["opt_prompt"], c["template"])
    V = Joined(tdv, MixVecs(c["state_prompt"], c["opt_prompt"], c["template"]))
    train = mix_rows()
    test = [e for w in WORKFLOWS for e in td[(w, "test")]]
    tte, tood = tensors(V, test), tensors(V, ood)
    runs = []
    (ROOT / "out" / "heads").mkdir(parents=True, exist_ok=True)
    for s in seeds:
        head, temp, ep = train_head(V, c["tau"], train, s, epochs=30)
        torch.save({"state": head.state_dict(), "temp": temp}, ROOT / "out" / "heads" / f"a3-s{s}.pt")
        r = {"seed": s, "best_epoch": ep, "temp": temp.tolist(),
             "td_test": report(test, predict(head, tte, temp), key="workflow"),
             "ood": report(ood, predict(head, tood, temp))}
        print(f"A3 seed {s}: TD test {r['td_test']['accuracy']:.4f} {r['td_test']['by_type']} "
              f"{r['td_test']['by_subset']} OOD {r['ood']['accuracy']:.3f} {r['ood']['by_subset']} (best ep {ep})",
              flush=True)
        runs.append(r)
    out = {"n_train_rows": len(train), "n_sources": len({e["workflow"] for e in train}), "runs": runs,
           "mean_td_test_accuracy": float(np.mean([r["td_test"]["accuracy"] for r in runs])),
           "sd_td_test_accuracy": float(np.std([r["td_test"]["accuracy"] for r in runs])),
           "mean_ood_accuracy": float(np.mean([r["ood"]["accuracy"] for r in runs])),
           "mean_ood_by_subset": {k: float(np.mean([r["ood"]["by_subset"][k] for r in runs]))
                                  for k in runs[0]["ood"]["by_subset"]}}
    (RES / "a3.json").write_text(json.dumps(out, indent=1))
    print("A3", {k: v for k, v in out.items() if k != "runs"})


def a4(ns=(16, 64, 256), draws=(0, 1, 2)):
    """Label curve: n cases of one workflow, head started from A3 (seed 0) or from A0 (fresh head)."""
    import torch
    td, _ = load_all()
    V, tau0 = a0_choice()
    a3_state = torch.load(ROOT / "out" / "heads" / "a3-s0.pt")["state"]
    runs = []
    for held in WORKFLOWS:
        test = td[(held, "test")]
        tte = tensors(V, test)
        pool = td[(held, "train")]
        cases = sorted({e["case"] for e in pool})
        for n in ns:
            for d in draws:
                pick = set(np.random.default_rng(100 + d).choice(cases, n, replace=False).tolist())
                train = [e for e in pool if e["case"] in pick]
                for start in ("a3", "a0"):
                    head, temp, ep = train_head(V, tau0, train, d, epochs=40, val_frac=0.2,
                                                init=a3_state if start == "a3" else None)
                    acc = report(test, predict(head, tte, temp))["accuracy"]
                    runs.append({"held_out": held, "n": n, "draw": d, "start": start, "best_epoch": ep,
                                 "accuracy": acc})
                    print(f"A4 {held} n={n} draw {d} from {start}: {acc:.4f} (ep {ep})", flush=True)
    curve = {}
    for start in ("a3", "a0"):
        for n in ns:
            accs = [r["accuracy"] for r in runs if r["start"] == start and r["n"] == n]
            curve[f"{start}_n{n}"] = float(np.mean(accs))
    out = {"runs": runs, "curve": curve}
    (RES / "a4.json").write_text(json.dumps(out, indent=1))
    print("A4", curve)


def a3_diag():
    """Does the A3 head learn the mix itself? Untrained head (=A0) vs A3 seed 0 on seed 0's mix validation rows."""
    import torch
    c = json.loads((RES / "a0.json").read_text())["choice"]
    V = Joined(Vecs(c["state_prompt"], c["opt_prompt"], c["template"]),
               MixVecs(c["state_prompt"], c["opt_prompt"], c["template"]))
    rows = mix_rows()
    rng = np.random.default_rng(0)  # train_head's seed-0 validation draw
    cases = sorted({e["case"] for e in rows})
    val = set(rng.choice(cases, max(1, int(len(cases) * 0.1)), replace=False).tolist())
    va = [e for e in rows if e["case"] in val]
    tva = tensors(V, va)
    h = head_module(c["tau"])
    zs = report(va, predict(h, tva), key=None)["accuracy"]
    st = torch.load(ROOT / "out" / "heads" / "a3-s0.pt")
    h.load_state_dict(st["state"])
    tr = report(va, predict(h, tva, st["temp"]), key=None)["accuracy"]
    first = float(np.mean([int(np.argmax(e["gold"]) == 0) for e in va]))
    out = {"n_val": len(va), "zero_shot": zs, "a3_seed0": tr, "first_option_rate": first}
    (RES / "a3_diag.json").write_text(json.dumps(out, indent=1))
    print("A3 diag", out)


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        {"a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a3_diag": a3_diag}[arm]()
