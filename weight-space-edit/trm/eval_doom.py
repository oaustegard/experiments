"""Hand-written vs shipped (run7, step 13.5k) TRM weights on held-out E1M9 grids, exact match against BFS.

python3 eval_doom.py OUT.json [threads]
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from handtrm import model_from, load_bin

HERE = Path(__file__).parent
SETS = [("e1m9_32", 32, 500, 100), ("e1m9_64", 64, 200, 50), ("e1m9_96", 96, 100, 25)]
CAP = {"hand": 40, "run7": 16}


@torch.no_grad()
def evaluate(m, X, Y, n, cap, batch, stop_on_halt):
    m.set_grid(n, n)
    exact = np.zeros((cap, len(X)), bool); halt_at = np.full(len(X), -1)
    reach = Y.sum(1) > 0
    for b in range(0, len(X), batch):
        x = torch.from_numpy(X[b:b + batch].astype(np.int64)); yt = torch.from_numpy(Y[b:b + batch])
        h = m.embed(x); y, z = m.init_state(len(x), n * n, "cpu", h.dtype)
        for s in range(cap):
            y, z, lg, q = m.step(h, y, z)
            exact[s, b:b + len(x)] = ((lg[..., 1] > lg[..., 0]).to(torch.uint8) == yt).all(-1).numpy()
            hb = halt_at[b:b + len(x)]; new = (q > 0).numpy() & (hb < 0); hb[new] = s + 1
            if stop_on_halt and ((hb > 0) | ~reach[b:b + len(x)]).all():
                exact[s + 1:, b:b + len(x)] = exact[s, b:b + len(x)]  # the state is a fixed point from here
                break
    return exact, halt_at


def main():
    out = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    models = {"hand": model_from(torch.load(HERE / "weights" / "hand.pt")),
              "run7": model_from(load_bin(str(HERE / "doom" / "bot" / "trm")))}
    res = json.load(open(out)) if Path(out).exists() else {}
    for name, n, count, batch in SETS:
        X = np.load(HERE / "data" / name / "x.npy")[:count]; Y = np.load(HERE / "data" / name / "y.npy")[:count]
        bot_sup = -(-n // 16) + 2  # what bot.js asks for on a whole-level plan of this size
        for mname, m in models.items():
            key = f"{name}/{mname}"
            if key in res: continue
            t = time.time()
            ex, halt = evaluate(m, X, Y, n, CAP[mname], batch, mname == "hand")
            r = {"n": int(len(X)), "reachable": float((Y.sum(1) > 0).mean()),
                 "exact_by_step": {str(s + 1): round(float(ex[s].mean()), 4) for s in range(CAP[mname])},
                 "bot_sup": bot_sup, "exact_at_bot_sup": round(float(ex[bot_sup - 1].mean()), 4), "sec": round(time.time() - t)}
            if mname == "hand":
                hs = halt[halt > 0]; ok = [ex[h - 1, i] for i, h in enumerate(halt) if h > 0]
                r.update(halted=int(len(hs)), exact_at_halt=round(float(np.mean(ok)), 4) if ok else None,
                         halt_step_median=float(np.median(hs)) if len(hs) else None, halt_step_max=int(hs.max()) if len(hs) else None)
            res[key] = r; json.dump(res, open(out, "w"), indent=1)
            print(key, json.dumps(r)[:400], flush=True)


if __name__ == "__main__":
    main()
