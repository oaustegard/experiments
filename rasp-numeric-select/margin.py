"""Two measurements on the parabolic head: the exact gap, and the beta it costs.

The gap. Position j writes the key `(2j, -j^2)` and a query for address x is
`(x, 1)`, so `score(j) = 2xj - j^2 = x^2 - (j - x)^2`. The winner scores `x^2`
and the nearest runner-up scores `x^2 - 1`, so the winner/runner-up gap is
exactly 1 for every integer address. `gap_table` checks that numerically for
every address in range at several lengths rather than trusting the algebra.

The beta. Average-hard attention takes the argmax outright. Softmax attention
leaks `exp(-beta (j - x)^2)` onto every other position, so the output error
scales with beta and with the magnitude of the values being read.
`beta_threshold` reports the smallest beta on a fixed grid at which softmax
output agrees with hard output to within 0.5 everywhere.
"""

from __future__ import annotations

import numpy as np

import compile_ns as C
import programs as P

BETAS = np.round(np.arange(0.25, 24.01, 0.25), 4)
TOL = 0.5


def gap_table(lengths=(8, 32, 128, 512)) -> list[dict]:
    """Winner/runner-up score gap for every integer address in range."""
    rows = []
    for n in lengths:
        j = np.arange(n, dtype=float)
        keys = np.stack([2.0 * j, -(j * j)], axis=1)
        gaps = []
        for x in range(n):
            scores = keys @ np.array([float(x), 1.0])
            order = np.sort(scores)
            gaps.append(order[-1] - order[-2])
            assert int(np.argmax(scores)) == x, f"argmax missed address {x} at n={n}"
        rows.append(
            {
                "n": n,
                "min_gap": float(min(gaps)),
                "max_gap": float(max(gaps)),
                "exactly_one": bool(all(g == 1.0 for g in gaps)),
            }
        )
    return rows


def _errors(
    key: str, n: int, trials: int = 6, seed: int = 0, gate_bound: float | None = None
) -> np.ndarray:
    """Max |softmax - hard| over the beta grid, worst case over trials."""
    prog = P.PROGRAMS[key]
    sop = prog.build()
    model = C.compile_program(
        sop, vocab=prog.vocab, gate_bound=gate_bound or C.GATE_BOUND
    )
    rng = np.random.default_rng(seed)
    worst = np.zeros(len(BETAS))
    for _ in range(trials):
        seq = prog.sample(rng, n)
        hard = model.run(seq, mode="hard")
        for bi, beta in enumerate(BETAS):
            soft = model.run(seq, mode="soft", beta=float(beta))
            worst[bi] = max(worst[bi], float(np.abs(soft - hard).max()))
    return worst


def beta_threshold(
    key: str, n: int, trials: int = 6, seed: int = 0, gate_bound: float | None = None
) -> float | None:
    """Smallest beta on the grid where softmax matches hard argmax within TOL."""
    errs = _errors(key, n, trials, seed, gate_bound)
    ok = np.nonzero(errs <= TOL)[0]
    if ok.size == 0:
        return None
    # the error is not guaranteed monotone, so take the first beta from which it
    # stays under tolerance
    start = ok[0]
    for i in ok:
        if np.all(errs[i:] <= TOL):
            start = i
            break
    return float(BETAS[start])


def sweep(keys=None, lengths=(8, 32, 128), trials: int = 6, seed: int = 0) -> dict:
    keys = keys or P.NUMERIC_KEYS
    return {
        key: {n: beta_threshold(key, n, trials, seed) for n in lengths} for key in keys
    }


def gate_sweep(key: str = "shift", n: int = 32, bounds=(1e2, 1e3, 1e4, 1e6)) -> dict:
    """The range gate multiplies an error in its 0/1 flag by its bound M, so M
    sets how sharp softmax has to be. Reported for one program at one length."""
    return {float(m): beta_threshold(key, n, gate_bound=float(m)) for m in bounds}


def figure(path: str = "gap_vs_beta.png", lengths=(8, 32, 128)) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = P.NUMERIC_KEYS
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(lengths)))
    for ax, key in zip(axes, keys):
        for c, n in zip(colors, lengths):
            errs = _errors(key, n)
            ax.plot(BETAS, np.maximum(errs, 1e-12), color=c, label=f"n = {n}")
            thr = beta_threshold(key, n)
            if thr is not None:
                ax.axvline(thr, color=c, ls=":", lw=1)
        ax.axhline(TOL, color="0.35", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("key scale beta")
        ax.set_title(f"{key}: {P.PROGRAMS[key].title}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("max |softmax - hard argmax|")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Softmax attention against average-hard attention on a gap-1 parabolic head",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("winner / runner-up gap")
    for row in gap_table():
        print(
            f"  n={row['n']:5d}  min={row['min_gap']:.6f}  max={row['max_gap']:.6f}  "
            f"exactly 1: {row['exactly_one']}"
        )
    print()
    lengths = (8, 32, 128)
    table = sweep(lengths=lengths)
    print("smallest beta with |softmax - hard| <= 0.5")
    print("  " + "program".ljust(10) + "".join(f"n={n:<8d}" for n in lengths))
    for key, row in table.items():
        cells = "".join(f"{row[n]!s:<10}" for n in lengths)
        print("  " + key.ljust(10) + cells)
    print()
    print("beta threshold against the range-gate bound M (shift, n=32)")
    for m, thr in gate_sweep().items():
        print(f"  M={m:>10.0f}  beta={thr}")
    print()
    print("figure:", figure())
