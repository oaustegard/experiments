"""Fit a residual-stream code for a compiled ALTA program by gradient descent.

Every compiled weight stays frozen. The only trainable object is a linear code
``U`` (D x d) applied to every write into the residual stream, with a readout
``R`` applied to every read. ``R = U`` is Tracr's shared-`W` convention; the
untied arm frees ``R``.

The objective is teacher-forced on the compiled model's own trajectory, so
nothing is unrolled, and it is a pair of hinges rather than a regression:

  margin      every attention head selects positions by a hard comparison, so
              for each (layer, head, query) the smallest selected logit minus
              the largest unselected logit must stay above half the compiled
              gap
  tolerance   every residual coordinate that some weight reads must be
              reconstructed to within that coordinate's own decision
              tolerance -- 0.25 for an indicator, a quarter of the tightest
              bucket gap for a numerical variable

Each hinge is divided by its own threshold, so a coordinate whose tolerance is
1.6e-3 and an attention gap of 5e3 carry the same weight. That is the design,
following `learned.py` in the LAC run: what the machine needs from each
constraint is binary, while the raw quantities span six orders of magnitude.

Codes are fit by continuation from the identity at ``d = D``, one dimension at
a time. Random initialisation does not find the optimum even where one
provably exists, so a threshold read off a random init would be a fact about
the optimizer.

Run: ``python3 train_code.py [--programs subleq,parity_seq,parity_ff]``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import alta_common as A

HERE = Path(__file__).resolve().parent
MARGIN_FRACTION = 0.5  # target gap, as a fraction of the compiled gap
SURVIVOR_NORM = 1e-6
SCREEN = 20
ORTHOGONAL = 1e-8


# --------------------------------------------------------------------------
# Constraint harvest
# --------------------------------------------------------------------------


class Problem:
    """The frozen machine's decisions, as tensors that a code must preserve."""

    def __init__(self, compiled: A.Compiled, train_inputs, verbose=False):
        self.compiled = compiled
        self.dim = compiled.dim
        slices = A.var_slices(compiled)
        alta = A.import_alta()
        scalar = alta["compiler_config"].Config(**compiled.case.config_kwargs).attention_scalar
        self.gap = scalar**2

        table: dict[bytes, int] = {}
        rows: list[np.ndarray] = []
        groups: list[list[int]] = []
        used = np.zeros(self.dim, dtype=bool)

        def intern(vec: np.ndarray) -> int:
            clean = np.where(np.abs(vec) > A.ZERO, vec, 0.0)
            key = np.round(clean, 9).tobytes()
            idx = table.get(key)
            if idx is None:
                idx = len(rows)
                table[key] = idx
                rows.append(clean)
            return idx

        for input_ids in train_inputs:
            tr = A.trace(compiled, input_ids)
            for layer in range(compiled.case.num_layers):
                pre, mid = tr["pre"][layer], tr["mid"][layer]
                groups.append([intern(v) for v in pre])
                for v in mid:
                    intern(v)
                used |= (np.abs(pre) > A.ZERO).any(axis=0)
                used |= (np.abs(mid) > A.ZERO).any(axis=0)

        states = np.array(rows)
        self.used = used
        self.read = A.read_dims(compiled)
        self.live = self.read & used
        self.tol = A.tolerances(compiled)

        nz = np.nonzero(states)
        self.sparse = torch.sparse_coo_tensor(
            torch.tensor(np.array(nz), dtype=torch.int64),
            torch.tensor(states[nz], dtype=torch.float64),
            size=states.shape,
        ).coalesce()
        self.dense = torch.tensor(states, dtype=torch.float64)
        self.group_idx = torch.tensor(np.array(groups), dtype=torch.int64)
        self.read_idx = torch.tensor(np.nonzero(self.read)[0], dtype=torch.int64)
        self.tol_t = torch.tensor(self.tol[self.read], dtype=torch.float64)

        # Per-head selection pattern of the uncompressed machine.
        self.heads = []
        n_pos = self.group_idx.shape[1]
        for head_spec in compiled.spec.head_specs:
            qs, ks = slices[head_spec.query], slices[head_spec.key]
            q = self.dense[self.group_idx][:, :, qs[0] : qs[1]]
            k = self.dense[self.group_idx][:, :, ks[0] : ks[1]]
            logits = torch.bmm(q, k.transpose(1, 2)) * self.gap
            allowed = _allowed_pairs(head_spec.relative_position_mask, n_pos)
            selected = (logits > 0.5 * self.gap) & allowed
            unselected = (~selected) & allowed
            valid = selected.any(dim=2) & unselected.any(dim=2)
            self.heads.append(
                dict(
                    name=head_spec.output,
                    q=(qs[0], qs[1]),
                    k=(ks[0], ks[1]),
                    selected=selected,
                    unselected=unselected,
                    valid=valid,
                )
            )
        self.n_margin = int(sum(int(h["valid"].sum()) for h in self.heads))
        self.n_tol = int(states.shape[0] * self.read.sum())
        if verbose:
            print(
                f"  {compiled.case.name}: {len(train_inputs)} inputs, "
                f"{states.shape[0]} unique states, D={self.dim}, "
                f"live={int(self.live.sum())}, margins={self.n_margin}",
                flush=True,
            )

    def parts(self, code, readout):
        """Return (margin hinge, tolerance hinge) for a candidate code."""
        transfer = code @ readout.T
        recon = torch.sparse.mm(self.sparse, transfer)
        err = (recon - self.dense)[:, self.read_idx].abs()
        tol = torch.relu(err / self.tol_t - 1.0).clamp(max=100.0).mean()

        target = MARGIN_FRACTION * self.gap
        margins = []
        for head in self.heads:
            q = recon[:, head["q"][0] : head["q"][1]][self.group_idx]
            k = recon[:, head["k"][0] : head["k"][1]][self.group_idx]
            logits = torch.bmm(q, k.transpose(1, 2)) * self.gap
            low = torch.where(head["selected"], logits, torch.inf).amin(dim=2)
            high = torch.where(head["unselected"], logits, -torch.inf).amax(dim=2)
            sep = (low - high)[head["valid"]]
            margins.append(torch.relu(1.0 - sep / target))
        margin = torch.cat(margins).mean() if margins else torch.zeros((), dtype=torch.float64)
        return margin, tol


def _allowed_pairs(relative_position_mask, n_pos: int) -> torch.Tensor:
    allowed = torch.ones(n_pos, n_pos, dtype=torch.bool)
    if relative_position_mask:
        for i in range(n_pos):
            for j in range(n_pos):
                allowed[i, j] = (j - i) in relative_position_mask
    return allowed.unsqueeze(0)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(problem: Problem, d: int, init: np.ndarray, tied=True, iters=400, lr=0.02,
          init_readout=None):
    """Fit the (D, d) code from `init`. Returns (code, readout, diagnostics)."""
    code = torch.tensor(np.asarray(init, dtype=float), requires_grad=True)
    if tied:
        readout, params = code, [code]
    else:
        start = init if init_readout is None else init_readout
        readout = torch.tensor(np.asarray(start, dtype=float), requires_grad=True)
        params = [code, readout]

    def loss_fn():
        margin, tol = problem.parts(code, readout)
        return margin + tol, margin, tol

    total, _, _ = loss_fn()
    if total.item() > 0:
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr / 500)
        for _ in range(iters):
            total, _, _ = loss_fn()
            if total.item() <= 0.0:
                break
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        lbfgs = torch.optim.LBFGS(
            params, max_iter=200, line_search_fn="strong_wolfe",
            tolerance_grad=1e-14, tolerance_change=1e-16,
        )

        def closure():
            lbfgs.zero_grad()
            value, _, _ = loss_fn()
            value.backward()
            return value

        lbfgs.step(closure)

    with torch.no_grad():
        total, margin, tol = loss_fn()
    return (
        code.detach().numpy(),
        readout.detach().numpy(),
        dict(loss=float(total), margin=float(margin), tol=float(tol)),
    )


def project(code: np.ndarray, d: int, states: np.ndarray | None = None) -> np.ndarray:
    """Drop the direction the code is using least, keeping its own geometry.

    `learned.py`'s rule takes the SVD of the code itself; that is the default
    here (`states=None`) and it is what the LAC numbers are comparable to. The
    alternative weights the SVD by the code's image of the trajectory, which
    drops the direction the visited states use least. Both are run: the two
    initialisations reach the same widths but not the same geometry, and
    saying which parts of the answer depend on the initialisation is the point
    of running both.
    """
    if d >= code.shape[1]:
        return code
    basis = code if states is None else states @ code
    _, _, vt = np.linalg.svd(basis, full_matrices=False)
    return code @ vt[:d].T


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def gram_stats(problem: Problem, code: np.ndarray, readout: np.ndarray | None = None) -> dict:
    """Geometry of the surviving live feature directions.

    `max_off_diagonal` is the pre-registered statistic: the Gram matrix of the
    unit-normalised code rows of the live dimensions that survive, where a
    survivor is a row of norm above 1e-6.

    That norm threshold turned out to be too generous to be the whole story --
    a row can shrink continuously here, where LAC's went to exactly zero -- so
    the transfer matrix `U R^T` restricted to live dimensions is reported
    alongside it. Its diagonal is how much of a feature its own readout
    returns, and its off-diagonal is how much of one feature leaks into
    another's readout, which is the interference the machine actually feels.
    """
    readout = code if readout is None else readout
    live = np.nonzero(problem.live)[0]
    rows = code[live]
    norms = np.linalg.norm(rows, axis=1)
    keep = norms > SURVIVOR_NORM
    survivors = rows[keep] / norms[keep][:, None]
    gram = survivors @ survivors.T
    off = np.abs(gram - np.eye(len(survivors)))

    transfer = code[live] @ readout[live].T
    diagonal = np.diag(transfer)
    transfer_off = np.abs(transfer - np.diag(diagonal))
    return dict(
        n_live=int(len(live)),
        n_survivors=int(keep.sum()),
        max_off_diagonal=float(off.max()) if len(survivors) > 1 else 0.0,
        transfer_off_diagonal=float(transfer_off.max()) if len(live) > 1 else 0.0,
        self_transfer_min=float(diagonal.min()),
        n_read_back=int((diagonal > 0.5).sum()),
        mean_survivor_norm=float(norms[keep].mean()) if keep.any() else 0.0,
        max_deleted_norm=float(norms[~keep].max()) if (~keep).any() else 0.0,
    )


def exact_fraction(compiled: A.Compiled, eval_inputs, references, code, readout) -> float:
    """Fraction of held-out inputs the compressed model still computes exactly."""
    outputs = A.forward(compiled, eval_inputs, code, readout)
    hits = sum(
        int([int(x) for x in got] == want) for got, want in zip(outputs, references)
    )
    return hits / len(eval_inputs)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def sweep(name: str, n_eval=100, iters=400, floor=1, verbose=True,
          projection="code") -> dict:
    """Walk the code down one dimension at a time and measure it at each width."""
    case = A.get_case(name)
    compiled = A.compile_case(case)
    train_inputs = A.training_inputs(case)
    eval_inputs = A.sample_inputs(case, n_eval, seed=2, exclude=train_inputs)
    references = [[int(v) for v in A.run_interpreter(compiled, x)] for x in eval_inputs]
    if not all(case.reference(x, r) for x, r in zip(eval_inputs, references)):
        raise RuntimeError(f"{name}: interpreter disagrees with the task definition")

    problem = Problem(compiled, train_inputs, verbose=verbose)
    live = int(problem.live.sum())
    used = int(problem.used.sum())
    states = problem.dense.numpy() if projection == "data" else None

    # Unused dimensions are pure gauge: they are zero on every state the
    # machine visits, so restricting the identity to the used dimensions is an
    # exact, lossless step rather than a compression. The continuation proper
    # starts there and moves one dimension at a time.
    rows, saved = [], {}
    code = np.eye(compiled.dim)
    readout = code
    widths = [compiled.dim] + list(range(used, floor - 1, -1))
    started = time.time()
    zeros = 0
    previous = 1.0

    # A width can cost minutes on SUBLEQ, so the sweep checkpoints after each
    # one and picks up where it stopped.
    checkpoint = HERE / f"partial_{name}_{projection}.npz"
    resume_after = None
    if checkpoint.exists():
        blob = np.load(checkpoint, allow_pickle=True)
        rows = json.loads(str(blob["rows"]))
        code, readout = blob["code"], blob["readout"]
        zeros, previous = int(blob["zeros"]), float(blob["previous"])
        resume_after = int(blob["d"])
        saved = dict(blob["saved"].item())
        print(f"  resumed after d={resume_after}", flush=True)

    skipping = resume_after is not None
    for d in widths:
        if skipping:
            skipping = d != resume_after
            continue
        if d == used:
            code = np.eye(compiled.dim)[:, problem.used]
        code = project(code, d, states)
        code, readout, diag = train(problem, d, code, tied=True, iters=iters)
        # A 20-input screen at every width, and the full 100 whenever the
        # screen passes or the previous width was a working one -- so every
        # working width and the first failing width carry the pre-registered
        # 100-input number, and nothing else pays for it.
        screen = exact_fraction(
            compiled, eval_inputs[:SCREEN], references[:SCREEN], code, readout
        )
        fraction = None
        if screen == 1.0 or previous == 1.0:
            fraction = exact_fraction(compiled, eval_inputs, references, code, readout)
        previous = screen
        row = dict(d=d, fraction=fraction, screen=screen, **diag,
                   **gram_stats(problem, code, readout))
        rows.append(row)
        if d <= live + 12:
            saved[d] = code.copy()
        if verbose:
            shown = f"{screen:.2f}*" if fraction is None else f"{fraction:.2f} "
            print(
                f"  d={d:4d} exact={shown} loss={diag['loss']:.3e} "
                f"read_back={row['n_read_back']:4d} "
                f"gram_off={row['max_off_diagonal']:.2f} "
                f"xfer_off={row['transfer_off_diagonal']:.2f} "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )
        zeros = zeros + 1 if screen == 0.0 else 0
        np.savez(
            checkpoint,
            rows=json.dumps(rows),
            code=code,
            readout=readout,
            zeros=zeros,
            previous=previous,
            d=d,
            saved=np.array(saved, dtype=object),
        )
        if zeros >= 3:
            break

    working = [r["d"] for r in rows if r["fraction"] == 1.0]
    d_min = min(working) if working else None
    record = dict(
        program=name,
        kind=case.kind,
        note=case.note,
        dim=compiled.dim,
        num_layers=case.num_layers,
        n_read=int(problem.read.sum()),
        n_used=int(problem.used.sum()),
        n_live=live,
        projection=projection,
        n_train_inputs=len(train_inputs),
        n_eval_inputs=len(eval_inputs),
        n_unique_states=int(problem.dense.shape[0]),
        n_margin_constraints=problem.n_margin,
        n_tolerance_constraints=problem.n_tol,
        d_min=d_min,
        rows=rows,
    )
    if d_min is not None:
        record["untied"] = untied_arm(
            compiled, problem, eval_inputs, references, saved,
            [d for d in (d_min, d_min - 1) if d in saved], iters=iters,
        )
    return record


def untied_arm(compiled, problem, eval_inputs, references, saved, d_values, iters=400):
    """Refit at chosen widths with the readout freed from the code.

    Tracr ties the readout to the code (a shared `W`), and the LAC run checked
    that the tie, rather than the machine, was what forbade superposition. Same
    check here: each width restarts from the tied solution and lets `R` move,
    which asks whether a non-orthogonal code is reachable from the answer the
    tied arm found.
    """
    out = []
    for d in d_values:
        code, readout, diag = train(
            problem, d, saved[d], tied=False, iters=iters, init_readout=saved[d]
        )
        fraction = exact_fraction(compiled, eval_inputs, references, code, readout)
        out.append(dict(d=d, fraction=fraction, **diag, **gram_stats(problem, code, readout)))
        print(f"  untied d={d} exact={fraction:.2f} "
              f"gram_off={out[-1]['max_off_diagonal']:.2e}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", default=",".join(A.CASE_ORDER))
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--projection", default="code", choices=("code", "data"))
    parser.add_argument("--out", default=str(HERE / "results.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.programs.split(","):
        if f"{name}__{args.projection}" in results:      # --resume: skip finished programs
            print(f"[{name}] done in {out_path.name}, skipping", flush=True)
            continue
        print(f"[{name}]", flush=True)
        record = sweep(name, iters=args.iters, projection=args.projection)
        results[f"{name}__{args.projection}"] = record
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"  d_min={record['d_min']} of {record['n_live']} live", flush=True)


if __name__ == "__main__":
    main()
