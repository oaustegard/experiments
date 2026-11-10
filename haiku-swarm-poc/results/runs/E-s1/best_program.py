import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 26
_PI = []
_PJ = []
for _i in range(N):
    for _j in range(_i + 1, N):
        _PI.append(_i); _PJ.append(_j)
PI = np.array(_PI); PJ = np.array(_PJ); M = len(PI)


def _solve_lp_radii(centers):
    n = centers.shape[0]
    b_border = np.empty(n)
    for i in range(n):
        x, y = centers[i]
        b_border[i] = min(x, y, 1.0 - x, 1.0 - y)
    dvals = np.hypot(centers[PI, 0] - centers[PJ, 0],
                     centers[PI, 1] - centers[PJ, 1])
    m = len(dvals)
    A = np.zeros((m + n, n))
    A[np.arange(m), PI] = 1.0
    A[np.arange(m), PJ] = 1.0
    A[m + np.arange(n), np.arange(n)] = 1.0
    b = np.concatenate([dvals, b_border])
    res = linprog(c=-np.ones(n), A_ub=A, b_ub=b,
                  bounds=[(0.0, None)] * n, method="highs")
    if res.x is not None:
        r = np.maximum(res.x, 0.0)
    else:
        r = np.minimum(b_border, 0.05)
    return r


def _joint_optimize(centers0, radii0, maxiter=350):
    n = centers0.shape[0]
    nvar = 3 * n

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros(nvar); g[2 * n:] = -1.0; return g

    def cons(z):
        c, r = unpack(z)
        dx = c[PI, 0] - c[PJ, 0]; dy = c[PI, 1] - c[PJ, 1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-14
        pair = d - r[PI] - r[PJ]
        border = np.empty(4 * n)
        border[0::4] = c[:, 0] - r
        border[1::4] = 1.0 - c[:, 0] - r
        border[2::4] = c[:, 1] - r
        border[3::4] = 1.0 - c[:, 1] - r
        return np.concatenate([pair, border])

    def cons_jac(z):
        c, r = unpack(z)
        dx = c[PI, 0] - c[PJ, 0]; dy = c[PI, 1] - c[PJ, 1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-14
        ux = dx / d; uy = dy / d
        J = np.zeros((M + 4 * n, nvar))
        rows = np.arange(M)
        J[rows, 2 * PI] += ux; J[rows, 2 * PI + 1] += uy
        J[rows, 2 * PJ] -= ux; J[rows, 2 * PJ + 1] -= uy
        J[rows, 2 * n + PI] -= 1.0; J[rows, 2 * n + PJ] -= 1.0
        idx = np.arange(n)
        J[M + 4 * idx, 2 * idx] = 1.0; J[M + 4 * idx, 2 * n + idx] = -1.0
        J[M + 4 * idx + 1, 2 * idx] = -1.0; J[M + 4 * idx + 1, 2 * n + idx] = -1.0
        J[M + 4 * idx + 2, 2 * idx + 1] = 1.0; J[M + 4 * idx + 2, 2 * n + idx] = -1.0
        J[M + 4 * idx + 3, 2 * idx + 1] = -1.0; J[M + 4 * idx + 3, 2 * n + idx] = -1.0
        return J

    z0 = np.concatenate([np.clip(centers0, 0, 1).ravel(), radii0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    res = minimize(obj, z0, method="SLSQP", jac=obj_jac, bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": maxiter, "ftol": 1e-10})
    c, r = unpack(res.x)
    return np.clip(c, 0.0, 1.0), np.maximum(r, 0.0)


def _penalty_optimize(centers0, radii0):
    """Fast smooth penalty exploration; final feasibility via LP radii."""
    n = N
    z = np.concatenate([np.clip(centers0, 0, 1).ravel(), np.maximum(radii0, 0)])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    def make_fg(P):
        def fg(zv):
            c = zv[:2 * n].reshape(n, 2); r = zv[2 * n:]
            dx = c[PI, 0] - c[PJ, 0]; dy = c[PI, 1] - c[PJ, 1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            v = r[PI] + r[PJ] - d
            mask = v > 0
            f = -np.sum(r)
            grad_c = np.zeros((n, 2)); grad_r = -np.ones(n)
            if np.any(mask):
                vm = v[mask]
                f += P * np.sum(vm * vm)
                g = 2.0 * P * vm
                ux = dx[mask] / d[mask]; uy = dy[mask] / d[mask]
                np.add.at(grad_c, PI[mask], np.stack([-g * ux, -g * uy], axis=1))
                np.add.at(grad_c, PJ[mask], np.stack([g * ux, g * uy], axis=1))
                np.add.at(grad_r, PI[mask], g)
                np.add.at(grad_r, PJ[mask], g)
            # borders: left, right, bottom, top violations
            bv = np.empty((n, 4))
            bv[:, 0] = r - c[:, 0]
            bv[:, 1] = r - (1.0 - c[:, 0])
            bv[:, 2] = r - c[:, 1]
            bv[:, 3] = r - (1.0 - c[:, 1])
            bmask = bv > 0
            if np.any(bmask):
                vm = bv[bmask]
                f += P * np.sum(vm * vm)
                g = 2.0 * P * vm
                idx = np.where(bmask)
                # coordinate derivatives: left -1, right +1, bottom -1, top +1 for x/y
                sgn = np.array([-1.0, 1.0, -1.0, 1.0])
                coord = np.where(idx[1] < 2, 0, 1)
                np.add.at(grad_c, (idx[0], coord), g * sgn[idx[1]])
                np.add.at(grad_r, idx[0], g)
            return f, np.concatenate([grad_c.ravel(), grad_r])
        return fg

    for P, it in ((25.0, 180), (250.0, 180), (2500.0, 200), (25000.0, 220)):
        try:
            res = minimize(make_fg(P), z, method="L-BFGS-B", jac=True,
                           bounds=bounds, options={"maxiter": it, "ftol": 1e-12})
            if res.x is not None:
                z = res.x
        except Exception:
            break
    c = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0)
    return c


def _rows_centers(counts):
    centers = []; nr = len(counts)
    for ri, cnt in enumerate(counts):
        y = (ri + 0.5) / nr
        for j in range(cnt):
            x = (j + 0.5) / cnt
            if ri % 2 == 1:
                x = np.clip(x + 0.5 / cnt * 0.3, 0.02, 0.98)
            centers.append([x, y])
    return np.array(centers, dtype=float)


def _hex_centers(counts):
    centers = []; nr = len(counts)
    for ri, cnt in enumerate(counts):
        y = (ri + 0.5) / nr
        off = 0.5 / cnt if ri % 2 == 1 else -0.5 / cnt
        for j in range(cnt):
            x = np.clip((j + 0.5) / cnt + off * 0.5, 0.03, 0.97)
            centers.append([x, y])
    # trim / pad to 26 (counts given sum to 26)
    return np.array(centers[:26], dtype=float)


def _shell_centers():
    pts = [[0.11, 0.11], [0.89, 0.11], [0.11, 0.89], [0.89, 0.89],
           [0.5, 0.07], [0.5, 0.93], [0.07, 0.5], [0.93, 0.5]]
    for y in (0.28, 0.5, 0.72):
        for x in (0.18, 0.34, 0.5, 0.66, 0.82, 0.26):
            if len(pts) < 26:
                pts.append([x, y])
    # keep order shuffled-ish deterministic to 26
    return np.array(pts[:26], dtype=float)


def run_packing():
    start_time = time.time()
    time_budget = 26.0
    rng = np.random.default_rng(1)

    starts = []
    starts.append(_rows_centers([5, 5, 6, 5, 5]))
    starts.append(_rows_centers([4, 6, 6, 6, 4]))
    starts.append(_hex_centers([4, 6, 6, 6, 4]))
    starts.append(_hex_centers([5, 5, 6, 5, 5]))
    starts.append(_shell_centers())
    starts.append(_rows_centers([6, 5, 5, 5, 5]))
    starts.append(_rows_centers([5, 6, 5, 5, 5]))

    best_centers = None; best_radii = None; best_sum = -1.0
    elites = []  # list of (sum, centers)

    def evaluate(c):
        nonlocal best_centers, best_radii, best_sum
        c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        if c.shape != (26, 2) or not np.all(np.isfinite(c)):
            return -1.0
        r = _solve_lp_radii(c)
        s = float(np.sum(r))
        elites.append((s, c.copy()))
        elites.sort(key=lambda t: -t[0])
        del elites[5:]
        if s > best_sum:
            best_sum = s; best_centers = c.copy(); best_radii = r.copy()
        return s

    for c0 in starts:
        evaluate(c0)

    # Fast penalty + SLSQP on structured starts
    for c0 in starts:
        if time.time() - start_time > time_budget * 0.45:
            break
        try:
            r0 = _solve_lp_radii(c0)
            cp = _penalty_optimize(c0, r0)
            sp = evaluate(cp)
            # SLSQP polish if promising
            if sp > best_sum - 0.08:
                r1 = _solve_lp_radii(cp)
                co, _ = _joint_optimize(cp, r1, maxiter=300)
                evaluate(co)
        except Exception:
            pass

    # Iterated perturbation-polish until time expires
    sigmas = [0.012, 0.02, 0.03, 0.045, 0.07, 0.10, 0.16]
    attempt = 0
    while time.time() - start_time < time_budget:
        attempt += 1
        u = rng.random()
        if u < 0.08:
            c0 = rng.uniform(0.06, 0.94, size=(26, 2))
        elif u < 0.30 and len(elites) > 1:
            c0 = elites[rng.integers(0, len(elites))][1].copy()
            sigma = sigmas[rng.integers(0, len(sigmas))]
            c0 = np.clip(c0 + rng.normal(0, sigma, size=c0.shape), 0.02, 0.98)
        else:
            if best_centers is None:
                continue
            c0 = best_centers.copy()
            sigma = sigmas[(attempt - 1) % len(sigmas)]
            c0 = np.clip(c0 + rng.normal(0, sigma, size=c0.shape), 0.02, 0.98)
            if attempt % 17 == 0:  # large basin-hopping kick
                c0 = np.clip(c0 + rng.uniform(-0.14, 0.14, size=c0.shape), 0.02, 0.98)
        try:
            r0 = _solve_lp_radii(c0)
            cp = _penalty_optimize(c0, r0)
            sp = evaluate(cp)
            # Exact polish for near-elite candidates; occasionally polish all
            if sp > best_sum - 0.025 or attempt % 4 == 0:
                r1 = _solve_lp_radii(cp)
                co, _ = _joint_optimize(cp, r1, maxiter=260)
                evaluate(co)
        except Exception:
            pass
        if best_sum > 2.70:
            break

    # Final SLSQP polish of the best
    if best_centers is not None:
        try:
            co, _ = _joint_optimize(best_centers, best_radii, maxiter=700)
            evaluate(co)
        except Exception:
            pass
        # a couple of tiny perturbations of final best if time remains
        while time.time() - start_time < time_budget + 2.0:
            try:
                c0 = np.clip(best_centers + rng.normal(0, 0.012, size=best_centers.shape), 0.02, 0.98)
                r0 = _solve_lp_radii(c0)
                cp = _penalty_optimize(c0, r0)
                evaluate(cp)
                r1 = _solve_lp_radii(cp)
                co, _ = _joint_optimize(cp, r1, maxiter=300)
                evaluate(co)
            except Exception:
                break

    centers = np.clip(best_centers, 0.0, 1.0)
    radii = _solve_lp_radii(centers) * (1.0 - 1e-7)
    for i in range(len(radii)):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1.0 - x, 1.0 - y)
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
