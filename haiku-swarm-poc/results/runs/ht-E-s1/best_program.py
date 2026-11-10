import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, linprog

H = np.sqrt(3.0) / 2.0
TOTAL_AREA = H / 2.0
SQ3 = np.sqrt(3.0)
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)), dtype=int)
I_IDX = TRIPLES[:, 0]; J_IDX = TRIPLES[:, 1]; K_IDX = TRIPLES[:, 2]
M = len(TRIPLES)


def heilbronn_triangle11() -> np.ndarray:
    start_time = time.time()
    time_budget = 24.0
    rng = np.random.default_rng(20240607)

    def clip_xy(xy):
        xy = np.asarray(xy, float).copy()
        b2 = np.clip(xy[:, 1] / H, 0.0, 1.0)
        b1 = xy[:, 0] - 0.5 * b2
        b0 = 1.0 - b1 - b2
        b = np.column_stack([b0, b1, b2])
        b = np.clip(b, 0.0, None)
        tot = b.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
        return (b / tot) @ V

    def exact_score_xy(xy):
        xy = np.asarray(xy, float)
        Pi = xy[I_IDX]; Pj = xy[J_IDX]; Pk = xy[K_IDX]
        A = 0.5 * np.abs((Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1])
                         - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0]))
        return float(np.min(A) / TOTAL_AREA)

    def areas_grad_xy(xy):
        Pi = xy[I_IDX]; Pj = xy[J_IDX]; Pk = xy[K_IDX]
        d1 = Pj - Pi; d2 = Pk - Pi
        signed = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        sign = np.sign(signed); sign[sign == 0.0] = 1.0
        A = np.abs(signed)
        v0 = Pj - Pk; v1 = Pk - Pi; v2 = Pi - Pj
        G = np.empty((M, 3, 2))
        G[:, 0, :] = 0.5 * np.column_stack([v0[:, 1], -v0[:, 0]]) * sign[:, None]
        G[:, 1, :] = 0.5 * np.column_stack([v1[:, 1], -v1[:, 0]]) * sign[:, None]
        G[:, 2, :] = 0.5 * np.column_stack([v2[:, 1], -v2[:, 0]]) * sign[:, None]
        Gfull = np.zeros((M, N, 2))
        rows = np.arange(M)
        for c in range(3):
            Gfull[rows, TRIPLES[:, c], :] = G[:, c, :]
        return A, Gfull

    def lp_polish(xy0, max_cycles=12, deadline=None):
        pts = clip_xy(xy0)
        for _ in range(max_cycles):
            if deadline is not None and time.time() > deadline:
                break
            cur = exact_score_xy(pts); improved = False
            Pi = pts[I_IDX]; Pj = pts[J_IDX]; Pk = pts[K_IDX]
            areas = 0.5 * np.abs((Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1])
                                 - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0]))
            visit = []
            for t in np.argsort(areas)[:24]:
                for p in TRIPLES[t]:
                    if int(p) not in visit:
                        visit.append(int(p))
            for i in visit + [q for q in range(N) if q not in visit]:
                if deadline is not None and time.time() > deadline:
                    break
                others = [q for q in range(N) if q != i]
                A_ub = []; b_ub = []
                for ai in range(len(others)):
                    for bi in range(ai + 1, len(others)):
                        j = others[ai]; k = others[bi]
                        Pj_, Pk_ = pts[j], pts[k]
                        Ac = Pj_[1] - Pk_[1]; Bc = Pk_[0] - Pj_[0]
                        Cc = Pj_[0] * Pk_[1] - Pk_[0] * Pj_[1]
                        s = 0.5 * (Ac * pts[i, 0] + Bc * pts[i, 1] + Cc)
                        sgn = 1.0 if s >= 0 else -1.0
                        A_ub.append([-sgn * Ac / (2 * TOTAL_AREA), -sgn * Bc / (2 * TOTAL_AREA), 1.0])
                        b_ub.append(sgn * Cc / (2 * TOTAL_AREA))
                A_ub += [[0.0, -1.0, 0.0], [-SQ3, 1.0, 0.0], [SQ3, 1.0, 0.0]]
                b_ub += [0.0, 0.0, SQ3]
                try:
                    res = linprog([0.0, 0.0, -1.0], A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                                  bounds=[(0.0, 1.0), (0.0, H), (0.0, 0.25)], method="highs")
                except Exception:
                    res = None
                if res is not None and np.all(np.isfinite(res.x)) and float(res.x[2]) > cur + 1e-10:
                    trial = pts.copy(); trial[i] = res.x[:2]; trial = clip_xy(trial)
                    if exact_score_xy(trial) > cur + 1e-10:
                        pts = trial; cur = exact_score_xy(pts); improved = True
            if not improved:
                break
        return pts

    # ---- symmetric parametrization: 5 pairs + 1 axis ----
    K, MM = 5, 1
    R = 2 * K + MM

    def expand_sym(z):
        z = np.asarray(z, float); pts = np.zeros((2 * K + MM, 2))
        for i in range(K):
            x, y = z[2 * i], z[2 * i + 1]
            pts[2 * i] = (x, y); pts[2 * i + 1] = (1.0 - x, y)
        pts[2 * K] = (0.5, z[2 * K])
        return pts

    def sym_areas_grad(z):
        pts = expand_sym(z)
        A, G = areas_grad_xy(pts)
        gr = np.zeros((M, R))
        for i in range(K):
            gl = G[:, 2 * i, :]; grr = G[:, 2 * i + 1, :]
            gr[:, 2 * i] = gl[:, 0] - grr[:, 0]
            gr[:, 2 * i + 1] = gl[:, 1] + grr[:, 1]
        gr[:, 2 * K] = G[:, 2 * K, 1]
        return A, gr

    def clip_sym(z):
        z = np.array(z, float)
        for i in range(K):
            z[2 * i] = np.clip(z[2 * i], 0.0, 0.5); z[2 * i + 1] = np.clip(z[2 * i + 1], 0.0, H)
        z[2 * K] = np.clip(z[2 * K], 0.0, H)
        return z

    def run_epigraph_sym(z0, maxiter=350):
        z0 = clip_sym(z0)
        t0 = exact_score_xy(expand_sym(z0)) * TOTAL_AREA
        zz0 = np.concatenate([z0, [max(t0, 0.0)]])
        # linear interior constraints as ineq functions stacked manually via SLSQP dict
        def cons_lin(zz):
            z = zz[:R]; out = []
            for i in range(K):
                x = z[2 * i]; y = z[2 * i + 1]
                out += [y, SQ3 * x - y, -SQ3 * x - y + SQ3]
            out += [z[2 * K], -z[2 * K] + H]
            return np.array(out)
        def cons_lin_jac(zz):
            Jm = np.zeros((3 * K + 2, R + 1)); r = 0
            for i in range(K):
                Jm[r, 2 * i + 1] = 1.0; r += 1
                Jm[r, 2 * i] = SQ3; Jm[r, 2 * i + 1] = -1.0; r += 1
                Jm[r, 2 * i] = -SQ3; Jm[r, 2 * i + 1] = -1.0; r += 1
            Jm[r, 2 * K] = 1.0; r += 1
            Jm[r, 2 * K] = -1.0
            return Jm
        def cf(zz):
            A, _ = sym_areas_grad(zz[:R]); return A - zz[-1]
        def cj(zz):
            _, gr = sym_areas_grad(zz[:R])
            Jm = np.zeros((M, R + 1)); Jm[:, :R] = gr; Jm[:, -1] = -1.0; return Jm
        def obj(zz):
            g = np.zeros_like(zz); g[-1] = -1.0; return -float(zz[-1]), g
        bounds = [(0.0, 0.5), (0.0, H)] * K + [(0.0, H)] + [(0.0, 0.25)]
        try:
            res = minimize(obj, zz0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cf, "jac": cj},
                                        {"type": "ineq", "fun": cons_lin, "jac": cons_lin_jac}],
                           options={"maxiter": int(maxiter), "ftol": 1e-11})
            if res is not None and np.all(np.isfinite(res.x)):
                cand = clip_sym(res.x[:R])
                if exact_score_xy(expand_sym(cand)) >= exact_score_xy(expand_sym(z0)) - 1e-12:
                    return cand
        except Exception:
            pass
        return z0

    # ---- asymmetric epigraph polish (allows symmetry breaking) ----
    def run_epigraph_asym(xy0, maxiter=300):
        xy0 = clip_xy(xy0)
        x0 = xy0.reshape(-1)
        t0 = exact_score_xy(xy0) * TOTAL_AREA
        z0 = np.concatenate([x0, [max(t0, 0.0)]])
        D = 2 * N
        def cf(zz):
            A, _ = areas_grad_xy(zz[:D].reshape(N, 2)); return A - zz[-1]
        def cj(zz):
            _, Gfull = areas_grad_xy(zz[:D].reshape(N, 2))
            Jm = np.zeros((M, D + 1)); Jm[:, :D] = Gfull.reshape(M, D); Jm[:, -1] = -1.0; return Jm
        def clin(zz):
            p = zz[:D].reshape(N, 2)
            return np.concatenate([p[:, 1], SQ3 * p[:, 0] - p[:, 1], -SQ3 * p[:, 0] - p[:, 1] + SQ3])
        def clin_jac(zz):
            Jm = np.zeros((3 * N, D + 1))
            for i in range(N):
                Jm[i, 2 * i + 1] = 1.0
                Jm[N + i, 2 * i] = SQ3; Jm[N + i, 2 * i + 1] = -1.0
                Jm[2 * N + i, 2 * i] = -SQ3; Jm[2 * N + i, 2 * i + 1] = -1.0
            return Jm
        def obj(zz):
            g = np.zeros_like(zz); g[-1] = -1.0; return -float(zz[-1]), g
        bounds = [(0.0, 1.0), (0.0, H)] * N + [(0.0, 0.25)]
        try:
            res = minimize(obj, z0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cf, "jac": cj},
                                        {"type": "ineq", "fun": clin, "jac": clin_jac}],
                           options={"maxiter": int(maxiter), "ftol": 1e-11})
            if res is not None and np.all(np.isfinite(res.x)):
                cand = clip_xy(res.x[:D].reshape(N, 2))
                if exact_score_xy(cand) >= exact_score_xy(xy0) - 1e-12:
                    return cand
        except Exception:
            pass
        return xy0

    def lattice(D):
        pts = []
        for i in range(D + 1):
            for j in range(D + 1 - i):
                k = D - i - j
                pts.append((i * V[0] + j * V[1] + k * V[2]) / D)
        return np.array(pts)

    # fallback best (lattice)
    lat3 = lattice(3); lat12 = lattice(12)
    best_xy = None; best_s = -1.0
    for cand_pt in lat12:
        if np.min(np.linalg.norm(lat3 - cand_pt, axis=1)) < 1e-9:
            continue
        trial = np.vstack([lat3, cand_pt[None, :]])
        s = exact_score_xy(trial)
        if s > best_s:
            best_s = s; best_xy = trial.copy()
    if best_xy is None:
        best_xy = np.vstack([lat3, [0.5, H / 3.0]]); best_s = exact_score_xy(best_xy)
    try:
        lp = lp_polish(best_xy, max_cycles=8, deadline=start_time + 3.0)
        if exact_score_xy(lp) > best_s:
            best_xy, best_s = lp, exact_score_xy(lp)
    except Exception:
        pass

    # ---- symmetric seeds ----
    base_pairs = [(0.0, 0.0), (1.0 / 3.0, 0.0), (1.0 / 6.0, H / 3.0), (1.0 / 3.0, 2 * H / 3.0)]
    def make_z(pairs, axis_y):
        z = []
        for (x, y) in pairs: z += [x, y]
        z.append(axis_y); return np.array(z, float)

    seeds = []
    seeds.append(make_z(base_pairs + [(0.42, H / 3.0)], H))
    seeds.append(make_z(base_pairs + [(0.30, H / 2.0)], H / 3.0))
    seeds.append(make_z(base_pairs + [(0.38, 2 * H / 3.0)], H))
    seeds.append(make_z(base_pairs + [(0.25, 0.5 * H)], 2 * H / 3.0))
    seeds.append(make_z(base_pairs + [(0.45, 0.15)], H / 2.0))
    # symmetrized lattice seeds: take left-half lattice points
    for D in (4, 5, 6, 8):
        lat = lattice(D)
        left = [p for p in lat if p[0] < 0.5 - 1e-9]
        # farthest-point selection of 5 left points
        if len(left) >= 5:
            left = np.array(left); chosen = [0]
            while len(chosen) < 5:
                d = np.min([np.sum((left - left[c]) ** 2, axis=1) for c in chosen], axis=0)
                d[chosen] = -1.0; chosen.append(int(np.argmax(d)))
            sel = left[chosen]
            seeds.append(make_z([(float(p[0]), float(p[1])) for p in sel], H * rng.uniform(0.3, 1.0)))
    # random symmetric seeds (Latin-hypercube style via stratified y)
    for _ in range(24):
        ys = np.sort(rng.uniform(0.0, H, size=K))
        xs = rng.uniform(0.0, 0.5, size=K)
        # keep inside triangle: x <= 0.5 - y/(2*?) left edge x >= y/SQ3
        xs = np.maximum(xs, ys / SQ3 + 0.01)
        xs = np.minimum(xs, 0.49)
        pairs = [(float(xs[i]), float(ys[i])) for i in range(K)]
        seeds.append(make_z(pairs, float(rng.uniform(0.0, H))))

    rng.shuffle(seeds) if hasattr(rng, "shuffle") else None
    # run epigraph on seeds in order, handcrafted first after shuffle of rest
    seeds = seeds[:5] + seeds[5:]
    best_z = None
    cand_z = []
    for idx, z0 in enumerate(seeds):
        if time.time() - start_time > 13.0:
            break
        try:
            zc = run_epigraph_sym(z0, maxiter=300)
            s = exact_score_xy(expand_sym(zc))
            cand_z.append((s, zc))
            if s > best_s:
                best_s, best_z = s, zc.copy(); best_xy = clip_xy(expand_sym(zc))
        except Exception:
            continue
    if cand_z:
        cand_z.sort(key=lambda t: -t[0])
        if best_z is None:
            best_z = cand_z[0][1]
    else:
        best_z = seeds[0]

    # ---- symmetric basin hopping / simulated-annealing style perturbations ----
    sigmas = [0.015, 0.03, 0.05, 0.08, 0.12, 0.02, 0.06]
    it = 0
    current_z = best_z.copy(); current_s = best_s
    while time.time() - start_time < time_budget - 5.0:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        # temperature decreasing; accept worse with prob exp((s-cur)/T)
        T = 0.0015 * max(0.15, 1.0 - (time.time() - start_time) / time_budget)
        try:
            zp = clip_sym(current_z + rng.normal(scale=sigma, size=R))
            zc = run_epigraph_sym(zp, maxiter=250)
            s = exact_score_xy(expand_sym(zc))
            if s > current_s or rng.uniform() < np.exp(max(-50.0, (s - current_s) / max(T, 1e-9))):
                current_z, current_s = zc, s
            if s > best_s:
                best_s, best_z = s, zc.copy(); best_xy = clip_xy(expand_sym(zc))
                current_z = zc.copy()
        except Exception:
            continue

    # ---- final asymmetric polish (idea 3): allow symmetry to break ----
    try:
        if time.time() - start_time < time_budget - 2.0:
            xy = lp_polish(best_xy, max_cycles=10, deadline=start_time + time_budget - 1.5)
            if exact_score_xy(xy) > best_s:
                best_xy, best_s = xy, exact_score_xy(xy)
            if time.time() - start_time < time_budget - 1.2:
                xy2 = run_epigraph_asym(best_xy, maxiter=250)
                if exact_score_xy(xy2) > best_s:
                    best_xy, best_s = xy2, exact_score_xy(xy2)
                xy3 = lp_polish(best_xy, max_cycles=8, deadline=start_time + time_budget + 1.0)
                if exact_score_xy(xy3) > best_s:
                    best_xy, best_s = xy3, exact_score_xy(xy3)
    except Exception:
        pass

    points = clip_xy(np.asarray(best_xy, float).reshape(N, 2))
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        points = clip_xy(np.vstack([lat3, [0.5, H / 3.0]]))
    return points
