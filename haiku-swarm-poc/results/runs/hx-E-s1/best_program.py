# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def hexagon_packing_11():
    t0 = time.time()
    Deadline = t0 + 24.0
    sqrt3 = float(np.sqrt(3.0))
    cos30 = float(np.cos(np.pi / 6.0))
    apo = sqrt3 / 2.0
    rng = np.random.default_rng(98765)

    verts0 = np.array([[np.cos(k * np.pi / 3.0), np.sin(k * np.pi / 3.0)]
                       for k in range(6)], dtype=float)
    base_n = np.array([[np.cos(np.pi / 6.0 + k * np.pi / 3.0),
                        np.sin(np.pi / 6.0 + k * np.pi / 3.0)] for k in range(3)])

    neighbours = np.array([sqrt3 * np.array([np.cos(np.deg2rad(30.0 + 60.0 * k)),
                                             np.sin(np.deg2rad(30.0 + 60.0 * k))]) for k in range(6)])
    corners = np.array([3.0 * np.array([np.cos(np.deg2rad(60.0 * k)),
                                        np.sin(np.deg2rad(60.0 * k))]) for k in range(6)])
    base_all = np.vstack([np.zeros((1, 2)), neighbours])
    fallback_centers = np.vstack([base_all, corners[:4]])

    pairs = list(itertools.combinations(range(11), 2))
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])

    def rotated_verts(centers, angles_rad):
        ca = np.cos(angles_rad)[:, None]; sa = np.sin(angles_rad)[:, None]
        vx = ca * verts0[None, :, 0] - sa * verts0[None, :, 1]
        vy = sa * verts0[None, :, 0] + ca * verts0[None, :, 1]
        return centers[:, None, :] + np.stack([vx, vy], axis=-1)

    def outer_normals(phi_rad):
        return np.array([[np.cos(phi_rad + np.pi / 6.0 + k * np.pi / 3.0),
                          np.sin(phi_rad + np.pi / 6.0 + k * np.pi / 3.0)] for k in range(3)], dtype=float)

    def required_R(centers, angles_rad, outer_center, phi_rad):
        pts = rotated_verts(centers, angles_rad)
        n = outer_normals(phi_rad)
        vals = np.einsum("ijk,lk->ijl", pts - outer_center, n)
        return float(np.max(np.abs(vals)) / cos30)

    def pair_slacks(centers, angles_rad):
        ca = np.cos(angles_rad)[:, None]; sa = np.sin(angles_rad)[:, None]
        vx = ca * verts0[None, :, 0] - sa * verts0[None, :, 1]
        vy = sa * verts0[None, :, 0] + ca * verts0[None, :, 1]
        v = np.stack([vx, vy], axis=-1)
        nrm = np.empty((11, 3, 2))
        nrm[:, :, 0] = ca * base_n[None, :, 0] - sa * base_n[None, :, 1]
        nrm[:, :, 1] = sa * base_n[None, :, 0] + ca * base_n[None, :, 1]
        slacks = []
        for i, j in itertools.combinations(range(11), 2):
            d = centers[i] - centers[j]
            best = -1e9
            for src in (i, j):
                for k in range(3):
                    n = nrm[src, k]
                    proj = float(np.dot(d, n))
                    hi = float(np.max(v[i] @ n)); hj = float(np.max(v[j] @ n))
                    best = max(best, abs(proj) - hi - hj)
            slacks.append(best)
        return np.array(slacks)

    def validate(centers, angles, oc, phi, R):
        if R < required_R(centers, angles, oc, phi) - 1e-9:
            return False
        return np.min(pair_slacks(centers, angles)) >= -1e-7

    def axis_assignment(centers0, angles0):
        ca = np.cos(angles0); sa = np.sin(angles0)
        vx = ca[:, None] * verts0[None, :, 0] - sa[:, None] * verts0[None, :, 1]
        vy = sa[:, None] * verts0[None, :, 0] + ca[:, None] * verts0[None, :, 1]
        v = np.stack([vx, vy], axis=-1)
        axis_src = np.zeros(55, dtype=int); axis_idx = np.zeros(55, dtype=int); axis_sign = np.ones(55)
        for t, (i, j) in enumerate(pairs):
            d = centers0[i] - centers0[j]
            best_slack, best_info = -1e9, (0, 0, 1.0)
            for src in (i, j):
                for k in range(3):
                    n = np.array([np.cos(angles0[src]) * base_n[k, 0] - np.sin(angles0[src]) * base_n[k, 1],
                                  np.sin(angles0[src]) * base_n[k, 0] + np.cos(angles0[src]) * base_n[k, 1]])
                    proj = float(np.dot(d, n))
                    slack = abs(proj) - float(np.max(v[i] @ n)) - float(np.max(v[j] @ n))
                    if slack > best_slack:
                        best_slack = slack
                        best_info = (0 if src == i else 1, k, 1.0 if proj >= 0 else -1.0)
            axis_src[t], axis_idx[t], axis_sign[t] = best_info
        src_idx = np.where(axis_src == 0, pi, pj)
        return src_idx, base_n[axis_idx].copy(), axis_sign.copy()

    def make_constraints(centers, angles, oc, phi, R, margin, src_idx, n0, axis_sign):
        ca = np.cos(angles); sa = np.sin(angles)
        vx = ca[:, None] * verts0[None, :, 0] - sa[:, None] * verts0[None, :, 1]
        vy = sa[:, None] * verts0[None, :, 0] + ca[:, None] * verts0[None, :, 1]
        v = np.stack([vx, vy], axis=-1)
        cs = np.cos(angles[src_idx]); ss = np.sin(angles[src_idx])
        n = np.stack([cs * n0[:, 0] - ss * n0[:, 1], ss * n0[:, 0] + cs * n0[:, 1]], axis=1)
        diff = centers[pi] - centers[pj]
        proj = np.einsum("ij,ij->i", diff, n)
        hi = np.max(np.einsum("ij,ikj->ik", n, v[pi]), axis=1)
        hj = np.max(np.einsum("ij,ikj->ik", n, v[pj]), axis=1)
        sep = axis_sign * proj - hi - hj - margin
        pts = centers[:, None, :] + v
        no = outer_normals(phi)
        vals = np.einsum("ijk,lk->ijl", pts - oc, no)
        ap = R * cos30
        contain = np.concatenate([(ap - vals).ravel(), (ap + vals).ravel()]) - margin
        return np.concatenate([sep, contain])

    # ---------- full independent polish ----------
    def polish_full_once(centers0, angles0, outer0, phi0, margin):
        from scipy.optimize import minimize
        src_idx, n0, axis_sign = axis_assignment(centers0, angles0)
        init_c = centers0.copy(); init_a = angles0.copy()
        R0 = required_R(centers0, angles0, outer0, phi0) + 0.01

        def unpack(x):
            return (x[:22].reshape(11, 2), x[22:33], x[33:35], x[35], x[36])
        def objective(x):
            centers, angles, _, _, R = unpack(x)
            return float(R + 1e-6 * np.sum((centers - init_c) ** 2) + 1e-5 * np.sum((angles - init_a) ** 2))
        def ineq(x):
            centers, angles, oc, phi, R = unpack(x)
            return make_constraints(centers, angles, oc, phi, R, margin, src_idx, n0, axis_sign)

        x0 = np.concatenate([centers0.ravel(), angles0, outer0, [phi0, R0]])
        bounds = ([(-6, 6)] * 22 + [(a - 0.85, a + 0.85) for a in angles0]
                  + [(-3, 3)] * 2 + [(phi0 - 0.85, phi0 + 0.85), (2.5, 6.0)])
        res = minimize(objective, x0, method="SLSQP", constraints=[{"type": "ineq", "fun": ineq}],
                       bounds=bounds, options={"maxiter": 150, "ftol": 1e-10})
        centers, angles, oc, phi, _ = unpack(res.x)
        R = required_R(centers, angles, oc, phi) + 0.003
        if not validate(centers, angles, oc, phi, R):
            return None
        if np.min(pair_slacks(centers, angles)) < 0.0004:
            return None
        return (float(R), centers.copy(), angles.copy(), oc.copy(), float(phi))

    def polish_full(centers0, angles0, outer0, phi0, rounds=(0.0025, 0.0012)):
        cur = (centers0, angles0, outer0, phi0); out = None
        for margin in rounds:
            if time.time() > Deadline - 1.0:
                break
            try:
                res = polish_full_once(cur[0], cur[1], cur[2], cur[3], margin)
            except Exception:
                res = None
            if res is None:
                break
            out = res; cur = (res[1], res[2], res[3], res[4])
        return out

    # ---------- symmetric polish: centre + 5 opposite pairs, outer centre 0 ----------
    def expand_sym(x):
        centers = np.zeros((11, 2)); angles = np.zeros(11)
        pc = x[:10].reshape(5, 2)
        for k in range(5):
            centers[1 + 2 * k] = pc[k]; centers[2 + 2 * k] = -pc[k]
            angles[1 + 2 * k] = x[11 + k]; angles[2 + 2 * k] = x[11 + k]
        angles[0] = x[10]
        return centers, angles, x[16], x[17]

    def polish_sym_once(centers0, angles0, phi0, margin):
        from scipy.optimize import minimize
        src_idx, n0, axis_sign = axis_assignment(centers0, angles0)
        pair_init = np.array([centers0[1 + 2 * k] for k in range(5)])
        ang_init = np.array([angles0[0]] + [angles0[1 + 2 * k] for k in range(5)])
        R0 = required_R(centers0, angles0, np.zeros(2), phi0) + 0.01
        x0 = np.concatenate([pair_init.ravel(), ang_init, [phi0, R0]])
        init_x = x0.copy()

        def objective(x):
            return float(x[17] + 1e-6 * np.sum((x[:16] - init_x[:16]) ** 2))
        def ineq(x):
            centers, angles, phi, R = expand_sym(x)
            return make_constraints(centers, angles, np.zeros(2), phi, R, margin, src_idx, n0, axis_sign)

        bounds = ([(-6, 6)] * 10 + [(a - 0.85, a + 0.85) for a in ang_init]
                  + [(phi0 - 0.85, phi0 + 0.85), (2.5, 6.0)])
        res = minimize(objective, x0, method="SLSQP", constraints=[{"type": "ineq", "fun": ineq}],
                       bounds=bounds, options={"maxiter": 180, "ftol": 1e-10})
        centers, angles, phi, _ = expand_sym(res.x)
        # break exact symmetry slightly for the following full polish is optional; keep as is
        R = required_R(centers, angles, np.zeros(2), phi) + 0.003
        if not validate(centers, angles, np.zeros(2), phi, R):
            return None
        if np.min(pair_slacks(centers, angles)) < 0.0004:
            return None
        return (float(R), centers.copy(), angles.copy(), np.zeros(2), float(phi))

    def polish_sym(centers0, angles0, phi0):
        cur_c, cur_a, cur_phi = centers0, angles0, phi0; out = None
        for margin in (0.0025, 0.0012, 0.0008):
            if time.time() > Deadline - 1.0:
                break
            try:
                res = polish_sym_once(cur_c, cur_a, cur_phi, margin)
            except Exception:
                res = None
            if res is None:
                break
            out = res; cur_c, cur_a, cur_phi = res[1], res[2], res[4]
        return out

    # ---------- symmetric lattice initialisations ----------
    def make_sym(delta_deg, corner_axes):
        dr = np.deg2rad(delta_deg)
        rot = np.array([[np.cos(dr), -np.sin(dr)], [np.sin(dr), np.cos(dr)]])
        pts = [np.zeros(2)]
        for ang in (30.0, 90.0, 150.0):
            v = sqrt3 * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
            pts.append(v); pts.append(-v)
        for ax in corner_axes:
            v = 3.0 * np.array([np.cos(np.deg2rad(60.0 * ax)), np.sin(np.deg2rad(60.0 * ax))])
            pts.append(v); pts.append(-v)
        centers = (rot @ np.array(pts).T).T.copy()
        return centers

    sym_inits = []
    for delta in (-14.0, -10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0, 14.0):
        for corner_axes in ((0, 1), (1, 2), (0, 2)):
            for phi_deg in (0.0, 30.0):
                c = make_sym(delta, corner_axes)
                ang = np.full(11, np.deg2rad(delta))
                R = required_R(c, ang, np.zeros(2), np.deg2rad(phi_deg))
                sym_inits.append((R, c, ang, np.deg2rad(phi_deg)))
    sym_inits.sort(key=lambda t: t[0])

    # ---------- LP stage for asymmetric candidates (limited, time-aware) ----------
    lp_candidates = []
    def lp_from_init(base, delta_deg, phi_deg, niter=2):
        try:
            from scipy.optimize import linprog
        except Exception:
            return
        phi_rad = np.deg2rad(phi_deg)
        normals_outer = outer_normals(phi_rad)
        normals_inner = np.array([[np.cos(np.deg2rad(delta_deg + 30.0 + 60.0 * k)),
                                   np.sin(np.deg2rad(delta_deg + 30.0 + 60.0 * k))] for k in range(3)])
        supports = []
        for n in normals_outer:
            ang_n = np.arctan2(n[1], n[0])
            bd = min(abs((ang_n - np.deg2rad(delta_deg + 60.0 * m) + np.pi) % (2 * np.pi) - np.pi) for m in range(6))
            supports.append(float(np.cos(bd)))
        curr = np.array(base, dtype=float).copy(); outer_center = np.zeros(2); R_val = 4.5; ok = False
        for _ in range(niter):
            rows, rhs = [], []
            for i, j in itertools.combinations(range(11), 2):
                d = curr[i] - curr[j]
                projs = [float(np.dot(d, nn)) for nn in normals_inner]
                k = int(np.argmax(np.abs(projs))); sign = 1.0 if projs[k] >= 0 else -1.0
                row = np.zeros(25); row[2*i:2*i+2] = -sign * normals_inner[k]; row[2*j:2*j+2] = sign * normals_inner[k]
                rows.append(row); rhs.append(-(sqrt3 + 0.001))
            for i in range(11):
                for k in range(3):
                    n = normals_outer[k]
                    row = np.zeros(25); row[2*i:2*i+2] = n; row[22:24] = -n; row[24] = -apo
                    rows.append(row); rhs.append(-(supports[k] + 0.001))
                    row = np.zeros(25); row[2*i:2*i+2] = -n; row[22:24] = n; row[24] = -apo
                    rows.append(row); rhs.append(-(supports[k] + 0.001))
            c_obj = np.zeros(25); c_obj[24] = 1.0
            res = linprog(c_obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                          bounds=[(-6, 6)]*22 + [(-3, 3)]*2 + [(2.5, 6)], method="highs")
            if not res.success:
                break
            curr = res.x[:22].reshape(11, 2).copy(); outer_center = res.x[22:24].copy(); R_val = float(res.x[24]); ok = True
        if not ok:
            return
        ang = np.full(11, np.deg2rad(delta_deg))
        if np.min(pair_slacks(curr, ang)) < -1e-7:
            return
        R_use = max(R_val, required_R(curr, ang, outer_center, phi_rad)) + 0.002
        lp_candidates.append((R_use, curr.copy(), outer_center.copy(), float(delta_deg), float(phi_deg)))

    for delta in (-8.0, 0.0, 8.0):
        dr = np.deg2rad(delta); rot = np.array([[np.cos(dr), -np.sin(dr)], [np.sin(dr), np.cos(dr)]])
        for phi_deg in (0.0, 30.0):
            for extra_idx in itertools.combinations(range(6), 4):
                if time.time() > t0 + 7.0:
                    break
                base = np.vstack([base_all, corners[list(extra_idx)]])
                lp_from_init((rot @ base.T).T.copy(), delta, phi_deg)
            if time.time() > t0 + 7.0:
                break
        if time.time() > t0 + 7.0:
            break
    lp_candidates.sort(key=lambda t: t[0])

    best = None
    def consider(out):
        nonlocal best
        if out is not None and (best is None or out[0] < best[0]):
            if validate(out[1], out[2], out[3], out[4], out[0]):
                best = out

    # polish best symmetric inits: sym -> full
    tried_delta = []
    for R_use, c, ang, phi in sym_inits:
        if len(tried_delta) >= 5:
            break
        key = round(float(np.degrees(ang[0])) / 3.0)
        if key in tried_delta:
            continue
        tried_delta.append(key)
        consider((R_use + 0.002, c.copy(), ang.copy(), np.zeros(2), float(phi)))
        out_s = polish_sym(c, ang, phi)
        consider(out_s)
        if out_s is not None:
            consider(polish_full(out_s[1], out_s[2], out_s[3], out_s[4], rounds=(0.0012,)))
        if time.time() > t0 + 13.0:
            break

    # polish LP candidates fully (independent angles from the start in full polish)
    for cand in lp_candidates[:4]:
        if time.time() > t0 + 16.0:
            break
        R_use, c, oc, delta, phi_deg = cand
        ang = np.full(11, np.deg2rad(delta))
        consider((R_use, c, ang, oc, np.deg2rad(phi_deg)))
        out = polish_full(c, ang, oc, np.deg2rad(phi_deg))
        consider(out)

    if best is None:
        c = fallback_centers * 1.002; ang = np.zeros(11); oc = np.zeros(2); phi = 0.0
        R = required_R(c, ang, oc, phi) + 0.01
        best = (R, c, ang, oc, phi)

    # ---------- perturbed restarts around best, with per-hex angle kicks ----------
    scales = [0.04, 0.08, 0.14, 0.22]
    it = 0
    while time.time() < Deadline - 2.5 and best is not None:
        it += 1
        scale = scales[it % len(scales)]
        c = best[1] + rng.normal(scale=scale, size=best[1].shape)
        a = best[2] + rng.normal(scale=0.09, size=best[2].shape)
        # occasionally rotate a few outer hexagons by ~30 deg to try interlocking
        if rng.random() < 0.45:
            idx = rng.choice(11, size=3, replace=False)
            a[idx] += np.deg2rad(rng.choice([-30.0, 30.0]))
        oc = best[3] + rng.normal(scale=0.03, size=2)
        phi = best[4] + float(rng.normal(scale=0.05))
        # if the perturbed start is infeasible, SLSQP with margins may still recover; keep simple full polish
        out = polish_full(c, a, oc, phi, rounds=(0.0015,))
        consider(out)
        # occasionally re-symmetrise trial from current best via sym polish on symmetric inits is already done;
        # try one sym polish from a symmetrised version of best every 4 restarts
        if it % 4 == 0 and time.time() < Deadline - 4.0:
            # build a symmetric guess: centre + averaged opposite pairs using nearest-opposite pairing
            bc = best[1].copy()
            # pair each non-centre hex with the hex most opposite it
            used = [int(np.argmin(np.linalg.norm(bc, axis=1)))]
            pair_centers = []
            for i in range(11):
                if i in used:
                    continue
                # find opposite
                j = min((k for k in range(11) if k not in used and k != i), key=lambda k: np.linalg.norm(bc[k] + bc[i]))
                used.append(i); used.append(j)
                pair_centers.append(0.5 * (bc[i] - bc[j]))
                if len(pair_centers) == 5:
                    break
            if len(pair_centers) == 5:
                sym_c = [np.zeros(2)]
                sym_a = [best[2][used[0]] if used else 0.0]
                for pc in pair_centers:
                    sym_c.append(pc); sym_c.append(-pc); sym_a.append(best[2][0]); sym_a.append(best[2][0])
                sym_c = np.array(sym_c); sym_a = np.array(sym_a[:11])
                if sym_a.shape[0] == 11 and np.min(pair_slacks(sym_c, sym_a)) > -0.3:
                    out_s = polish_sym(sym_c, sym_a, best[4])
                    consider(out_s)
                    if out_s is not None:
                        consider(polish_full(out_s[1], out_s[2], out_s[3], out_s[4], rounds=(0.0012,)))

    R, centers, angles, oc, phi = best
    # final safety shrink: recompute exact required R and add a tiny positive margin
    R_final = required_R(centers, angles, oc, phi) + 0.0015
    if not validate(centers, angles, oc, phi, R_final):
        R_final = float(R) + 0.001
    inner = np.column_stack([centers, np.degrees(angles) % 360.0])
    outer = np.array([oc[0], oc[1], float(np.degrees(phi) % 360.0)])
    return inner.astype(float), outer.astype(float), float(R_final)
# EVOLVE-BLOCK-END
