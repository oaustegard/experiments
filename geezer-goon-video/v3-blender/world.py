"""The road and everything beside it, built once for the whole 40 km route as merged chunk meshes."""
import math, os
import numpy as np
import bpy, bmesh
from mathutils import Vector
import common as C
from bl_util import hexc, mesh_from_arrays, add_obj, main_material, glow_material, window_material

ROAD_W = 7.2
CORRIDOR = 17.0       # no terrain this close to a road centreline; the lawn ribbon covers it
CHUNK = 100          # samples per chunk (x SAMP metres)
SAMP = 4.0           # metres between road samples


# ----------------------------------------------------------------------------------------- road frame
class Road:
    def __init__(self):
        self.s = np.arange(0, C.LEN, SAMP)
        n = len(self.s)
        self.x = np.interp(self.s, C.S_GRID, C.MX)
        self.y = np.interp(self.s, C.S_GRID, C.MY)
        z0 = (np.interp(self.s, C.S_GRID, C.ALT) - C.ALT[0]) * C.ELEV_EX
        # the ride goes out and back over the same roads with GPS altitudes that differ by metres: average every
        # pass within 14 m so a road has ONE height (otherwise a second road floats over the first)
        cell = 20.0
        gx = np.floor(self.x / cell).astype(int); gy = np.floor(self.y / cell).astype(int)
        buckets = {}
        for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
            buckets.setdefault(key, []).append(i)
        z1 = np.empty_like(z0)
        for i in range(len(z0)):
            cand = []
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    cand += buckets.get((gx[i] + a, gy[i] + b), [])
            cand = np.array(cand)
            d = np.hypot(self.x[cand] - self.x[i], self.y[cand] - self.y[i])
            z1[i] = z0[cand[d < 14.0]].mean()
        k = 5
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        self.z = np.convolve(np.pad(z1, k, mode='edge'), ker, mode='valid')
        # heading from a wide chord so GPS jitter doesn't twitch the road edges
        self._merge_repeated_passes()
        k = 3
        xi = np.pad(self.x, k, mode='edge'); yi = np.pad(self.y, k, mode='edge')
        dx = xi[2 * k:] - xi[:-2 * k]; dy = yi[2 * k:] - yi[:-2 * k]
        L = np.hypot(dx, dy); L[L < 1e-6] = 1
        self.tx = dx / L; self.ty = dy / L
        self.nx = -self.ty; self.ny = self.tx            # left normal
        self.n = n

    def _merge_repeated_passes(self, reach=16.0, min_gap_samples=75):
        """The ride goes over the same roads more than once (out-and-back, and the finish is the start corner). Each pass
        is its own GPS trace, a few metres off the other in position and height, so two road meshes overlapped and the
        later one's lawn lay on top of the earlier one's asphalt. Snap every later pass onto the earlier pass (position
        and height); opposite-direction traffic then simply uses the other half of the same road."""
        cell = 24.0
        x0, y0, z0 = self.x.copy(), self.y.copy(), self.z.copy()
        n = self.n if hasattr(self, 'n') else len(x0)
        gx = np.floor(x0 / cell).astype(int); gy = np.floor(y0 / cell).astype(int)
        grid = {}
        for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
            grid.setdefault(key, []).append(i)
        primary = np.ones(len(x0), bool)
        nx_, ny_, nz_ = x0.copy(), y0.copy(), z0.copy()
        snapped = 0
        for j in range(len(x0)):
            best, bd = None, reach
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for i in grid.get((gx[j] + a, gy[j] + b), ()):
                        if i < j - min_gap_samples and primary[i]:
                            d = math.hypot(x0[i] - x0[j], y0[i] - y0[j])
                            if d < bd:
                                best, bd = i, d
            if best is None:
                continue
            # project onto the nearer of the two segments around the primary sample
            bp, bt, bz = None, None, None
            for i2 in (best - 1, best):
                if i2 < 0 or i2 + 1 >= len(x0):
                    continue
                ax, ay, bx_, by_ = x0[i2], y0[i2], x0[i2 + 1], y0[i2 + 1]
                dx, dy = bx_ - ax, by_ - ay
                L2 = dx * dx + dy * dy or 1.0
                t = max(0.0, min(1.0, ((x0[j] - ax) * dx + (y0[j] - ay) * dy) / L2))
                px, py = ax + dx * t, ay + dy * t
                d = math.hypot(px - x0[j], py - y0[j])
                if bp is None or d < bp[0]:
                    bp = (d, px, py)
                    bz = z0[i2] + (z0[i2 + 1] - z0[i2]) * t
            if bp is not None:
                nx_[j], ny_[j], nz_[j] = bp[1], bp[2], bz
                primary[j] = False
                snapped += 1
        # the snapped chain is jagged (samples of one pass fall between samples of the other): smooth it lightly
        k = 2
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        sm = lambda a: np.convolve(np.pad(a, k, mode='edge'), ker, mode='valid')
        self.x, self.y, self.z = sm(nx_), sm(ny_), sm(nz_)
        self.merged = snapped
        if os.environ.get('GG_NOBLEND') != '1':
            self._blend_crossings()

    def _blend_crossings(self, R=30.0, min_gap_samples=75, iters=3):
        """Where two different stretches of the route cross, their GPS altitudes differ by a few tenths of a metre, which
        puts one road's lawn on top of the other road's asphalt. Pull the heights of nearby samples toward each other
        (weights fall off with distance) so crossing roads share one surface."""
        gxx, gyy = np.gradient(self.x), np.gradient(self.y)
        L_ = np.hypot(gxx, gyy); L_[L_ < 1e-6] = 1.0
        tx_, ty_ = gxx / L_, gyy / L_
        cell = R
        gx = np.floor(self.x / cell).astype(int); gy = np.floor(self.y / cell).astype(int)
        grid = {}
        for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
            grid.setdefault(key, []).append(i)
        nb = []
        for j in range(len(self.x)):
            ids, ws = [], []
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for i in grid.get((gx[j] + a, gy[j] + b), ()):
                        if abs(i - j) > min_gap_samples:
                            d = math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])
                            if d < R and abs(self.z[i] - self.z[j]) < 1.0 and not (d < 3.0 and abs(tx_[i] * ty_[j] - ty_[i] * tx_[j]) < 0.42):
                                ids.append(i); ws.append((1 - d / R) ** 2)
            nb.append((np.array(ids, int), np.array(ws)))
        z = self.z.copy()
        for _ in range(iters):
            zn = z.copy()
            for j, (ids, ws) in enumerate(nb):
                if len(ids):
                    zn[j] = (z[j] + (ws * z[ids]).sum()) / (1 + ws.sum())
            z = zn
        # keep the road profile smooth along its own length
        k = 2
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        self.z = np.convolve(np.pad(z, k, mode='edge'), ker, mode='valid')
        self.blend_delta = float(np.abs(self.z - np.convolve(np.pad(self._z_pre_blend, k, mode='edge'), ker, mode='valid')).max()) if hasattr(self, '_z_pre_blend') else 0.0

    def at(self, s):
        """(x, y, z, tx, ty) at arc-length s (scalar)."""
        f = min(max(s / SAMP, 0), self.n - 1.001)
        i = int(f); a = f - i
        lerp = lambda arr: arr[i] + (arr[i + 1] - arr[i]) * a
        tx, ty = lerp(self.tx), lerp(self.ty)
        L = math.hypot(tx, ty) or 1
        return lerp(self.x), lerp(self.y), lerp(self.z), tx / L, ty / L


ROAD = None


def get_road():
    global ROAD
    if ROAD is None:
        ROAD = Road()
    return ROAD


# ----------------------------------------------------------------------------------------- ribbons
def ribbon(rd, i0, i1, o0, o1, dz0, dz1, color_fn, dip=None):
    """Triangle strip between lateral offsets o0..o1 over samples i0..i1 (inclusive). color_fn(i, side)->rgba."""
    idx = np.arange(i0, i1 + 1)
    k = len(idx)
    V = np.zeros((2 * k, 3), np.float32)
    V[0::2, 0] = rd.x[idx] + rd.nx[idx] * o0; V[0::2, 1] = rd.y[idx] + rd.ny[idx] * o0; V[0::2, 2] = rd.z[idx] + dz0
    V[1::2, 0] = rd.x[idx] + rd.nx[idx] * o1; V[1::2, 1] = rd.y[idx] + rd.ny[idx] * o1; V[1::2, 2] = rd.z[idx] + dz1
    if dip is not None:      # sink the vertices that would lie on another road's asphalt (crossings, parallel passes)
        for j, i in enumerate(idx):
            V[2 * j, 2] += dip(V[2 * j, 0], V[2 * j, 1], int(i)); V[2 * j + 1, 2] += dip(V[2 * j + 1, 0], V[2 * j + 1, 1], int(i))
    Cc = np.zeros((2 * k, 4), np.float32)
    for j, i in enumerate(idx):
        Cc[2 * j] = color_fn(i, 0); Cc[2 * j + 1] = color_fn(i, 1)
    q = np.arange(k - 1)
    a = 2 * q; b = 2 * q + 1; c = 2 * q + 2; d = 2 * q + 3
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.int32)
    return V, F, Cc


def merge(parts):
    Vs, Fs, Cs, off = [], [], [], 0
    for V, F, Cc in parts:
        Vs.append(V); Fs.append(F + off); Cs.append(Cc); off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)


def jitter(i, k, amp=0.06):
    h = math.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return (h - math.floor(h) - 0.5) * 2 * amp


def lin(h, v=0.0):
    return np.array(hexc(h), np.float32)



# ----------------------------------------------------------------------------------------- ground palette
GC = [np.array(hexc(h), np.float32) for h in ('#5f6c38', '#7f8c4a', '#a09b58', '#8a7a4a')]


def ground_color(x, y):
    """Smoothly varying dry-lawn colour (olive -> straw -> a little brown); works on scalars or arrays."""
    x = np.atleast_1d(np.asarray(x, np.float64)); y = np.atleast_1d(np.asarray(y, np.float64))
    u = (np.sin(x * 0.011 + y * 0.007) + np.sin(x * 0.023 - y * 0.031) + 0.5 * np.sin(x * 0.061 + y * 0.047) + 2.5) / 5.0
    u = np.clip(u, 0, 0.9999) * 3.0
    i = u.astype(int); f = (u - i)[:, None]
    g = np.stack(GC)
    return (g[i] * (1 - f) + g[i + 1] * f).astype(np.float32)


# ----------------------------------------------------------------------------------------- terrain
class Terrain:
    """Ground height = inverse-distance blend of nearby road heights, so hills follow the road and parallel passes can't overlap."""
    CELL = 10.0
    REACH = 170.0

    def __init__(self, rd):
        sub = slice(None, None, 4)
        self.px, self.py, self.pz = rd.x[sub], rd.y[sub], rd.z[sub]

    def height(self, X, Y, cand=None):
        px, py, pz = (self.px, self.py, self.pz) if cand is None else cand
        dx = X[:, None] - px[None, :]; dy = Y[:, None] - py[None, :]
        d = np.hypot(dx, dy)
        wgt = 1.0 / (d + 3.0) ** 6
        wgt[d > 230] = 0
        return (wgt * pz[None, :]).sum(1) / np.maximum(wgt.sum(1), 1e-12), d.min(1)


def terrain_chunk(rd, terr, i0, i1, cell=10.0, reach=170.0):
    xs = rd.x[i0:i1 + 1]; ys = rd.y[i0:i1 + 1]
    gx0 = math.floor((xs.min() - reach) / cell); gx1 = math.ceil((xs.max() + reach) / cell)
    gy0 = math.floor((ys.min() - reach) / cell); gy1 = math.ceil((ys.max() + reach) / cell)
    gi, gj = np.meshgrid(np.arange(gx0, gx1 + 1), np.arange(gy0, gy1 + 1), indexing='ij')
    X = (gi * cell).ravel().astype(np.float64); Y = (gj * cell).ravel().astype(np.float64)
    # candidate road samples near this tile
    m = (terr.px > xs.min() - 260) & (terr.px < xs.max() + 260) & (terr.py > ys.min() - 260) & (terr.py < ys.max() + 260)
    cand = (terr.px[m], terr.py[m], terr.pz[m])
    # distance to this chunk's own road centreline decides which cells it owns (cells near the road)
    d_own = np.hypot(X[:, None] - xs[None, ::2], Y[:, None] - ys[None, ::2]).min(1)
    H, dmin = terr.height(X, Y, cand)
    H = H - 0.9 + 0.35 * np.sin(X * 0.012) * np.cos(Y * 0.010) * np.clip((dmin - 20) / 60, 0, 1)
    # true distance to any road centreline (4 m samples): terrain never gets built inside the road + lawn corridor, because
    # a 10 m grid can't follow a sloping road and would poke through the asphalt (bikes 'below the road', riders 'in the grass')
    allx, ally = rd.x, rd.y
    m2 = (allx > xs.min() - reach - 40) & (allx < xs.max() + reach + 40) & (ally > ys.min() - reach - 40) & (ally < ys.max() + reach + 40)
    dtrue = np.hypot(X[:, None] - allx[m2][None, :], Y[:, None] - ally[m2][None, :]).min(1)
    too_near = (dtrue < CORRIDOR)
    nx, ny = gi.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    cells = []
    own = (d_own <= reach).reshape(nx, ny)
    for a in range(nx - 1):
        for b in range(ny - 1):
            if own[a, b] or own[a + 1, b] or own[a, b + 1] or own[a + 1, b + 1]:
                if too_near[idx[a, b]] or too_near[idx[a + 1, b]] or too_near[idx[a, b + 1]] or too_near[idx[a + 1, b + 1]]:
                    continue
                cells.append((idx[a, b], idx[a + 1, b], idx[a + 1, b + 1], idx[a, b + 1]))
    if not cells:
        return None
    cells = np.array(cells)
    used = np.unique(cells)
    remap = -np.ones(nx * ny, np.int64); remap[used] = np.arange(len(used))
    V = np.stack([X[used], Y[used], H[used]], 1).astype(np.float32)
    quad = remap[cells]
    F = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]).astype(np.int32)
    # colour: patchy lawn, darker toward the horizon
    Cc = ground_color(X[used], Y[used]) * (0.92 - 0.14 * np.clip((dmin[used] - 40) / 130, 0, 1))[:, None]
    Cc[:, 3] = 1
    return V, F, Cc.astype(np.float32)


# ----------------------------------------------------------------------------------------- templates
def _tmpl_icosphere(sub=1):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    V = np.array([v.co[:] for v in bm.verts], np.float32)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], np.int32)
    bm.free()
    return V, F


def _tmpl_cone(seg=7, r=1.0, h=1.0, z0=0.0):
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    V = np.concatenate([np.stack([r * np.cos(ang), r * np.sin(ang), np.full(seg, z0)], 1), [[0, 0, z0 + h]], [[0, 0, z0]]]).astype(np.float32)
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F.append([i, j, seg]); F.append([j, i, seg + 1])
    return V, np.array(F, np.int32)


def _tmpl_box(w, d, h, z0=0.0):
    x, y = w / 2, d / 2
    V = np.array([[-x, -y, z0], [x, -y, z0], [x, y, z0], [-x, y, z0], [-x, -y, z0 + h], [x, -y, z0 + h], [x, y, z0 + h], [-x, y, z0 + h]], np.float32)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return V, F


def _tmpl_house_roof(w, d, h, ridge):
    x, y = w / 2, d / 2
    V = np.array([[-x - .3, -y - .3, h], [x + .3, -y - .3, h], [x + .3, y + .3, h], [-x - .3, y + .3, h], [-x - .3, 0, h + ridge], [x + .3, 0, h + ridge]], np.float32)
    F = np.array([[0, 1, 5], [0, 5, 4], [3, 4, 5], [3, 5, 2], [0, 4, 3], [1, 2, 5], [0, 3, 2], [0, 2, 1]], np.int32)
    return V, F


def _place(Vt, Ft, pos, scale, yaw, color, voff, colt=None):
    c, s = math.cos(yaw), math.sin(yaw)
    sx, sy, sz = scale if hasattr(scale, '__len__') else (scale, scale, scale)
    V = np.empty_like(Vt)
    V[:, 0] = pos[0] + (Vt[:, 0] * sx) * c - (Vt[:, 1] * sy) * s
    V[:, 1] = pos[1] + (Vt[:, 0] * sx) * s + (Vt[:, 1] * sy) * c
    V[:, 2] = pos[2] + Vt[:, 2] * sz
    Cc = np.tile(np.array(color, np.float32), (len(Vt), 1)) if colt is None else colt
    return V, Ft + voff, Cc


# ----------------------------------------------------------------------------------------- build
def chunk_bbox(rd, ci):
    i0 = ci * CHUNK; i1 = min(rd.n - 1, i0 + CHUNK)
    return rd.x[i0:i1 + 1].min(), rd.x[i0:i1 + 1].max(), rd.y[i0:i1 + 1].min(), rd.y[i0:i1 + 1].max()


class RoadHash:
    """Spatial hash of the road samples (any pass of the route)."""
    def __init__(self, rd, cell=30.0):
        self.cell = cell
        self.x, self.y, self.z = rd.x, rd.y, rd.z
        self.tx, self.ty = rd.tx, rd.ty
        self.b = {}
        gx = np.floor(rd.x / cell).astype(int); gy = np.floor(rd.y / cell).astype(int)
        for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
            self.b.setdefault(key, []).append(i)

    def other_near(self, x, y, i, r, min_gap=75, dzmax=1.0):
        """Samples from a DIFFERENT stretch of the route within r metres that sit at about the same height, i.e. a
        road that crosses or runs alongside this one. Same-road duplicates (merged passes) and bridges/underpasses
        (heights >= dzmax apart) are not 'other roads' for this purpose."""
        gx, gy = int(x // self.cell), int(y // self.cell)
        out = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for k in self.b.get((gx + a, gy + b), ()):
                    if abs(k - i) <= min_gap:
                        continue
                    d = math.hypot(self.x[k] - x, self.y[k] - y)
                    if d >= r or abs(self.z[k] - self.z[i]) >= dzmax:
                        continue
                    if d < 3.0 and abs(self.tx[i] * self.ty[k] - self.ty[i] * self.tx[k]) < 0.42:
                        continue          # the same road driven in the other direction (or again)
                    out.append(k)
        return out

    def clear(self, x, y, r):
        gx, gy = int(x // self.cell), int(y // self.cell)
        idx = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                idx += self.b.get((gx + a, gy + b), [])
        if not idx:
            return True
        idx = np.array(idx)
        return float(np.hypot(self.x[idx] - x, self.y[idx] - y).min()) >= r


def build_world(coll, seed=7, only=None):
    """Whole-route world as merged chunk meshes. `only(ci)` limits which chunks get built (one Blender process per shot)."""
    rd = get_road()
    mat = main_material()
    glow = window_material('M_windows')
    lamp = window_material('M_lamps')
    nchunks = (rd.n + CHUNK - 1) // CHUNK

    ico = _tmpl_icosphere(1)
    cone = _tmpl_cone(7)
    trunk = _tmpl_cone(6, 0.16, 2.4)
    hbox = _tmpl_box(1, 1, 1)

    asphalt = lambda i, sd: np.array(hexc('#3b3d44'), np.float32) * (0.92 + jitter(i, sd, 0.06))
    walk = lambda i, sd: np.array(hexc('#a7a39a'), np.float32) * (0.95 + jitter(i, sd, 0.04))
    grass_cols = ['#6aa84f', '#78b055', '#5f9b48', '#82b95a']
    grass = lambda i, sd: ground_color(rd.x[i], rd.y[i])[0] * (1.0 + jitter(i, sd, 0.05))
    far_grass = lambda i, sd: np.array(hexc(grass_cols[(i // 9 + sd) % 4]), np.float32) * (0.78 + jitter(i, sd + 3, 0.05))
    yellow = lambda i, sd: np.array(hexc('#e8b923'), np.float32)
    white = lambda i, sd: np.array(hexc('#e9e9e4'), np.float32)

    terr = Terrain(rd)
    rh = RoadHash(rd)

    def dip(x, y, i):
        return -0.7 if rh.other_near(x, y, i, 4.6) else 0.0
    chunk_objs = []
    for ci in range(nchunks):
        i0 = ci * CHUNK; i1 = min(rd.n - 1, i0 + CHUNK)
        if i1 - i0 < 2:
            continue
        if only is not None and not only(ci):
            continue
        rng = np.random.default_rng(seed * 100003 + ci)       # per-chunk, so any subset of the world is identical
        parts = []
        parts.append(ribbon(rd, i0, i1, -ROAD_W / 2, ROAD_W / 2, 0, 0, asphalt))
        parts.append(ribbon(rd, i0, i1, -0.2, -0.08, 0.02, 0.02, yellow)); parts.append(ribbon(rd, i0, i1, 0.08, 0.2, 0.02, 0.02, yellow))
        parts.append(ribbon(rd, i0, i1, ROAD_W / 2 - 0.45, ROAD_W / 2 - 0.33, 0.02, 0.02, white))
        parts.append(ribbon(rd, i0, i1, -ROAD_W / 2 + 0.33, -ROAD_W / 2 + 0.45, 0.02, 0.02, white))
        for sd in (1, -1):
            a, b = sd * ROAD_W / 2, sd * (ROAD_W / 2 + 0.25)
            parts.append(ribbon(rd, i0, i1, min(a, b), max(a, b), 0.12, 0.12, walk, dip))           # curb top
            a2, b2 = sd * (ROAD_W / 2 + 0.25), sd * (ROAD_W / 2 + 1.9)
            parts.append(ribbon(rd, i0, i1, min(a2, b2), max(a2, b2), 0.12, 0.12, walk, dip))       # sidewalk
            a3, b3 = sd * (ROAD_W / 2 + 1.9), sd * (ROAD_W / 2 + 17.5)
            # ribbon() runs from the smaller offset to the larger one, so on the right (sd<0) the OUTER edge comes first;
            # the lawn drops away from the road on both sides
            dz_lo, dz_hi = (-0.55, 0.0) if sd < 0 else (0.0, -0.55)
            parts.append(ribbon(rd, i0, i1, min(a3, b3), max(a3, b3), dz_lo, dz_hi, grass, dip))
        V, F, Cc = merge(parts)
        me = mesh_from_arrays(f'road_{ci}', V, F, Cc)
        me.materials.append(mat)
        chunk_objs.append(add_obj(me, f'road_{ci}', coll))
        tt = terrain_chunk(rd, terr, i0, i1)
        if tt is not None:
            me = mesh_from_arrays(f'ground_{ci}', *tt)
            me.materials.append(mat)
            chunk_objs.append(add_obj(me, f'ground_{ci}', coll))

        # ---- trees / houses / lamps for this chunk
        tp, hp, wp, lp, lg = [], [], [], [], []
        tv = hv = wv = lv = gv = 0
        s_lo, s_hi = i0 * SAMP, i1 * SAMP
        for sd in (1, -1):
            s = s_lo + rng.uniform(0, 8)
            while s < s_hi:
                i = min(rd.n - 1, int(s / SAMP))
                px, py, pz = rd.x[i], rd.y[i], rd.z[i]
                nx, ny = sd * rd.nx[i], sd * rd.ny[i]
                off = rng.uniform(ROAD_W / 2 + 9.5, ROAD_W / 2 + 34)
                x, y = px + nx * off, py + ny * off
                if not rh.clear(x, y, 9.0):
                    s += rng.uniform(4, 8)
                    continue
                kind = rng.random()
                h = rng.uniform(5.5, 12.0)
                g = np.array(hexc(rng.choice(['#4f9a3c', '#5aa843', '#3f8a38', '#6bb04a', '#c98a2a', '#b5542d'] if rng.random() < 0.08 else ['#4f9a3c', '#5aa843', '#3f8a38', '#6bb04a', '#478f40'])), np.float32) * rng.uniform(0.85, 1.05)
                brown = np.array(hexc('#5b4330'), np.float32)
                z0 = pz - 0.15
                if kind < 0.72:     # round deciduous: trunk + three blobs
                    V, F, Cc = _place(trunk[0], trunk[1], (x, y, z0), (1, 1, h * 0.42 / 2.4), 0, brown, 0); tp.append((V, F, Cc)); tv += len(V)
                    for bx, by, bz, br in ((0, 0, h * 0.62, h * 0.30), (h * 0.12, h * 0.05, h * 0.78, h * 0.24), (-h * 0.1, -h * 0.08, h * 0.74, h * 0.23)):
                        V, F, Cc = _place(ico[0], ico[1], (x + bx, y + by, z0 + bz), (br, br, br * 0.85), rng.uniform(0, 6.28), g * rng.uniform(0.92, 1.05), 0); tp.append((V, F, Cc)); tv += len(V)
                else:               # conifer
                    V, F, Cc = _place(trunk[0], trunk[1], (x, y, z0), (1, 1, 0.5), 0, brown, 0); tp.append((V, F, Cc)); tv += len(V)
                    gg = g * np.array([0.6, 0.85, 0.7, 1], np.float32)
                    for lev in range(3):
                        r = h * (0.30 - 0.07 * lev); zc = h * (0.22 + 0.27 * lev)
                        V, F, Cc = _place(cone[0], cone[1], (x, y, z0 + zc), (r, r, h * 0.38), rng.uniform(0, 6.28), gg, 0); tp.append((V, F, Cc)); tv += len(V)
                s += rng.uniform(8.5, 15)
            # houses
            s = s_lo + rng.uniform(0, 20)
            while s < s_hi:
                i = min(rd.n - 1, int(s / SAMP))
                if rng.random() < 0.72:
                    px, py, pz = rd.x[i], rd.y[i], rd.z[i]
                    nx, ny, tx, ty = sd * rd.nx[i], sd * rd.ny[i], rd.tx[i], rd.ty[i]
                    off = rng.uniform(ROAD_W / 2 + 17, ROAD_W / 2 + 26)
                    x, y = px + nx * off, py + ny * off
                    if not rh.clear(x, y, 20.0):
                        s += rng.uniform(10, 20)
                        continue
                    yaw = math.atan2(ty, tx)
                    w, d, hh = rng.uniform(9, 13), rng.uniform(8, 10), rng.uniform(4.6, 6.6)
                    wall = np.array(hexc(rng.choice(['#e9d8b4', '#d9e4ec', '#f0c9c0', '#cfe0c3', '#f2e6a8', '#c7bcd6', '#e8e0d0', '#b5c9d9', '#d6a78a'])), np.float32)
                    roofc = np.array(hexc(rng.choice(['#4a3b36', '#38414a', '#5d3b32', '#2f3b33'])), np.float32)
                    z0 = pz - 0.3
                    V, F, Cc = _place(hbox[0] + np.array([0, 0, 0], np.float32), hbox[1], (x, y, z0 - 0.5), (w, d, hh + 0.5), yaw, wall * rng.uniform(0.95, 1.03), 0); hp.append((V, F, Cc)); hv += len(V)
                    rv = _tmpl_house_roof(1, 1, 1, 0.45)
                    V, F, Cc = _place(rv[0], rv[1], (x, y, z0 + hh - 0.5), (w, d, 1), yaw, roofc, 0); V[:, 2] += 0; hp.append((V, F, Cc)); hv += len(V)
                    # windows on the road-facing wall (outward = toward the road)
                    ox, oy = -nx, -ny      # direction from house centre to road
                    nwin = int(rng.integers(2, 4))
                    for wi in range(nwin):
                        u = (wi - (nwin - 1) / 2) * (w / (nwin + 0.2))
                        cx = x + tx * u + ox * (d / 2 + 0.02); cy = y + ty * u + oy * (d / 2 + 0.02)
                        wq = np.array([[-0.7, 0, -0.55], [0.7, 0, -0.55], [0.7, 0, 0.7], [-0.7, 0, 0.7]], np.float32)
                        # window quad spans along the wall tangent
                        P = np.stack([cx + tx * wq[:, 0], cy + ty * wq[:, 0], np.full(4, z0 + 1.9) + wq[:, 2] * 1.0], 1).astype(np.float32)
                        cc = np.tile(np.array(hexc(rng.choice(['#ffd27a', '#ffc15a', '#fff0b0', '#ffe2a0'])), np.float32), (4, 1))
                        wp.append((P, np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], np.int32), cc)); wv += 4
                s += rng.uniform(22, 34)
            # street lamps on this side every 40 m (left side only to keep the count down)
            if sd == 1:
                s = s_lo + 8
                while s < s_hi:
                    i = min(rd.n - 1, int(s / SAMP))
                    px, py, pz = rd.x[i], rd.y[i], rd.z[i]
                    nx, ny = rd.nx[i], rd.ny[i]
                    off = ROAD_W / 2 + 2.6
                    x, y = px + nx * off, py + ny * off
                    pole = _tmpl_cone(6, 0.1, 7.0)
                    V, F, Cc = _place(pole[0], pole[1], (x, y, pz), (1, 1, 1), 0, np.array(hexc('#3a3f47'), np.float32), 0); lp.append((V, F, Cc)); lv += len(V)
                    head = _place(ico[0], ico[1], (x - nx * 0.9, y - ny * 0.9, pz + 6.9), (0.34, 0.34, 0.15), 0, np.array(hexc('#fff1c9'), np.float32), 0); lg.append(head); gv += len(head[0])
                    s += 42
        for nm, parts_, m_ in (('trees', tp, mat), ('houses', hp, mat), ('windows', wp, glow), ('poles', lp, mat), ('lamps', lg, lamp)):
            if parts_:
                V, F, Cc = merge(parts_)
                me = mesh_from_arrays(f'{nm}_{ci}', V, F, Cc)
                me.materials.append(m_)
                chunk_objs.append(add_obj(me, f'{nm}_{ci}', coll))
    return chunk_objs
