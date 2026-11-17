"""Which shots does the crossing fix change?  (run inside Blender: blender -b -P find_affected.py)"""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bpy
import common as C, world as W, film as F
from shots import SHOTS
os.environ['GG_NOBLEND'] = '1'
W.ROAD = None
old = W.get_road(); z_old = old.z.copy()
os.environ['GG_NOBLEND'] = '0'
W.ROAD = None
new = W.get_road(); z_new = new.z.copy()
dz = np.abs(z_new - z_old)
rh = W.RoadHash(new)
cross = np.zeros(new.n, bool)
for i in range(new.n):
    for k in rh.other_near(new.x[i], new.y[i], i, 8.0):
        if abs(new.tx[i] * new.ty[k] - new.ty[i] * new.tx[k]) > 0.42:     # a genuinely angled crossing
            cross[i] = True; break
print('AFF samples with |dz|>0.08:', int((dz > 0.08).sum()), ' crossing samples:', int(cross.sum()), ' max dz %.2f' % dz.max())
film = F.Film.__new__(F.Film)
film.rd = new
import types
sc = bpy.context.scene
res = []
for sh in SHOTS:
    if sh['kind'] in ('calendar',):
        continue
    if sh['kind'] == 'map':
        res.append((sh, 'map shows the whole route')); continue
    s_list = [F.Film.head_s(film, sh['t0'], sh), F.Film.head_s(film, sh['t1'], sh), F.Film.shot_anchor(film, sh)]
    ref = sh['cam'].get('ref', '')
    if isinstance(ref, str) and ref.startswith('fixed:'):
        s_list.append(float(ref.split(':')[1]))
    lo, hi = min(s_list) - 110, max(s_list) + 40            # the pack trails ~70 m behind its head
    i0, i1 = max(0, int(lo / W.SAMP)), min(new.n - 1, int(hi / W.SAMP))
    bad = bool(((dz[i0:i1 + 1] > 0.08) | cross[i0:i1 + 1]).any())
    late = sh['t0'] >= 230.4
    if bad or late:
        res.append((sh, 'crossing/profile change' if bad else 'gate moved'))
frames = []
for sh, why in res:
    a, b = int(round(sh['t0'] * 30)), min(7480, int(round(sh['t1'] * 30)) - 1)
    frames.append((a, b, why))
    print('AFF shot %.2f-%.2f frames %d-%d  %s' % (sh['t0'], sh['t1'], a, b, why))
print('AFF TOTAL frames', sum(b - a + 1 for a, b, _ in frames), 'of', C.NF)
json.dump([(a, b) for a, b, _ in frames], open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affected_ranges.json'), 'w'))
