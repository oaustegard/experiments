"""Procedural cyclist + bike. Everything is posed per frame in python (analytic two-bone IK for legs and arms)."""
import math
import numpy as np
import bpy, bmesh
from mathutils import Matrix, Vector
from bl_util import (hexc, add_obj, main_material, glow_material, unit_cylinder, unit_sphere, unit_box,
                     mesh_from_arrays, bm_to_mesh, limb_matrix)

_SHARED = {}


def shared():
    if not _SHARED:
        _SHARED['cyl'] = unit_cylinder(14, 'u_cyl')
        _SHARED['cyl6'] = unit_cylinder(10, 'u_cyl6')
        _SHARED['sph'] = unit_sphere(14, 9, 'u_sph')
        _SHARED['sph_lo'] = unit_sphere(10, 7, 'u_sph_lo')
        _SHARED['box'] = unit_box('u_box')
        _SHARED['box'].materials.append(main_material())
        bl = unit_box('u_box_light'); bl.materials.append(glow_material('M_bikelight'))
        _SHARED['box_light'] = bl
        _SHARED['wheel'] = {}
    return _SHARED


# ----------------------------------------------------------------------------------------- bike meshes
def _tube(bm, a, b, r, seg=6):
    d = Vector(b) - Vector(a)
    L = d.length
    res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r, depth=L)
    verts = res['verts']
    z = d / L
    up = Vector((0, 0, 1)) if abs(z.z) < 0.95 else Vector((1, 0, 0))
    x = up.cross(z).normalized(); y = z.cross(x)
    rot = Matrix(((x.x, y.x, z.x, 0), (x.y, y.y, z.y, 0), (x.z, y.z, z.z, 0), (0, 0, 0, 1)))
    M = Matrix.Translation((Vector(a) + Vector(b)) / 2) @ rot
    bmesh.ops.transform(bm, matrix=M, verts=verts)
    return verts


def _colored(bm, verts, col, cl):
    for v in verts:
        for l in v.link_loops:
            l[cl] = col


def bike_mesh(name, frame_hex, kind='road'):
    """Static bike frame + bars + saddle in rider space (x fwd, z up, origin under the bottom bracket)."""
    bm = bmesh.new()
    cl = bm.loops.layers.float_color.new('Col')
    fc = hexc(frame_hex); dark = hexc('#17181c'); metal = hexc('#9ea4ad')
    BB = (0, 0, 0.27); RA = (-0.41, 0, 0.34); FA = (0.61, 0, 0.34)
    ST = (-0.20, 0, 0.97); HTt = (0.49, 0, 0.93); HTb = (0.455, 0, 0.70)
    tubes = [  # (a, b, r, colour)
        (BB, ST, 0.017, fc), (ST, HTt, 0.016, fc), (BB, HTb, 0.021, fc), (HTb, HTt, 0.021, fc),
        ((-0.02, 0.06, 0.28), (-0.41, 0.06, 0.34), 0.011, fc), ((-0.02, -0.06, 0.28), (-0.41, -0.06, 0.34), 0.011, fc),
        ((-0.20, 0.0, 0.95), (-0.41, 0.06, 0.34), 0.009, fc), ((-0.20, 0.0, 0.95), (-0.41, -0.06, 0.34), 0.009, fc),
        (HTb, (0.61, 0.045, 0.34), 0.013, dark), (HTb, (0.61, -0.045, 0.34), 0.013, dark),
        (ST, (-0.215, 0, 1.03), 0.012, metal),
        (HTt, (0.535, 0, 0.99), 0.014, dark),
    ]
    stem_end = (0.545, 0, 0.995)
    # bars: drop bars (road/single/gravel) -- hoods forward of stem, drops curve down
    tubes += [(stem_end, (0.55, 0.20, 0.99), 0.012, dark), (stem_end, (0.55, -0.20, 0.99), 0.012, dark)]
    for sd in (1, -1):
        tubes += [((0.55, sd * 0.20, 0.99), (0.64, sd * 0.19, 0.96), 0.014, dark),     # hoods
                  ((0.64, sd * 0.19, 0.96), (0.68, sd * 0.19, 0.89), 0.012, dark),
                  ((0.68, sd * 0.19, 0.89), (0.60, sd * 0.19, 0.84), 0.012, dark)]
    for a, b, r, c in tubes:
        _colored(bm, _tube(bm, a, b, r), c, cl)
    # saddle
    res = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((-0.205, 0, 1.045)) @ Matrix.Diagonal((0.26, 0.13, 0.05, 1)), verts=res['verts'])
    _colored(bm, res['verts'], dark, cl)
    # derailleur / cassette blob
    res = bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.05)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((-0.41, -0.05, 0.34)), verts=res['verts'])
    _colored(bm, res['verts'], metal, cl)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    return me


def wheel_mesh(tire_minor=0.022, r_out=0.335):
    key = (round(tire_minor, 3), round(r_out, 3))
    cache = shared()['wheel']
    if key in cache:
        return cache[key]
    bm = bmesh.new()
    cl = bm.loops.layers.float_color.new('Col')
    # tyre torus around the y axis
    nM, nm = 28, 6
    R0 = r_out - tire_minor
    verts = []
    for i in range(nM):
        a = 2 * math.pi * i / nM
        row = []
        for j in range(nm):
            b = 2 * math.pi * j / nm
            rr = R0 + tire_minor * math.cos(b)
            row.append(bm.verts.new((rr * math.cos(a), tire_minor * math.sin(b), rr * math.sin(a))))
        verts.append(row)
    for i in range(nM):
        for j in range(nm):
            bm.faces.new((verts[i][j], verts[(i + 1) % nM][j], verts[(i + 1) % nM][(j + 1) % nm], verts[i][(j + 1) % nm]))
    tire = hexc('#101114')
    for v in bm.verts:
        for l in v.link_loops:
            l[cl] = tire
    # rim (thin torus) + spokes + hub
    metal = hexc('#b9bec7')
    rim_r = R0 - tire_minor * 0.75
    for k in range(20):
        a = 2 * math.pi * k / 20 + 0.15
        tilt = 0.06 if k % 2 else -0.06
        vs = _tube(bm, (0.02 * math.cos(a), tilt * 0.4, 0.02 * math.sin(a)), (rim_r * math.cos(a), 0, rim_r * math.sin(a)), 0.0028, 4)
        _colored(bm, vs, metal, cl)
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.028, radius2=0.028, depth=0.1)
    bmesh.ops.rotate(bm, verts=res['verts'], cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'X'))
    _colored(bm, res['verts'], metal, cl)
    # one extra aero disc-ish ring for the rim
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new('wheel')
    bm.to_mesh(me); bm.free()
    me.materials.append(main_material())
    cache[key] = me
    return me


def crank_mesh():
    """Chainring + two crank arms, in a frame where the arms point along +x (right pedal) and -x (left pedal)."""
    bm = bmesh.new()
    cl = bm.loops.layers.float_color.new('Col')
    metal = hexc('#8d939c'); dark = hexc('#202228')
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=0.095, radius2=0.095, depth=0.012)
    bmesh.ops.rotate(bm, verts=res['verts'], cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'X'))
    bmesh.ops.transform(bm, matrix=Matrix.Translation((0, 0.05, 0)), verts=res['verts'])
    _colored(bm, res['verts'], metal, cl)
    for sd, ox in ((1, 1), (-1, -1)):
        vs = _tube(bm, (0, sd * 0.06, 0), (ox * 0.1725, sd * 0.07, 0), 0.011, 5)
        _colored(bm, vs, dark, cl)
    me = bpy.data.meshes.new('crank')
    bm.to_mesh(me); bm.free()
    return me


# ----------------------------------------------------------------------------------------- IK
def ik2(a, b, l1, l2, pole):
    a = Vector(a); b = Vector(b)
    d = b - a
    L = min(max(d.length, 1e-5), (l1 + l2) * 0.999)
    dh = d.normalized() if d.length > 1e-6 else Vector((0, 0, -1))
    cosA = max(-1, min(1, (l1 * l1 + L * L - l2 * l2) / (2 * l1 * L)))
    sinA = math.sqrt(max(0.0, 1 - cosA * cosA))
    p = Vector(pole)
    p = p - dh * p.dot(dh)
    p = p.normalized() if p.length > 1e-6 else Vector((0, 1, 0))
    return a + dh * (l1 * cosA) + p * (l1 * sinA), a + dh * L


def seg_mx(a, b, rx, ry):
    """Cylinder from a to b with elliptical section: x-axis 'depth' (perp in the sagittal plane), y lateral."""
    a = Vector(a); b = Vector(b)
    d = b - a
    L = d.length or 1e-4
    z = d / L
    y = Vector((0, 1, 0))
    x = y.cross(z)
    if x.length < 1e-4:
        x = Vector((1, 0, 0))
    x = x.normalized()
    y = z.cross(x)
    rot = Matrix(((x.x, y.x, z.x, 0), (x.y, y.y, z.y, 0), (x.z, y.z, z.z, 0), (0, 0, 0, 1)))
    return Matrix.Translation((a + b) / 2) @ rot @ Matrix.Diagonal((rx, ry, L, 1))


def sph_mx(p, rx, ry=None, rz=None, rot=None):
    ry = rx if ry is None else ry; rz = rx if rz is None else rz
    M = Matrix.Translation(p)
    if rot is not None:
        M = M @ rot.to_4x4()
    return M @ Matrix.Diagonal((rx, ry, rz, 1))


def box_mx(p, sx, sy, sz, rot=None):
    M = Matrix.Translation(p)
    if rot is not None:
        M = M @ rot.to_4x4()
    return M @ Matrix.Diagonal((sx, sy, sz, 1))


# ----------------------------------------------------------------------------------------- the rider
class Rider:
    def __init__(self, spec, coll, detail=True):
        self.spec = spec
        self.detail = detail
        sh = shared()
        self.mat = main_material()
        b = spec.get('build', 1.0)
        self.b = b
        self.root = bpy.data.objects.new('r_' + spec['id'], None)
        coll.objects.link(self.root)
        self.parts = {}
        sk = hexc(spec.get('skin', '#e0ac86')); jr = hexc(spec['jersey']); sht = hexc(spec.get('shorts', '#1b1c22'))
        bikecol = spec.get('frame', spec['jersey'])
        # bike
        bike = bike_mesh('bike_' + spec['id'], bikecol, spec.get('bike', 'road'))
        bike.materials.append(self.mat)
        self._part('bike', bike, (1, 1, 1, 1))
        tm = 0.032 if spec.get('bike') == 'gravel' else 0.021
        ro = 0.355 if spec.get('bike') == 'gravel' else 0.335
        wm = wheel_mesh(tm, ro)
        self.wr = ro
        self._part('wheelF', wm, (1, 1, 1, 1)); self._part('wheelR', wm, (1, 1, 1, 1))
        cm = shared().get('crank_mesh')
        if cm is None:
            cm = crank_mesh(); cm.materials.append(self.mat); shared()['crank_mesh'] = cm
        self._part('crank', cm, (1, 1, 1, 1))
        # body
        self._part('torso', sh['cyl'], jr); self._part('head', sh['sph'], sk)
        self._part('helmet', sh['sph'], hexc(spec.get('helm', '#f4f4f4')))
        for s in ('L', 'R'):
            self._part('thigh' + s, sh['cyl6'], sht); self._part('shin' + s, sh['cyl6'], sk)
            self._part('knee' + s, sh['sph_lo'], sht)
            self._part('shoe' + s, sh['box'], hexc(spec.get('shoe', '#e8e8e8')))
            self._part('uarm' + s, sh['cyl6'], jr); self._part('farm' + s, sh['cyl6'], sk)
            self._part('elbow' + s, sh['sph_lo'], jr); self._part('hand' + s, sh['sph_lo'], hexc('#1c1c20'))
            self._part('shoulder' + s, sh['sph_lo'], jr)
        self._part('hips', sh['sph'], sht)
        if spec.get('pack'):
            self._part('pack', sh['box'], hexc('#2b2f3a')); self._part('packlid', sh['box'], hexc('#e9c46a'))
        if spec.get('stripe'):
            self._part('stripe', sh['cyl'], hexc(spec['stripe']))
        if spec.get('stache'):
            self._part('stache', sh['box'], hexc(spec.get('hair', '#3a2a1a')))
        if spec.get('glasses', True):
            self._part('glasses', sh['box'], hexc('#101216'))
        self.face = spec.get('face', False)
        if self.face:
            for s in ('L', 'R'):
                self._part('eye' + s, sh['sph_lo'], hexc('#ffffff')); self._part('pupil' + s, sh['sph_lo'], hexc('#101216'))
            self._part('mouth', sh['sph_lo'], hexc('#5a1f1f'))
        # lights
        self._part('tail', sh['box_light'], hexc('#ff2a2a')); self._part('head_l', sh['box_light'], hexc('#fff6dd'))

    def _part(self, name, mesh, color):
        ob = bpy.data.objects.new(f'{self.spec["id"]}_{name}', mesh)
        ob.color = color if len(color) == 4 else (*color, 1)
        if not len(mesh.materials):
            mesh.materials.append(self.mat)
        ob.parent = self.root
        self.root.users_collection[0].objects.link(ob)
        self.parts[name] = ob
        return ob

    # ---- per-frame pose --------------------------------------------------------------------------
    def pose(self, theta, spin, stand=0.0, roll=0.0, steer=0.0, mood=None, stress=0.0, look=0.0, lights=0.0, flat_back=False):
        """theta crank angle, spin wheel angle (rad). Local frame: x fwd, y left, z up, origin below the BB."""
        sp = self.spec
        p = self.parts
        b = self.b
        up = sp.get('upright', 0.0)
        # bike roll (rocking when standing) about the ground-contact line
        rollM = Matrix.Rotation(roll, 4, 'X')
        bikeM = rollM
        p['bike'].matrix_basis = bikeM
        # wheels
        yaw = Matrix.Rotation(steer, 4, 'Z')
        p['wheelR'].matrix_basis = bikeM @ Matrix.Translation((-0.41, 0, 0.34)) @ Matrix.Rotation(spin, 4, 'Y')
        p['wheelF'].matrix_basis = bikeM @ Matrix.Translation((0.61, 0, 0.34)) @ Matrix.Rotation(spin, 4, 'Y')
        # crank
        BB = Vector((0, 0, 0.27))
        p['crank'].matrix_basis = bikeM @ Matrix.Translation(BB) @ Matrix.Rotation(-theta, 4, 'Y')
        # joints
        def B(v):                      # point in bike frame -> rider frame (apply roll)
            return bikeM @ Vector(v)
        H = Vector((-0.25 + 0.0 * stand, 0, 1.02 + 0.14 * stand + 0.0))
        H = bikeM @ H
        lean = math.radians(40 + 16 * stand + 22 * up)
        tl = 0.50 * b
        S = H + Vector((math.cos(lean) * tl, 0, math.sin(lean) * tl))
        hips_w = 0.085 * b
        sh_w = 0.185 * b
        crank_r = 0.1725
        # torso
        p['torso'].matrix_basis = seg_mx(H + Vector((0.0, 0, 0.0)), S, 0.115 * b, 0.17 * b)
        p['hips'].matrix_basis = sph_mx(H + Vector((-0.02, 0, -0.0)), 0.13 * b, 0.15 * b, 0.11 * b)
        if 'stripe' in p:
            p['stripe'].matrix_basis = seg_mx(H + (S - H) * 0.44, H + (S - H) * 0.58, 0.121 * b, 0.178 * b)
        # head
        neck = S + Vector((0.035, 0, 0.05))
        hc = neck + Vector((0.06 + 0.02 * up, 0, 0.115 + 0.03 * stress * 0))
        hr = 0.098
        yawh = Matrix.Rotation(look + steer, 3, 'Z')
        p['head'].matrix_basis = sph_mx(hc, hr * 0.98, hr * 0.92, hr * 1.04, yawh)
        p['helmet'].matrix_basis = sph_mx(hc + Vector((-0.012, 0, 0.022)), hr * 1.28, hr * 1.1, hr * 0.98, yawh)
        fwd = yawh @ Vector((1, 0, 0)); lat = yawh @ Vector((0, 1, 0))
        if 'glasses' in p:
            gl = hc + fwd * (hr * 0.82) + Vector((0, 0, hr * 0.12))
            p['glasses'].matrix_basis = box_mx(gl, 0.03, hr * 1.55, hr * 0.42, yawh)
        if 'stache' in p:
            p['stache'].matrix_basis = box_mx(hc + fwd * (hr * 0.93) + Vector((0, 0, -hr * 0.42)), 0.026, hr * 1.0, 0.022, yawh)
        if self.face:
            wide = 1.0 + 0.55 * (mood == 'wide' or stress > 0.6)
            for s, sd in (('L', 1), ('R', -1)):
                ep = hc + fwd * (hr * 0.88) + lat * (sd * hr * 0.36) + Vector((0, 0, hr * 0.1))
                p['eye' + s].matrix_basis = sph_mx(ep, 0.024 * wide, 0.027 * wide, 0.028 * wide * (0.45 if mood == 'squint' else 1.0), yawh)
                p['pupil' + s].matrix_basis = sph_mx(ep + fwd * 0.016, 0.012, 0.012, 0.013, yawh)
            mp = hc + fwd * (hr * 0.9) + Vector((0, 0, -hr * 0.46))
            if mood == 'o' or stress > 0.65:
                p['mouth'].matrix_basis = sph_mx(mp, 0.02, 0.026, 0.03, yawh)
            elif mood == 'grit':
                p['mouth'].matrix_basis = sph_mx(mp, 0.014, 0.05, 0.012, yawh)
            else:
                p['mouth'].matrix_basis = sph_mx(mp, 0.012, 0.036, 0.008, yawh)
        # legs
        L1 = 0.43 * b ** 0.5; L2 = 0.43 * b ** 0.5
        for s, sd, ph in (('R', -1, 0.0), ('L', 1, math.pi)):
            ang = theta + ph
            pedal = BB + Vector((crank_r * math.cos(ang), sd * 0.085, crank_r * math.sin(ang)))
            pedal = bikeM @ pedal
            ankle = pedal + Vector((-0.03, 0, 0.085))
            hip = H + Vector((0, sd * hips_w, 0))
            knee, ank = ik2(hip, ankle, L1, L2, (1.0, sd * 0.12, 0.05))
            p['thigh' + s].matrix_basis = seg_mx(hip, knee, 0.072 * b, 0.072 * b)
            p['knee' + s].matrix_basis = sph_mx(knee, 0.062 * b)
            p['shin' + s].matrix_basis = seg_mx(knee, ank, 0.048 * b ** 0.5, 0.048 * b ** 0.5)
            foot = math.radians(-12 + 8 * math.sin(ang))
            p['shoe' + s].matrix_basis = box_mx(ankle + Vector((0.075, 0, -0.045)), 0.27, 0.095, 0.085, Matrix.Rotation(-foot, 3, 'Y'))
        # arms
        A1 = 0.30 * b ** 0.5; A2 = 0.30 * b ** 0.5
        for s, sd in (('R', -1), ('L', 1)):
            sho = S + Vector((0.0, sd * sh_w, -0.02))
            hand = bikeM @ Vector((0.605, sd * 0.195, 0.965 - 0.0))
            if up > 0.5:
                hand = bikeM @ Vector((0.53, sd * 0.2, 1.0))
            el, hd = ik2(sho, hand, A1, A2, (-0.2, sd * 1.0, -0.2))
            p['shoulder' + s].matrix_basis = sph_mx(sho, 0.058 * b)
            p['uarm' + s].matrix_basis = seg_mx(sho, el, 0.046 * b, 0.046 * b)
            p['elbow' + s].matrix_basis = sph_mx(el, 0.042 * b)
            p['farm' + s].matrix_basis = seg_mx(el, hd, 0.036 * b ** 0.5, 0.036 * b ** 0.5)
            p['hand' + s].matrix_basis = sph_mx(hd, 0.043)
        if 'pack' in p:
            c = H + (S - H) * 0.55 + Vector((-0.17, 0, 0.06))
            rot = Matrix.Rotation(math.radians(90) - lean, 3, 'Y')
            p['pack'].matrix_basis = box_mx(c, 0.17, 0.30, 0.38, rot)
            p['packlid'].matrix_basis = box_mx(c + Vector((-0.012, 0, 0)), 0.178, 0.305, 0.06, rot)
        # lights (strength is on the shared material; here we just toggle visibility via scale)
        on = 1.0 if lights > 0.01 else 0.0
        p['tail'].matrix_basis = box_mx(bikeM @ Vector((-0.235, 0, 0.95)), 0.04 * on, 0.04 * on, 0.04 * on) if on else box_mx((0, 0, -50), 0.001, 0.001, 0.001)
        p['head_l'].matrix_basis = box_mx(bikeM @ Vector((0.64, 0, 0.80)), 0.04 * on, 0.05 * on, 0.04 * on) if on else box_mx((0, 0, -50), 0.001, 0.001, 0.001)

    def place(self, x, y, z, yaw, pitch=0.0, roll=0.0):
        """World placement of the rider root; the bike's own lean (roll) is applied inside pose()."""
        M = Matrix.Translation((x, y, z)) @ Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Rotation(-pitch, 4, 'Y') @ Matrix.Rotation(roll, 4, 'X')
        self.root.matrix_world = M
