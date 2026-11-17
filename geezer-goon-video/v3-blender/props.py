"""Set pieces. Every prop lives in road coordinates (s along the route, lat to the left of the centreline)."""
import math
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix
import common as C
import world as W
from bl_util import hexc, add_obj, main_material, glow_material, bm_to_mesh
import riders as R
from riders import seg_mx, sph_mx, box_mx, shared


def road_frame(s, lat=0.0, dz=0.0):
    x, y, z, tx, ty = W.get_road().at(s)
    return Vector((x - ty * lat, y + tx * lat, z + dz)), Vector((tx, ty, 0)), Vector((-ty, tx, 0))


def frame_matrix(s, lat=0.0, dz=0.0, yaw_off=0.0):
    """World matrix at (s, lat): x along the road, y left, z up, following the road slope."""
    x, y, z, tx, ty = W.get_road().at(s)
    z0 = W.get_road().at(s - 1.5)[2]; z1 = W.get_road().at(s + 1.5)[2]
    pitch = math.atan2(z1 - z0, 3.0)
    yaw = math.atan2(ty, tx) + yaw_off
    return Matrix.Translation((x - ty * lat, y + tx * lat, z + dz)) @ Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Rotation(-pitch, 4, 'Y')


def add_box(bm, cl, center, size, color, yaw=0.0):
    res = bmesh.ops.create_cube(bm, size=1.0)
    M = Matrix.Translation(center) @ Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Diagonal((size[0], size[1], size[2], 1))
    bmesh.ops.transform(bm, matrix=M, verts=res['verts'])
    for v in res['verts']:
        for l in v.link_loops:
            l[cl] = color


def add_cyl(bm, cl, a, b, r, color, seg=8):
    d = Vector(b) - Vector(a)
    res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r, depth=d.length)
    z = d.normalized()
    up = Vector((0, 0, 1)) if abs(z.z) < 0.95 else Vector((1, 0, 0))
    x = up.cross(z).normalized(); y = z.cross(x)
    rot = Matrix(((x.x, y.x, z.x, 0), (x.y, y.y, z.y, 0), (x.z, y.z, z.z, 0), (0, 0, 0, 1)))
    bmesh.ops.transform(bm, matrix=Matrix.Translation((Vector(a) + Vector(b)) / 2) @ rot, verts=res['verts'])
    for v in res['verts']:
        for l in v.link_loops:
            l[cl] = color


def new_bm():
    bm = bmesh.new()
    return bm, bm.loops.layers.float_color.new('Col')


def finish(bm, name, coll, mat=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat or main_material())
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def set_hidden(objs, hidden):
    for o in objs:
        if o.hide_render != hidden:
            o.hide_render = hidden


# =================================================================================================== gate
class Gate:
    """Beach Dr half-gate: two striped arms leave a one-bike gap near the middle of the road."""
    GAP = (-2.5, -1.15)      # lateral extent of the gap (left of centre positive)

    def __init__(self, film, s, t0, t1, night=False):
        self.s, self.t0, self.t1 = s, t0, t1
        bm, cl = new_bm()
        red = hexc('#d63a2f'); white = hexc('#f2f2ee'); post = hexc('#e8e8e4'); amber = hexc('#ffb020')
        P0, T, N = road_frame(s)
        base = Matrix.Identity(4)
        def at(lat, along, z):
            return road_frame(s + along, lat, z)[0]
        # arms: left from +3.6 to GAP[1], right from -3.6 to GAP[0]
        for (l0, l1) in ((3.6, self.GAP[1]), (-3.6, self.GAP[0])):
            n = int(abs(l1 - l0) / 0.42) + 1
            for k in range(n):
                a = l0 + (l1 - l0) * k / n; b = l0 + (l1 - l0) * (k + 1) / n
                col = red if k % 2 == 0 else white
                for zz in (0.62, 0.98):
                    add_cyl(bm, cl, at(a, 0, zz), at(b, 0, zz), 0.055, col, 6)
            for lat in (l0, l1):
                add_cyl(bm, cl, at(lat, 0, 0.0), at(lat, 0, 1.25), 0.07, post, 6)
                add_cyl(bm, cl, at(lat, 0, 1.25), at(lat, 0, 1.33), 0.09, amber, 6)
        # warning sign boards either side
        for lat in (4.1, -4.1):
            add_cyl(bm, cl, at(lat, -0.6, 0), at(lat, -0.6, 1.9), 0.04, hexc('#555a62'), 6)
            add_box(bm, cl, at(lat, -0.62, 1.7), (0.05, 0.8, 0.55), hexc('#ff8a1c'), yaw=math.atan2(T.y, T.x))
            add_box(bm, cl, at(lat, -0.64, 1.7), (0.03, 0.62, 0.12), hexc('#1d1d1f'), yaw=math.atan2(T.y, T.x))
        self.ob = finish(bm, 'gate', film.coll)
        # blinking amber lamps (shared emission)
        self.objs = [self.ob]

    def update(self, t):
        set_hidden(self.objs, not (self.t0 - 1.0 <= t <= self.t1 + 1.0))


# =================================================================================================== pedestrians
class Person:
    """A walking cartoon person; local frame x forward, y left, z up (root placed by the owner)."""

    def __init__(self, film, spec, name):
        sh = shared()
        self.coll = film.coll
        self.spec = spec
        self.root = bpy.data.objects.new(name, None)
        film.coll.objects.link(self.root)
        self.parts = {}
        mat = main_material()
        def part(n, mesh, col):
            ob = bpy.data.objects.new(f'{name}_{n}', mesh)
            ob.color = col
            ob.parent = self.root
            film.coll.objects.link(ob)
            self.parts[n] = ob
        skin = hexc(spec.get('skin', '#e0ac86')); shirt = hexc(spec['shirt']); pants = hexc(spec.get('pants', '#2b3140'))
        part('torso', sh['cyl'], shirt); part('head', sh['sph'], skin); part('hair', sh['sph'], hexc(spec.get('hair', '#3a2a1a')))
        part('hips', sh['sph'], pants)
        for sd in ('L', 'R'):
            part('leg' + sd, sh['cyl6'], pants); part('foot' + sd, sh['box'], hexc('#f2f2f2'))
            part('arm' + sd, sh['cyl6'], shirt); part('hand' + sd, sh['sph_lo'], skin)
        if spec.get('hat'):
            part('hat', sh['sph'], hexc(spec['hat']))
        self.objs = [self.root] + list(self.parts.values())

    def pose(self, phase, hand_target=None, leash_hand=None, speed=1.0):
        p = self.parts
        sc = self.spec.get('scale', 1.0)
        walk = math.sin(phase)
        z_hip = 0.98 * sc
        bob = 0.015 * abs(math.cos(phase))
        hip = Vector((0, 0, z_hip + bob))
        sh_ = Vector((0.0, 0, 1.52 * sc + bob))
        p['hips'].matrix_basis = sph_mx(hip, 0.15 * sc, 0.18 * sc, 0.13 * sc)
        p['torso'].matrix_basis = seg_mx(hip, sh_, 0.13 * sc, 0.19 * sc)
        p['head'].matrix_basis = sph_mx(sh_ + Vector((0.01, 0, 0.19 * sc)), 0.11 * sc)
        p['hair'].matrix_basis = sph_mx(sh_ + Vector((-0.012, 0, 0.215 * sc)), 0.118 * sc, 0.118 * sc, 0.1 * sc)
        if 'hat' in p:
            p['hat'].matrix_basis = sph_mx(sh_ + Vector((0, 0, 0.255 * sc)), 0.135 * sc, 0.135 * sc, 0.07 * sc)
        for sd, side, ph in (('L', 1, 0.0), ('R', -1, math.pi)):
            w = math.sin(phase + ph)
            hp = hip + Vector((0, side * 0.1 * sc, 0))
            foot = Vector((0.32 * w * sc, side * 0.1 * sc, 0.06 * sc + 0.1 * max(0.0, math.cos(phase + ph)) * sc))
            p['leg' + sd].matrix_basis = seg_mx(hp, foot + Vector((0, 0, 0.05)), 0.075 * sc, 0.075 * sc)
            p['foot' + sd].matrix_basis = box_mx(foot + Vector((0.07, 0, 0.0)), 0.27 * sc, 0.1 * sc, 0.09 * sc)
            shp = sh_ + Vector((0, side * 0.22 * sc, -0.03))
            if hand_target is not None and sd == 'R' and False:
                hand = hand_target
            elif leash_hand is not None and sd == 'R':
                hand = leash_hand
            elif hand_target is not None:
                hand = hand_target + Vector((0, side * 0.2 * sc, 0))
            else:
                hand = shp + Vector((-0.30 * w * sc * (1 if side > 0 else 1), side * 0.05 * sc, -0.62 * sc))
                if side < 0:
                    hand = shp + Vector((0.30 * math.sin(phase) * sc, side * 0.05 * sc, -0.62 * sc))
                else:
                    hand = shp + Vector((-0.30 * math.sin(phase) * sc, side * 0.05 * sc, -0.62 * sc))
            p['arm' + sd].matrix_basis = seg_mx(shp, hand, 0.05 * sc, 0.05 * sc)
            p['hand' + sd].matrix_basis = sph_mx(hand, 0.055 * sc)

    def place(self, M):
        self.root.matrix_world = M


class Stroller:
    def __init__(self, film, name, col='#6a7fd1'):
        sh = shared()
        self.root = bpy.data.objects.new(name, None); film.coll.objects.link(self.root)
        self.parts = {}
        def part(n, mesh, color):
            ob = bpy.data.objects.new(f'{name}_{n}', mesh); ob.color = hexc(color) if isinstance(color, str) else color
            ob.parent = self.root; film.coll.objects.link(ob); self.parts[n] = ob
        part('seat', sh['box'], col); part('canopy', sh['sph'], col); part('baby', sh['sph'], '#f2c9a8'); part('hood', sh['sph_lo'], '#ffd23f')
        part('handle', sh['cyl6'], '#2b2e36')
        part('wF', sh['cyl6'], '#1b1c20'); part('wR', sh['cyl6'], '#1b1c20'); part('frame', sh['cyl6'], '#8a8f98')
        self.objs = [self.root] + list(self.parts.values())
        self.pose(0.0)

    def pose(self, spin, handle_height=0.98):
        p = self.parts
        p['seat'].matrix_basis = box_mx((0.18, 0, 0.62), 0.5, 0.42, 0.22)
        p['canopy'].matrix_basis = sph_mx((0.12, 0, 0.78), 0.3, 0.26, 0.2)
        p['baby'].matrix_basis = sph_mx((0.3, 0, 0.78), 0.1)
        p['hood'].matrix_basis = sph_mx((0.3, 0, 0.88), 0.065, 0.065, 0.055)
        p['handle'].matrix_basis = seg_mx((-0.2, -0.22, handle_height), (-0.2, 0.22, handle_height), 0.022, 0.022)
        p['frame'].matrix_basis = seg_mx((-0.2, 0, handle_height), (0.2, 0, 0.5), 0.018, 0.018)
        for n, x in (('wF', 0.42), ('wR', -0.12)):
            p[n].matrix_basis = Matrix.Translation((x, 0, 0.17)) @ Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Diagonal((0.17, 0.17, 0.44, 1))


class Dog:
    def __init__(self, film, name='dog'):
        sh = shared()
        self.root = bpy.data.objects.new(name, None); film.coll.objects.link(self.root)
        self.parts = {}
        def part(n, mesh, color):
            ob = bpy.data.objects.new(f'{name}_{n}', mesh); ob.color = hexc(color)
            ob.parent = self.root; film.coll.objects.link(ob); self.parts[n] = ob
        part('body', sh['sph'], '#c98a4b'); part('chest', sh['sph'], '#d9a064'); part('head', sh['sph'], '#c98a4b'); part('snout', sh['sph_lo'], '#e6c19a')
        part('nose', sh['sph_lo'], '#1b1b1b'); part('earL', sh['sph_lo'], '#8a5a2b'); part('earR', sh['sph_lo'], '#8a5a2b')
        part('tail', sh['cyl6'], '#c98a4b'); part('collar', sh['cyl6'], '#d63a2f')
        for l in ('FL', 'FR', 'BL', 'BR'):
            part('leg' + l, sh['cyl6'], '#b87a3c')
        self.objs = [self.root] + list(self.parts.values())

    def pose(self, phase, wag):
        p = self.parts
        h = 0.52
        p['body'].matrix_basis = sph_mx((0, 0, h), 0.42, 0.2, 0.2)
        p['chest'].matrix_basis = sph_mx((0.22, 0, h + 0.02), 0.22, 0.19, 0.21)
        hd = Vector((0.55, 0, h + 0.2))
        p['head'].matrix_basis = sph_mx(hd, 0.14, 0.12, 0.12)
        p['snout'].matrix_basis = sph_mx(hd + Vector((0.14, 0, -0.03)), 0.1, 0.07, 0.065)
        p['nose'].matrix_basis = sph_mx(hd + Vector((0.23, 0, -0.01)), 0.03)
        p['earL'].matrix_basis = sph_mx(hd + Vector((-0.02, 0.1, 0.0)), 0.05, 0.03, 0.09)
        p['earR'].matrix_basis = sph_mx(hd + Vector((-0.02, -0.1, 0.0)), 0.05, 0.03, 0.09)
        p['collar'].matrix_basis = seg_mx(hd + Vector((-0.1, 0, -0.1)), hd + Vector((-0.05, 0, -0.17)), 0.09, 0.09)
        tb = Vector((-0.4, 0, h + 0.1))
        p['tail'].matrix_basis = seg_mx(tb, tb + Vector((-0.2, 0.2 * math.sin(wag), 0.18)), 0.03, 0.03)
        for l, (x, sd, ph) in {'FL': (0.26, 1, 0), 'FR': (0.26, -1, math.pi), 'BL': (-0.28, 1, math.pi), 'BR': (-0.28, -1, 0)}.items():
            sw = 0.16 * math.sin(phase + ph)
            hip = Vector((x, sd * 0.12, h - 0.06)); foot = Vector((x + sw, sd * 0.12, 0.03))
            p['leg' + l].matrix_basis = seg_mx(hip, foot, 0.045, 0.045)


class Pedestrians:
    """Set dressing on the sidewalk: strollers (gate scenes) or walkers + dog on a 12-foot lead (verse 2)."""

    def __init__(self, film, kind, s, t0, t1, side=-1):
        self.kind, self.s, self.t0, self.t1, self.side = kind, s, t0, t1, side
        self.film = film
        self.people, self.strollers, self.dogs = [], [], []
        self.leash = None
        if kind == 'strollers':
            specs = [dict(shirt='#e9546b', pants='#2d3b55', hair='#2a1b10'), dict(shirt='#3fa7d6', pants='#2a2d34', hair='#a8742c', hat='#ffd23f')]
            for i, sp in enumerate(specs):
                self.people.append(Person(film, sp, f'ped{kind}{i}{s:.0f}'))
                self.strollers.append(Stroller(film, f'str{i}{s:.0f}', ['#6a7fd1', '#f4a261'][i]))
        elif kind == 'walkers':
            specs = [dict(shirt='#7a5fd1', pants='#3a3f4a', hair='#1b1b1b'), dict(shirt='#2ec4b6', pants='#4a4033', hair='#c9a05a', hat='#e63946')]
            for i, sp in enumerate(specs):
                self.people.append(Person(film, sp, f'ped{kind}{i}{s:.0f}'))
            self.dogs.append(Dog(film, f'dog{s:.0f}'))
            bm, cl = new_bm()
            add_cyl(bm, cl, (0, 0, 0), (0, 0, 1), 1.0, hexc('#d63a2f'), 4)
            self.leash = bpy.data.objects.new('leash', bpy.data.meshes.new('leash'))
            bm2, cl2 = new_bm(); add_cyl(bm2, cl2, (0, 0, -0.5), (0, 0, 0.5), 1.0, hexc('#d63a2f'), 4)
            me = bpy.data.meshes.new('leashm'); bm2.to_mesh(me); bm2.free(); me.materials.append(main_material())
            self.leash.data = me; film.coll.objects.link(self.leash); bm.free()
        elif kind == 'scatter':
            for i in range(5):
                sp = dict(shirt=['#e76f51', '#2a9d8f', '#e9c46a', '#8d6cd9', '#4ea8de'][i], pants='#303642', hair=['#1b1b1b', '#6b4423', '#c9a05a', '#2a1b10', '#444'][i], hat='#ff7a59' if i == 2 else None)
                self.people.append(Person(film, sp, f'ped{kind}{i}{s:.0f}'))
            self.strollers.append(Stroller(film, f'strS{s:.0f}', '#6a7fd1'))
        self.objs = [o for p in self.people for o in p.objs] + [o for p in self.strollers for o in p.objs] + [o for d in self.dogs for o in d.objs] + ([self.leash] if self.leash else [])

    def update(self, t):
        vis = self.t0 - 1.0 <= t <= self.t1 + 1.0
        set_hidden(self.objs, not vis)
        if not vis:
            return
        ph = t * 5.2
        s = self.s
        if self.kind == 'strollers':
            # two parents with strollers walking beside the gate, slowly
            for i, (p, st) in enumerate(zip(self.people, self.strollers)):
                sc = s + 2.0 + i * 3.4 + 1.3 * (t - self.t0)
                M = frame_matrix(sc, self.side * (4.6 + 0.4 * i), 0.12 + 0.0, 0.0)
                p.place(M); p.pose(ph + i * 1.7, hand_target=Vector((0.58, 0, 0.98)))
                st.root.matrix_world = M @ Matrix.Translation((0.82, 0, 0.0)); st.pose(ph)
        elif self.kind == 'walkers':
            M0 = frame_matrix(s + 1.1 * (t - self.t0), self.side * 4.7, 0.12)
            for i, p in enumerate(self.people):
                Mi = frame_matrix(s + 1.1 * (t - self.t0) - 0.2 * i, self.side * (4.35 + 0.75 * i), 0.12)
                p.place(Mi)
            # the dog is at the end of 3.6 m of lead, out in the bike lane, sniffing (stationary mostly)
            hand = frame_matrix(s + 1.1 * (t - self.t0), self.side * 4.35, 0.12) @ Vector((0.0, 0.0, 0.0))
            for i, p in enumerate(self.people):
                p.pose(ph + 1.2 * i, leash_hand=(Vector((0.1, -self.side * 0.25, 0.78)) if i == 0 else None))
            # dog position: lead pulled taut to the left across the lane
            dog_s = s + 1.1 * (t - self.t0) + 0.8 * math.sin(t * 1.3)
            dog_lat = self.side * 4.35 - self.side * 3.3
            Md = frame_matrix(dog_s, dog_lat, 0.0, yaw_off=math.radians(70 * (-self.side)))
            self.dogs[0].root.matrix_world = Md
            self.dogs[0].pose(t * 4.0 if abs(math.sin(t * 1.3)) > 0.3 else 0.0, t * 14)
            # lead: from the first walker's hand to the dog's collar
            a = (frame_matrix(s + 1.1 * (t - self.t0), self.side * 4.35, 0.12) @ Vector((0.1, -self.side * 0.25, 0.78)))
            b = Md @ Vector((0.5, 0, 0.45))
            self.leash.matrix_world = seg_mx(a, b, 0.012, 0.012)
        elif self.kind == 'scatter':
            for i, p in enumerate(self.people):
                lat = self.side * (4.7 + 0.5 * i) * (1.0 if i % 2 == 0 else -1.0)
                sc = s + 4 + 3.1 * i + 0.8 * (t - self.t0)
                p.place(frame_matrix(sc, lat, 0.12, yaw_off=(math.pi if i % 2 else 0.0)))
                p.pose(ph + i * 1.3)
            st = self.strollers[0]
            st.root.matrix_world = frame_matrix(s + 8.5 + 0.8 * (t - self.t0), self.side * 5.1, 0.12) @ Matrix.Translation((0.82, 0, 0)); st.pose(ph)


# =================================================================================================== traffic light
class TrafficLight:
    def __init__(self, film, s, states, t0, t1):
        """states: [(t_from, 'green'|'amber'|'red'), ...]"""
        self.s, self.states, self.t0, self.t1 = s, states, t0, t1
        bm, cl = new_bm()
        grey = hexc('#3a3f47'); dark = hexc('#17181b')
        def at(lat, along, z): return road_frame(s + along, lat, z)[0]
        add_cyl(bm, cl, at(-4.4, 0, 0), at(-4.4, 0, 6.2), 0.12, grey, 8)
        add_cyl(bm, cl, at(-4.4, 0, 6.1), at(-0.9, 0, 6.1), 0.08, grey, 8)
        T = road_frame(s)[1]
        yaw = math.atan2(T.y, T.x)
        add_box(bm, cl, at(-1.8, 0, 5.55), (0.34, 0.42, 1.15), dark, yaw=yaw)
        add_box(bm, cl, at(-1.8, 0.05, 5.55), (0.4, 0.5, 1.22), hexc('#0e0f11'), yaw=yaw)
        # stop line + pedestrian crossing
        for k in range(8):
            add_box(bm, cl, at(-3.2 + k * 0.78, -5.5, 0.03), (0.6, 0.42, 0.02), hexc('#f2f2ee'), yaw=yaw)
        add_box(bm, cl, at(-1.8, -8.2, 0.03), (0.3, 7.0, 0.02), hexc('#f2f2ee'), yaw=yaw)
        self.ob = finish(bm, 'tl_body', film.coll)
        self.lamps = {}
        sh = shared()
        for name, z, col in (('red', 5.9, '#ff3b30'), ('amber', 5.55, '#ffb020'), ('green', 5.2, '#2fe06e')):
            ob = bpy.data.objects.new(f'tl_{name}_{s:.0f}', sh['sph_lo'])
            M = Matrix.Translation(road_frame(s - 0.22, -1.8, z)[0]) @ Matrix.Diagonal((0.13, 0.13, 0.13, 1))
            ob.matrix_world = M
            mat = bpy.data.materials.new(f'M_tl_{name}_{s:.0f}'); mat.use_nodes = True
            nt = mat.node_tree; nt.nodes.clear()
            out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
            em.inputs['Color'].default_value = hexc(col); em.inputs['Strength'].default_value = 0.0
            nt.links.new(em.outputs[0], out.inputs[0])
            ob.data = sh['sph_lo'].copy(); ob.data.materials.clear(); ob.data.materials.append(mat)
            film.coll.objects.link(ob)
            self.lamps[name] = (ob, em)
        self.objs = [self.ob] + [v[0] for v in self.lamps.values()]

    def state(self, t):
        cur = self.states[0][1]
        for tf, st in self.states:
            if t >= tf:
                cur = st
        return cur

    def update(self, t):
        vis = self.t0 - 1.0 <= t <= self.t1 + 1.0
        set_hidden(self.objs, not vis)
        if not vis:
            return
        st = self.state(t)
        for name, (ob, em) in self.lamps.items():
            em.inputs['Strength'].default_value = 14.0 if name == st else 0.15


# =================================================================================================== potholes
class Potholes:
    def __init__(self, film, s, t0, t1, n=16, seed=3):
        self.s, self.t0, self.t1 = s, t0, t1
        rng = np.random.default_rng(seed)
        bm, cl = new_bm()
        dark = hexc('#0d0e10'); crack = hexc('#1d1e23'); patch = hexc('#767a84'); rim = hexc('#5b5e66'); chalk = hexc('#ff7a1a')
        self.holes = []
        rd = W.get_road()
        for i in range(n):
            ds = rng.uniform(4, 70); lat = rng.uniform(-3.2, 0.6)
            r = rng.uniform(0.22, 0.62)
            self.holes.append((ds, lat, r))
            seg = 10
            ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
            rr = r * (0.75 + 0.5 * rng.random(seg))
            c0 = road_frame(s + ds, lat, 0.03)[0]
            verts = []
            for a, q in zip(ang, rr):
                p, T, N = road_frame(s + ds + q * math.cos(a) * 1.3, lat + q * math.sin(a) * 0.9, 0.035)
                verts.append(bm.verts.new(p))
            ctr = bm.verts.new(c0 + Vector((0, 0, -0.01)))
            for k in range(seg):
                f = bm.faces.new((verts[k], verts[(k + 1) % seg], ctr))
                for l in f.loops: l[cl] = dark
            # lighter rim ring
            verts2 = []
            for a, q in zip(ang, rr):
                p, T, N = road_frame(s + ds + (q + 0.1) * math.cos(a) * 1.3, lat + (q + 0.1) * math.sin(a) * 0.9, 0.03)
                verts2.append(bm.verts.new(p))
            for k in range(seg):
                f = bm.faces.new((verts2[k], verts2[(k + 1) % seg], verts[(k + 1) % seg], verts[k]))
                for l in f.loops: l[cl] = rim
            # spray-paint ring the road crew leaves around a pothole
            ring_in, ring_out = [], []
            for a, q in zip(ang, rr):
                ring_in.append(bm.verts.new(road_frame(s + ds + (q + 0.22) * math.cos(a) * 1.3, lat + (q + 0.22) * math.sin(a) * 0.9, 0.036)[0]))
                ring_out.append(bm.verts.new(road_frame(s + ds + (q + 0.31) * math.cos(a) * 1.3, lat + (q + 0.31) * math.sin(a) * 0.9, 0.036)[0]))
            for k in range(seg):
                f = bm.faces.new((ring_in[k], ring_in[(k + 1) % seg], ring_out[(k + 1) % seg], ring_out[k]))
                for l in f.loops: l[cl] = chalk
        # cracks: long thin quads
        for i in range(18):
            ds0 = rng.uniform(0, 70); lat0 = rng.uniform(-3.4, 0.8)
            pts = []
            for j in range(7):
                ds0 += rng.uniform(1.0, 3.0); lat0 += rng.normal(0, 0.18)
                pts.append((ds0, lat0))
            for j in range(len(pts) - 1):
                a = road_frame(s + pts[j][0], pts[j][1], 0.032)[0]; b = road_frame(s + pts[j + 1][0], pts[j + 1][1], 0.032)[0]
                d = (b - a); nrm = Vector((-d.y, d.x, 0)).normalized() * 0.04
                v = [bm.verts.new(a + nrm), bm.verts.new(b + nrm), bm.verts.new(b - nrm), bm.verts.new(a - nrm)]
                f = bm.faces.new(v)
                for l in f.loops: l[cl] = crack
        # patched rectangles
        for i in range(5):
            ds = rng.uniform(5, 65); lat = rng.uniform(-3.0, 0.4)
            w, h = rng.uniform(0.8, 1.8), rng.uniform(0.6, 1.2)
            vs = [bm.verts.new(road_frame(s + ds + dx, lat + dy, 0.04)[0]) for dx, dy in ((-w, -h), (-w, h), (w, h), (w, -h))]
            f = bm.faces.new(vs)
            for l in f.loops: l[cl] = patch
        self.ob = finish(bm, 'potholes', film.coll)
        self.objs = [self.ob]

    def update(self, t):
        set_hidden(self.objs, not (self.t0 - 1.0 <= t <= self.t1 + 1.0))


# =================================================================================================== signs & floating items
def text_obj(film, body, size, color, name='txt', font=None, extrude=0.0, emit=0.0, align='CENTER'):
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = body; cu.size = size; cu.align_x = align; cu.align_y = 'CENTER'; cu.extrude = extrude
    if font:
        cu.font = font
    ob = bpy.data.objects.new(name, cu)
    film.coll.objects.link(ob)
    mat = bpy.data.materials.new('M_' + name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    if emit > 0:
        em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = hexc(color) if isinstance(color, str) else color; em.inputs['Strength'].default_value = emit
        nt.links.new(em.outputs[0], out.inputs[0])
    else:
        b = nt.nodes.new('ShaderNodeBsdfPrincipled'); b.inputs['Base Color'].default_value = hexc(color) if isinstance(color, str) else color; b.inputs['Roughness'].default_value = 0.6
        nt.links.new(b.outputs[0], out.inputs[0])
    cu.materials.append(mat)
    return ob


class HillSign:
    def __init__(self, film, s, t0, t1, font=None):
        self.t0, self.t1 = s and t0, t1
        bm, cl = new_bm()
        def at(lat, along, z): return road_frame(s + along, lat, z)[0]
        T = road_frame(s)[1]; yaw = math.atan2(T.y, T.x)
        add_cyl(bm, cl, at(-4.8, 0, 0), at(-4.8, 0, 2.6), 0.045, hexc('#5a6068'), 6)
        add_box(bm, cl, at(-4.8, -0.06, 2.2), (0.06, 1.0, 1.0), hexc('#ffd23f'), yaw=yaw)
        add_box(bm, cl, at(-4.8, -0.08, 2.2), (0.04, 0.88, 0.88), hexc('#ffe27a'), yaw=yaw)
        self.ob = finish(bm, 'hillsign', film.coll)
        self.txt = text_obj(film, '7%', 0.62, '#17171a', 'hill7', font)
        M = frame_matrix(s - 0.12, -4.8, 2.2) @ Matrix.Rotation(-math.pi / 2, 4, 'Y') @ Matrix.Rotation(-math.pi / 2, 4, 'Z') @ Matrix.Rotation(math.pi, 4, 'Z') @ Matrix.Rotation(math.pi, 4, 'X')
        self.txt.matrix_world = frame_matrix(s - 0.12, -4.8, 2.2) @ Matrix(((0, 0, -1, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
        self.txt.data.size = 0.55
        self.objs = [self.ob, self.txt]

    def update(self, t):
        set_hidden(self.objs, not (self.t0 - 1.0 <= t <= self.t1 + 1.0))


class FloatingItems:
    """laptop, badge, shoes pop out of Angelo's backpack (117.7 - 120.9)."""

    def __init__(self, film, t0, t1):
        self.t0, self.t1 = t0, t1
        self.film = film
        sh = shared()
        self.items = []
        # laptop
        bm, cl = new_bm()
        add_box(bm, cl, (0, 0, 0), (0.34, 0.24, 0.02), hexc('#9aa0a8'))
        add_box(bm, cl, (-0.12, 0, 0.12), (0.02, 0.24, 0.24), hexc('#9aa0a8'))
        add_box(bm, cl, (-0.115, 0, 0.12), (0.01, 0.2, 0.2), hexc('#8ec5ff'))
        lap = finish(bm, 'laptop', film.coll)
        # badge on a lanyard
        bm, cl = new_bm()
        add_box(bm, cl, (0, 0, 0), (0.07, 0.2, 0.3), hexc('#f7f7f2'))
        add_box(bm, cl, (0.01, 0, 0.07), (0.07, 0.17, 0.09), hexc('#1d6fb8'))
        add_box(bm, cl, (0.01, 0, -0.02), (0.07, 0.12, 0.05), hexc('#c9ccd2'))
        add_box(bm, cl, (0, 0, 0.19), (0.05, 0.06, 0.08), hexc('#2b2f3a'))
        for sd in (-1, 1):
            add_cyl(bm, cl, (0, sd * 0.03, 0.22), (0, sd * 0.2, 0.55), 0.01, hexc('#1d6fb8'), 4)
        badge = finish(bm, 'badge', film.coll)
        # shoes
        bm, cl = new_bm()
        for sd in (-1, 1):
            add_box(bm, cl, (0, sd * 0.1, 0), (0.28, 0.1, 0.07), hexc('#e9e9e4'))
            add_box(bm, cl, (0.09, sd * 0.1, 0.05), (0.12, 0.09, 0.07), hexc('#d63a2f'))
            add_box(bm, cl, (-0.1, sd * 0.1, 0.06), (0.08, 0.09, 0.1), hexc('#f2f2ee'))
        shoes = finish(bm, 'shoes', film.coll)
        self.items = [lap, badge, shoes]
        self.objs = list(self.items)

    def update(self, t, angelo_pos, cam_pos):
        vis = self.t0 <= t <= self.t1
        set_hidden(self.objs, not vis)
        if not vis:
            return
        for i, ob in enumerate(self.items):
            a = (t - self.t0 - 0.35 * i)
            k = max(0.0, min(1.0, a / 0.35))
            pop = 1 + 0.35 * math.sin(k * math.pi) if k < 1 else 1.0
            sc = k * pop
            out = 1 - max(0.0, min(1.0, (t - (self.t1 - 0.35)) / 0.35))
            sc *= out
            # float up and to the side of Angelo as seen from the camera
            d = (cam_pos - angelo_pos); d.z = 0; d.normalize()
            right = Vector((-d.y, d.x, 0))
            pos = angelo_pos + Vector((0, 0, 1.55 + 0.42 * i)) + right * (-0.75 + 0.62 * i) + Vector((0, 0, 0.05 * math.sin(t * 3 + i)))
            rot = Matrix.Rotation(t * 1.6 + i * 2.1, 4, 'Z')
            ob.matrix_world = Matrix.Translation(pos) @ rot @ Matrix.Diagonal((sc * 1.7, sc * 1.7, sc * 1.7, 1))
