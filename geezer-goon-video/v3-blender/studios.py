"""Bike computer on the handlebars, calendar sets, route-map glow, big 3D titles."""
import math, os
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix
import common as C
import world as W
from bl_util import hexc, main_material, glow_material, mesh_from_arrays
import props as P
from props import text_obj, new_bm, add_box, add_cyl, finish, set_hidden, road_frame, frame_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_ANTON = os.path.join(HERE, 'fonts', 'anton.ttf')
FONT_BEBAS = os.path.join(HERE, 'fonts', 'bebasneue.ttf')


def load_font(path):
    try:
        return bpy.data.fonts.load(path)
    except Exception:
        return None


def clock_str(t):
    sec = -45 + t * 4.7 if t < 9.5 else 0
    return None


# =================================================================================================== bike computer
class BikeComputer:
    """A Wahoo-ish head unit stuck on a rider's stem. Text is real geometry; the screen is emissive."""

    def __init__(self, film, rider_id, t_ranges, fonts):
        self.film, self.rid, self.ranges = film, rider_id, t_ranges
        root = film.riders[rider_id].root
        self.root = bpy.data.objects.new('computer_' + rider_id, None)
        film.coll.objects.link(self.root)
        self.root.parent = root
        # sits just ahead of the stem, tilted back toward the rider
        self.root.matrix_basis = Matrix.Translation((0.60, 0.0, 1.045)) @ Matrix.Rotation(math.radians(-24), 4, 'Y')
        # body + bezel + screen (rider space: x forward, y left, z up). The unit stands up facing the rider (-x).
        bm, cl = new_bm()
        add_box(bm, cl, (0, 0, 0), (0.016, 0.098, 0.066), hexc('#121317'))
        add_box(bm, cl, (0, 0, -0.045), (0.02, 0.03, 0.03), hexc('#2b2e36'))
        add_cyl(bm, cl, (0.0, 0, -0.062), (0.0, 0, -0.12), 0.011, hexc('#2b2e36'), 6)
        body = finish(bm, 'cmp_body', film.coll)
        body.parent = self.root
        # screen
        bm, cl = new_bm()
        add_box(bm, cl, (-0.0085, 0, 0.0), (0.002, 0.088, 0.056), hexc('#05080c'))
        scr = finish(bm, 'cmp_screen', film.coll, mat=self._screen_mat())
        scr.parent = self.root
        self.texts = {}
        def txt(name, body, size, col, x, z, align='CENTER'):
            ob = text_obj(film, body, size, col, 'cmp_' + name, fonts.get('bebas'), emit=1.4, align=align)
            ob.parent = self.root
            ob.matrix_basis = Matrix.Translation((-0.0105, x, z)) @ Matrix(((0, 0, -1, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
            self.texts[name] = ob
        txt('big', '', 0.034, '#ffd23f', 0.0, 0.004)
        txt('label', '', 0.0095, '#8fe3ff', 0.0, 0.0215)
        txt('sub', '', 0.0105, '#ffffff', 0.0, -0.0205)
        self.objs = [self.root, body, scr] + list(self.texts.values())
        self.body, self.scr = body, scr

    def _screen_mat(self):
        m = bpy.data.materials.new('M_cmp_screen')
        m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
        em.inputs['Color'].default_value = hexc('#0b1626'); em.inputs['Strength'].default_value = 0.7
        nt.links.new(em.outputs[0], out.inputs[0])
        return m

    def set_text(self, **kw):
        for k, v in kw.items():
            if k in self.texts and self.texts[k].data.body != v:
                self.texts[k].data.body = v
            if k == 'big' and k in self.texts:
                n = max(1, len(v))                     # long strings shrink to stay inside the screen
                sc = 1.0 if n <= 6 else max(0.34, 6.2 / n)
                self.texts[k].scale = (sc, sc, sc)

    def update(self, t):
        vis = any(a - 0.2 <= t <= b + 0.2 for a, b in self.ranges)
        set_hidden(self.objs, not vis)
        if not vis:
            return
        a, b = self.ranges[0]
        if t < 9.5:         # boot
            k = t / 9.5
            if t < 1.6:
                self.set_text(big='ROAM', label='', sub='')
            elif t < 4.0:
                self.set_text(big='GEEZER GOON', label='WEDNESDAY', sub='LOADING ROUTE ' + '.' * (1 + int(t * 3) % 3))
            else:
                n = int(38 * min(1, (t - 4.0) / 4.5))
                self.set_text(big='6:15', label='PM  ·  KENSINGTON MD', sub=f'{n} / 38 RIDERS')
        elif 166.0 <= t < 170.0:
            mph = 22 + 4.6 * min(1.0, (t - 166.4) / 2.5) + 0.35 * math.sin(t * 7)
            self.set_text(big=f'{mph:0.0f}', label='MPH', sub='KNOWLES AVE')
        elif t >= 213.5:
            k = min(1.0, max(0.0, (t - 214.07) / 2.0))
            sec = int(77 * 60 * k)
            self.set_text(big=f'1:{(sec // 60) % 60:02d}:{sec % 60:02d}' if sec >= 3600 else f'{sec // 60}:{sec % 60:02d}', label='ELAPSED', sub='SEVENTY-SEVEN MINUTES')


# =================================================================================================== calendar set
class Calendar:
    """A wall of weeks floating in a dark studio high above the neighbourhood. Wednesdays glow."""
    ORIGIN = Vector((0.0, 0.0, 3000.0))

    def __init__(self, film, mode, t0, t1, fonts):
        self.film, self.mode, self.t0, self.t1 = film, mode, t0, t1
        self.fonts = fonts
        o = self.ORIGIN + Vector((0, 40.0 if mode == 'weeks' else 90.0, 0))
        self.o = o
        bm, cl = new_bm()
        add_box(bm, cl, (o.x, o.y + 6, o.z), (60, 0.5, 40), hexc('#101830'))     # backdrop wall
        self.wall = finish(bm, 'cal_wall_' + mode, film.coll)
        self.tiles = []
        self.objs = [self.wall]
        days = ['MON', 'TUE', 'WED', 'THU', 'FRI']
        rows = 10 if mode == 'weeks' else 4
        for r in range(rows):
            for d in range(5):
                wed = d == 2
                base = '#2a3a6a' if not wed else '#ff7a29'
                if mode == 'tuethu':
                    base = '#2a3a6a' if d not in (1, 3) else '#2ec4b6'
                bm, cl = new_bm()
                add_box(bm, cl, (0, 0, 0), (1.5, 0.28, 1.2), hexc(base))
                ob = finish(bm, f'tile{r}{d}', film.coll)
                lab = text_obj(film, days[d] if r == 0 or True else '', 0.34, '#ffffff' if not wed else '#1b1b1b', f'tile_t{r}{d}', fonts.get('bebas'), emit=0.0)
                x = o.x + (d - 2) * 1.9; z = o.z + (rows / 2 - r) * 1.55 - 1.5
                ob.location = (x, o.y, z)
                lab.location = (x, o.y - 0.17, z)
                lab.rotation_euler = (math.pi / 2, 0, 0)
                self.tiles.append((ob, lab, r, d, wed))
                self.objs += [ob, lab]
        self.glow_mat = None
        lt = bpy.data.lights.new('calfill', 'AREA'); lt.energy = 2500.0; lt.size = 18.0; lt.color = (1.0, 0.95, 0.9)
        self.light = bpy.data.objects.new('calfill', lt); film.coll.objects.link(self.light)
        self.light.location = (o.x, o.y - 10.0, o.z + 7.0); self.light.rotation_euler = (math.radians(70), 0, 0)
        self.objs.append(self.light)

    def update(self, t):
        vis = self.t0 - 0.5 <= t <= self.t1 + 0.5
        set_hidden(self.objs, not vis)
        if not vis:
            return
        k = (t - self.t0) / max(1e-6, self.t1 - self.t0)
        for ob, lab, r, d, wed in self.tiles:
            if self.mode == 'weeks':
                pop = C.smooth((k * 1.4 - r * 0.07 - d * 0.015) / 0.18)
                lift = 0.5 * wed * C.smooth((k - 0.35 - r * 0.02) / 0.15)
            else:
                flip = 0.5 + 0.5 * math.sin(t * 5 + r * 2 + d)
                pop = 1.0
                lift = 0.35 * flip * (1.0 if d in (1, 3) else 0.0)
            ob.scale = (pop, pop, pop)
            ob.location.y = self.o.y - lift
            lab.scale = (pop, pop, pop)
            lab.location.y = self.o.y - lift - 0.17


# =================================================================================================== route glow (map scenes)
class RouteGlow:
    """The real route as a glowing ribbon over the neighbourhood; revealed by progress."""

    def __init__(self, film):
        rd = W.get_road()
        n = rd.n
        w = 14.0
        V = np.zeros((2 * n, 3), np.float32)
        V[0::2, 0] = rd.x - rd.nx * w / 2; V[0::2, 1] = rd.y - rd.ny * w / 2; V[0::2, 2] = rd.z + 0.8
        V[1::2, 0] = rd.x + rd.nx * w / 2; V[1::2, 1] = rd.y + rd.ny * w / 2; V[1::2, 2] = rd.z + 0.8
        q = np.arange(n - 1)
        F = np.concatenate([np.stack([2 * q, 2 * q + 1, 2 * q + 2], 1), np.stack([2 * q + 1, 2 * q + 3, 2 * q + 2], 1)]).astype(np.int32)
        prog = (np.arange(n) * W.SAMP / C.LEN).astype(np.float32)
        col = np.zeros((2 * n, 4), np.float32)
        col[:, 0] = np.repeat(prog, 2); col[:, 3] = 1
        me = mesh_from_arrays('routeglow', V, F, col)
        mat = bpy.data.materials.new('M_route')
        mat.use_nodes = True
        nt = mat.node_tree; nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission'); em.name = 'RouteEm'
        attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'Col'
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(attr.outputs['Color'], sep.inputs['Color'])
        # reveal: progress < head, brightness fades toward the tail
        head = nt.nodes.new('ShaderNodeValue'); head.name = 'Head'; head.outputs[0].default_value = 0.0
        tail = nt.nodes.new('ShaderNodeValue'); tail.name = 'Tail'; tail.outputs[0].default_value = 0.0
        sub1 = nt.nodes.new('ShaderNodeMath'); sub1.operation = 'SUBTRACT'
        nt.links.new(head.outputs[0], sub1.inputs[0]); nt.links.new(sep.outputs['Red'], sub1.inputs[1])
        gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.0
        nt.links.new(sub1.outputs[0], gt.inputs[0])
        sub2 = nt.nodes.new('ShaderNodeMath'); sub2.operation = 'SUBTRACT'
        nt.links.new(sep.outputs['Red'], sub2.inputs[0]); nt.links.new(tail.outputs[0], sub2.inputs[1])
        gt2 = nt.nodes.new('ShaderNodeMath'); gt2.operation = 'GREATER_THAN'; gt2.inputs[1].default_value = 0.0
        nt.links.new(sub2.outputs[0], gt2.inputs[0])
        both = nt.nodes.new('ShaderNodeMath'); both.operation = 'MULTIPLY'
        nt.links.new(gt.outputs[0], both.inputs[0]); nt.links.new(gt2.outputs[0], both.inputs[1])
        strength = nt.nodes.new('ShaderNodeMath'); strength.operation = 'MULTIPLY'; strength.inputs[1].default_value = 9.0
        nt.links.new(both.outputs[0], strength.inputs[0])
        em.inputs['Color'].default_value = hexc('#ffb02e')
        nt.links.new(strength.outputs[0], em.inputs['Strength'])
        nt.links.new(em.outputs[0], out.inputs[0])
        me.materials.append(mat)
        ob = bpy.data.objects.new('routeglow', me)
        film.coll.objects.link(ob)
        self.ob = ob; self.head = head; self.tail = tail
        self.t_ranges = []
        ob.hide_render = True

    def set(self, head, tail=0.0):
        self.head.outputs[0].default_value = head
        self.tail.outputs[0].default_value = tail



# =================================================================================================== 3D titles
class Title:
    def __init__(self, film, text, t0, t1, fonts, pos_fn, size=3.2, color='#ffd23f', sub=None):
        self.film, self.t0, self.t1, self.pos_fn = film, t0, t1, pos_fn
        self.ob = text_obj(film, text, size, color, 'title_' + text[:6], fonts.get('anton'), extrude=0.25, emit=0.0)
        self.objs = [self.ob]
        self.sub = None
        if sub:
            self.sub = text_obj(film, sub, size * 0.26, '#ffffff', 'titlesub_' + text[:4], fonts.get('bebas'), extrude=0.05, emit=0.0)
            self.objs.append(self.sub)

    def update(self, t):
        vis = self.t0 <= t <= self.t1
        set_hidden(self.objs, not vis)
        if not vis:
            return
        k = (t - self.t0)
        pop = min(1.0, k / 0.22)
        pop = 1 + 2.7 * (pop - 1) ** 3 + 1.9 * (pop - 1) ** 2 if k < 0.22 else 1.0
        out = 1.0 - C.smooth((t - (self.t1 - 0.7)) / 0.7)
        M = self.pos_fn(t)
        self.ob.matrix_world = M @ Matrix.Diagonal((pop * out, pop * out, pop * out, 1))
        if self.sub:
            self.sub.matrix_world = M @ Matrix.Translation((0, 0, -1.0)) @ Matrix.Diagonal((out, out, out, 1))
