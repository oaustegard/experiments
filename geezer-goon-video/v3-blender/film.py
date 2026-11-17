"""The film: pack model, camera rigs, per-frame baking (roots + camera are keyframed so Eevee can motion-blur them),
and a frame handler that poses limbs / lights / sky.

Time model. v2's odometer is a time-lapse (40 km in 232 s) so it can't be shown at real scale. Each shot is anchored at
the route position the lyric is about (odo at the shot's middle) and the pack then rides at its real Strava speed
through the shot. The compressed odometer is still what the HUD overlay reports.
"""
import math, os, random
import numpy as np
import bpy
from mathutils import Vector, Matrix, Euler, Quaternion
import common as C
import world as W
import riders as R
import scene_setup as S
import cast as K
from bl_util import hexc as hexc_
from bl_util import set_glow
from shots import SHOTS, PROPS, shot_at, mood_over, LM
import props as PR
import studios as ST

ROW = 2.25          # metres between rows of the pack
LANE_L = (-2.35, -1.2)   # lateral offsets of the two lanes (right lane of the road is negative)


def lerpv(v, k):
    """Scalar or [a, b] -> value at eased k."""
    if isinstance(v, (list, tuple)):
        return v[0] + (v[1] - v[0]) * k
    return v


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Film:
    def __init__(self, coll, n_far=32, frames=None):
        self.coll = coll
        self.rd = W.get_road()
        self.world_objs = W.build_world(coll, only=self._chunk_filter(frames))
        self.specs = [K.CAST[k] for k in K.NAMED] + K.far_specs(n_far)
        self.riders = {sp['id']: R.Rider(sp, coll) for sp in self.specs}
        self.ids = [sp['id'] for sp in self.specs]
        rng = random.Random(5)
        self.cad = {i: 78 + rng.uniform(-9, 9) for i in self.ids}
        self.ph = {i: rng.uniform(0, 2 * math.pi) for i in self.ids}
        self.sway = {i: (rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0.6, 1.4)) for i in self.ids}
        self.rhash = W.RoadHash(self.rd)
        self.cam = S.make_camera(coll)
        self.cam.rotation_mode = 'QUATERNION'
        self.cam.data.clip_start = 0.06; self.cam.data.clip_end = 20000.0
        self.sun, self.moon = S.add_sun(coll)
        S.setup_world()
        if os.environ.get('GG_NOCOMP') != '1':
            S.setup_compositor()
        for r in self.riders.values():
            r.root.rotation_mode = 'QUATERNION'
        self.fonts = {'anton': ST.load_font(ST.FONT_ANTON), 'bebas': ST.load_font(ST.FONT_BEBAS)}
        self.props = []
        self._build_props()
        self.cache = {}      # frame -> dict (screen positions etc.)
        self.markers = {}    # frame -> {rider: (px, py)}

    def _build_props(self):
        for p in PROPS:
            k = p['kind']
            if k == 'gate':
                self.props.append(PR.Gate(self, p['s'], p['t0'], p['t1']))
            elif k in ('strollers', 'walkers', 'scatter'):
                self.props.append(PR.Pedestrians(self, k, p['s'], p['t0'], p['t1']))
            elif k == 'light':
                self.props.append(PR.TrafficLight(self, p['s'], p['states'], p['t0'], p['t1']))
            elif k == 'potholes':
                self.props.append(PR.Potholes(self, p['s'], p['t0'], p['t1']))
            elif k == 'hillsign':
                self.props.append(PR.HillSign(self, p['s'], p['t0'], p['t1'], self.fonts.get('bebas')))
        comp = [(s['t0'], s['t1']) for s in SHOTS if s.get('computer')]
        if comp:
            self.computer = ST.BikeComputer(self, 'me', comp, self.fonts)
            self.props.append(self.computer)
        self.items = None
        its = [s for s in SHOTS if s.get('items')]
        if its:
            self.items = PR.FloatingItems(self, its[0]['t0'], its[0]['t1'])
        self.cal = [ST.Calendar(self, s['mode'], s['t0'], s['t1'], self.fonts) for s in SHOTS if s['kind'] == 'calendar']
        self.props += self.cal
        self.route = ST.RouteGlow(self)
        self.maps = [s for s in SHOTS if s['kind'] == 'map']
        self.title = ST.Title(self, 'GEEZER GOON', 26.5, 29.2, self.fonts, self.title_matrix, size=2.1,
                              sub='WEDNESDAY  ·  6:15 PM  ·  KENSINGTON MD')
        self.props.append(self.title)
        # night: warm pools of light that travel with the pack, plus a map table for the aerial scenes
        lt = bpy.data.lights.new('packlight', 'POINT'); lt.energy = 0.0; lt.shadow_soft_size = 1.5; lt.color = (1.0, 0.75, 0.5)
        self.packlight = bpy.data.objects.new('packlight', lt); self.coll.objects.link(self.packlight)
        lt2 = bpy.data.lights.new('packlight2', 'POINT'); lt2.energy = 0.0; lt2.shadow_soft_size = 2.0; lt2.color = (0.55, 0.7, 1.0)
        self.packlight2 = bpy.data.objects.new('packlight2', lt2); self.coll.objects.link(self.packlight2)
        bm, cl = PR.new_bm()
        PR.add_box(bm, cl, (-445, -327, -240), (24000, 24000, 4), hexc_('#0d1b2e'))
        self.table = PR.finish(bm, 'maptable', self.coll)
        self.table.hide_render = True

    def title_matrix(self, t):
        # 'GEEZER GOON' slams down between the camera and the pack at the start of verse 1; the text faces the camera ahead
        sh = shot_at(t)
        head = self.head_s(t, sh)
        return PR.frame_matrix(head + 7.0, -1.8, 2.3) @ Matrix.Rotation(math.pi / 2, 4, 'Z') @ Matrix.Rotation(math.pi / 2, 4, 'X')

    def _chunk_filter(self, frames, reach=900.0):
        """Which world chunks does this set of frames need? None = all (map shots show the whole route)."""
        if not frames:
            return None
        shots = []
        for fr in frames:
            sh = shot_at(fr / C.FPS)
            if sh not in shots:
                shots.append(sh)
        pts = []
        for sh in shots:
            if sh['kind'] == 'map':
                return None
            if sh['kind'] == 'calendar':
                continue
            s_list = [self.head_s(sh['t0'], sh), self.head_s(sh['t1'], sh), self.shot_anchor(sh)]
            ref = sh['cam'].get('ref', '')
            if isinstance(ref, str) and ref.startswith('fixed:'):
                s_list.append(float(ref.split(':')[1]))
            for s in s_list:
                x, y = self.rd.at(max(0.0, min(C.LEN, s)))[:2]
                pts.append((x, y))
        if not pts:
            return lambda ci: False
        pts = np.array(pts)
        cache = {}

        def want(ci):
            if ci not in cache:
                x0, x1, y0, y1 = W.chunk_bbox(self.rd, ci)
                dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - x1), 0)
                dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - y1), 0)
                cache[ci] = bool((np.hypot(dx, dy) < reach).any())
            return cache[ci]
        return want

    # ------------------------------------------------------------------------------------ road helpers
    def frame_at(self, s):
        x, y, z, tx, ty = self.rd.at(s)
        return x, y, z, tx, ty

    def pos(self, s, lat, dz=0.0):
        x, y, z, tx, ty = self.rd.at(s)
        return Vector((x - ty * lat, y + tx * lat, z + dz))

    def slope(self, s):
        z0 = self.rd.at(s - 1.5)[2]; z1 = self.rd.at(s + 1.5)[2]
        return math.atan2(z1 - z0, 3.0)

    def curvature(self, s):
        _, _, _, ax, ay = self.rd.at(s - 4); _, _, _, bx, by = self.rd.at(s + 4)
        return math.atan2(ax * by - ay * bx, ax * bx + ay * by) / 8.0

    # ------------------------------------------------------------------------------------ shot geometry
    def shot_anchor(self, shot):
        """Route position of the pack head at the shot's middle (explicit 'at', else snapped to a straight bit of road)."""
        if 'at' in shot:
            return shot['at']
        if '_anchor' not in shot:
            tm = 0.5 * (shot['t0'] + shot['t1'])
            shot['_anchor'] = self.snap_straight(C.odo(tm))
        return shot['_anchor']

    def snap_straight(self, s0, rad=300.0, win=100.0, step=10.0):
        rd = self.rd
        best, bs = None, s0
        for s in np.arange(max(80.0, s0 - rad), min(C.LEN - 80.0, s0 + rad), step):
            i0 = int((s - win / 2) / W.SAMP); i1 = int((s + win / 2) / W.SAMP)
            a = math.atan2(rd.ty[i0], rd.tx[i0]); b = math.atan2(rd.ty[i1], rd.tx[i1])
            turn = abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))
            grade = abs(rd.z[i1] - rd.z[i0]) / win
            sc = turn + 200 * grade + 0.01 * abs(s - s0)
            if best is None or sc < best:
                best, bs = sc, s
        return float(bs)

    def pack_speed(self, shot, s, k=None):
        v = shot.get('v')
        if v is None:
            v = min(12.0, max(7.0, C.route_speed(s)))
        if isinstance(v, (list, tuple)):
            return v[0] + (v[1] - v[0]) * (0.5 if k is None else k)
        return v

    def head_s(self, t, shot):
        s0 = self.shot_anchor(shot)
        tm = 0.5 * (shot['t0'] + shot['t1'])
        v = shot.get('v')
        if isinstance(v, (list, tuple)):
            a, b = v
            T = shot['t1'] - shot['t0']
            def integ(tt):    # integral of v from t0 to tt
                x = tt - shot['t0']
                return a * x + 0.5 * (b - a) * x * x / T
            return s0 + integ(t) - integ(tm)
        return s0 + self.pack_speed(shot, s0) * (t - tm)

    def order(self, shot):
        lead = shot.get('lead', ['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'])
        rest = [i for i in self.ids if i not in lead]
        return list(lead) + rest

    def rider_states(self, t, shot):
        sh = self.head_s(t, shot)
        order = self.order(shot)
        k_in = ease((t - shot['t0']) / max(1e-6, shot['t1'] - shot['t0']))
        v = self.pack_speed(shot, sh, k_in)
        formation = shot.get('formation', 'two')
        slots = shot.get('slots', {})
        st = {}
        gate_s = shot.get('gate_s')
        for k, rid in enumerate(order):
            if formation == 'single':
                ds = k * 2.2
                lat = LANE_L[0] + 0.12 * math.sin(t * 0.8 + k)
            elif formation == 'zipper':
                ds = k * 2.15
                lat = LANE_L[k % 2]
            else:
                row, lane = k // 2, k % 2
                ds = row * ROW + (0.35 if lane else 0.0)
                lat = LANE_L[lane]
            a, b, c = self.sway[rid]
            ds += 0.22 * math.sin(t * 0.7 * c + a)
            lat += 0.10 * math.sin(t * 0.55 * c + b)
            if rid in slots:
                ds, lat = slots[rid]
            extra = {}
            if shot.get('hook'):
                extra = shot['hook'](t, rid, k, self) or {}
            ds += extra.get('ds', 0.0); lat += extra.get('lat', 0.0)
            s_i = sh - ds
            if gate_s is not None:         # funnel to single file through the gap
                w = ease((s_i - (gate_s - 24)) / 16.0) * (1 - ease((s_i - (gate_s + 5)) / 14.0))
                lat = lat + (-1.82 - lat) * w + 0.05 * math.sin(t * 2 + k)
            if shot.get('avoid'):
                so, lo, push = shot['avoid']
                lat += push * math.exp(-((s_i - so) / 6.0) ** 2)
            grade = math.tan(self.slope(s_i))
            tend = {'angelo': 1.0, 'hulk': 0.55, 'gary': 0.25}.get(rid, 0.7)
            stand = max(0.0, min(1.0, (grade - 0.065) / 0.05)) * tend
            st[rid] = dict(s=s_i, lat=lat, v=v, stand=extra.get('stand', stand), k=k, extra=extra)
        return st, sh, v

    def rider_world(self, rid, stt, t):
        s, lat = stt['s'], stt['lat']
        x, y, z, tx, ty = self.rd.at(s)
        yaw = math.atan2(ty, tx)
        pitch = self.slope(s)
        roll = max(-0.35, min(0.35, -self.curvature(s) * stt['v'] * 0.9 * 0.45)) + 0.025 * math.sin(t * 1.9 + self.ph[rid])
        pos = Vector((x - ty * lat, y + tx * lat, z))
        ks = self.rhash.other_near(pos.x, pos.y, int(s / W.SAMP), 3.4)      # a crossing road's asphalt that is higher wins
        if ks:
            pos.z = max(pos.z, float(self.rd.z[ks].max()))
        return pos, yaw, pitch, roll

    # ------------------------------------------------------------------------------------ cameras
    def ref_point(self, ref, st, sh, shot):
        """(s, lat) of a camera reference given rider states st and head position sh."""
        if isinstance(ref, (int, float)):
            return float(ref), -1.8
        if ref == 'pack':
            ids = self.ids[:12]
            return float(np.mean([st[i]['s'] for i in ids])), float(np.mean([st[i]['lat'] for i in ids]))
        if ref == 'head':
            return sh, -1.8
        if ref == 'fixed':
            return self.shot_anchor(shot), -1.8
        if ref.startswith('fixed:'):
            return float(ref.split(':')[1]), -1.8
        return st[ref]['s'], st[ref]['lat']

    def map_head(self, t, shot):
        k = (t - shot['t0']) / max(1e-6, shot['t1'] - shot['t0'])
        h = shot.get('head')
        if isinstance(h, (list, tuple)):
            h = h[0] + (h[1] - h[0]) * k
        elif h is None:
            h = C.odo(t) / C.LEN
        if 'tailback' in shot:
            tl = max(0.0, h - shot['tailback'])
        else:
            tl = shot.get('tail', 0.0)
            tl = tl[0] + (tl[1] - tl[0]) * k if isinstance(tl, (list, tuple)) else tl
        return h, tl

    def camera_pose(self, t, shot, st, sh):
        k = ease((t - shot['t0']) / max(1e-6, shot['t1'] - shot['t0']))
        if shot['kind'] == 'calendar':
            o = [c for c in self.cal if c.t0 == shot['t0']][0].o
            kk = (t - shot['t0']) / (shot['t1'] - shot['t0'])
            eye = Vector((o.x + (-4.0 + 8.0 * kk) * (1 if shot['mode'] == 'weeks' else 0.3), o.y - 15.0 + 3.0 * kk, o.z + (3.6 - 5.0 * kk if shot['mode'] == 'weeks' else 0.0)))
            tgt = Vector((o.x + (-1.0 + 2.0 * kk), o.y, o.z + (2.0 - 4.5 * kk if shot['mode'] == 'weeks' else 0.0)))
            return dict(eye=eye, tgt=tgt, lens=[34, 40][0] + 6 * kk, roll=0.0, focus='tgt', fstop=0)
        if shot['kind'] == 'map':
            head, tail = self.map_head(t, shot)
            sP = head * C.LEN
            rd = self.rd
            P = Vector(rd.at(sP)[:3])
            kk = (t - shot['t0']) / (shot['t1'] - shot['t0'])
            if shot.get('loop') or shot.get('full'):
                ctr = Vector((-445.0, -327.0, 0.0))
                yaw = math.radians(-90 + 25 * kk) if not shot.get('loop') else math.radians(-90 + 70 * kk)
                dist = 9500 - 2400 * kk if shot.get('loop') else 9000 - 2500 * kk
                pitch = math.radians(78 - 18 * kk)
                eye = ctr + Vector((math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))) * dist
                return dict(eye=eye, tgt=ctr, lens=24, roll=math.pi / 2 if not shot.get('loop') else 0.0, focus='tgt', fstop=0)
            dist = 1500 - 700 * kk
            yaw = math.radians(-60 + 40 * kk)
            pitch = math.radians(48)
            eye = P + Vector((math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))) * dist
            return dict(eye=eye, tgt=P, lens=35, roll=0.0, focus='tgt', fstop=1.0)
        cm = shot['cam']
        ref = cm.get('ref', 'pack')
        static = cm.get('static', False)
        rs_t, rl_t = self.ref_point(ref, st, sh, shot)           # the subject now
        if static:                                                # eye fixed where the subject was at mid-shot
            tm = 0.5 * (shot['t0'] + shot['t1'])
            st_m, sh_m, _ = self.rider_states(tm, shot)
            rs_e, rl_e = self.ref_point(ref, st_m, sh_m, shot)
            tref = cm.get('tref', 'pack' if ref == 'fixed' else ref)
            rs_g, rl_g = self.ref_point(tref, st, sh, shot)
        else:
            rs_e, rl_e, rs_g, rl_g = rs_t, rl_t, rs_t, rl_t
        ex = cm.get('eye', (0, 6, 1.5)); tg = cm.get('tgt', (0, 0, 1.1))
        eye_o = [lerpv(ex[i], k) for i in range(3)]
        tgt_o = [lerpv(tg[i], k) for i in range(3)]
        if 'orbit' in cm:
            o = cm['orbit']
            ang = math.radians(o['a'][0] + (o['a'][1] - o['a'][0]) * k)
            eye_o = [o['r'] * math.cos(ang), o['r'] * math.sin(ang), eye_o[2]]

        def basis(rs, rl):
            x, y, z, tx, ty = self.rd.at(rs)
            T = Vector((tx, ty, 0)); N = Vector((-ty, tx, 0))
            return Vector((x, y, z)) + N * rl, T, N
        be, Te, Ne = basis(rs_e, rl_e)
        bg, Tg, Ng = basis(rs_g, rl_g)
        eye = be + Te * eye_o[0] + Ne * eye_o[1] + Vector((0, 0, eye_o[2]))
        tgt = bg + Tg * tgt_o[0] + Ng * tgt_o[1] + Vector((0, 0, tgt_o[2]))
        if static:      # tgt offsets are relative to the moving subject; eye offsets to the fixed spot
            pass
        sh_a = cm.get('shake', 0.0)
        if sh_a:
            eye += Vector((math.sin(t * 9.1), math.sin(t * 7.3 + 1), math.sin(t * 11.7 + 2))) * 0.015 * sh_a
            tgt += Vector((math.sin(t * 6.1 + 3), math.sin(t * 8.9), math.sin(t * 5.3 + 1))) * 0.02 * sh_a
        eye.z = max(eye.z, self.rd.at(rs_e)[2] + 0.12)
        return dict(eye=eye, tgt=tgt, lens=lerpv(cm.get('lens', 35), k), roll=lerpv(cm.get('roll', 0.0), k),
                    focus=cm.get('focus', 'tgt'), fstop=lerpv(cm.get('fstop', 0), k))

    def shot_uses_dof(self, shot):
        if shot['kind'] == 'map':
            return not (shot.get('full') or shot.get('loop'))
        if shot['kind'] == 'calendar':
            return False
        fs = shot['cam'].get('fstop', 0)
        return (max(fs) if isinstance(fs, (list, tuple)) else fs) > 0

    # ------------------------------------------------------------------------------------ baking
    def bake(self, frames):
        """Keyframe the rider roots and the camera for the given frames; also remember per-frame data for the handler.
        Neighbouring frames (f-1, f+1) are baked too so motion blur sees real velocity."""
        sc = bpy.context.scene
        cam = self.cam
        want = sorted(set(x + d for x in frames for d in (-1, 0, 1) if x + d >= 0))
        for f in want:
            t = f / C.FPS
            shot = shot_at(t)
            st, sh, v = self.rider_states(t, shot)
            for rid, stt in st.items():
                pos, yaw, pitch, roll = self.rider_world(rid, stt, t)
                self.riders[rid].place(pos.x, pos.y, pos.z, yaw, pitch, roll)
                r = self.riders[rid].root
                r.keyframe_insert('location', frame=f); r.keyframe_insert('rotation_quaternion', frame=f)
            cp = self.camera_pose(t, shot, st, sh)
            d = cp['tgt'] - cp['eye']
            q = d.to_track_quat('-Z', 'Y')
            if cp['roll']:
                q = q @ Quaternion((0, 0, 1), cp['roll'])
            cam.location = cp['eye']; cam.rotation_quaternion = q
            cam.keyframe_insert('location', frame=f); cam.keyframe_insert('rotation_quaternion', frame=f)
            cam.data.lens = cp['lens']; cam.data.keyframe_insert('lens', frame=f)
            # depth of field (use_dof is set per shot by render.py; here only the animated values)
            if cp['focus'] == 'tgt' or cp['focus'] not in st:
                dist = d.length
            else:
                pos, *_ = self.rider_world(cp['focus'], st[cp['focus']], t)
                dist = (pos + Vector((0, 0, 1.0)) - cp['eye']).length
            cam.data.dof.focus_distance = max(0.05, dist)
            cam.data.dof.aperture_fstop = cp['fstop'] if cp['fstop'] > 0 else 64.0
            cam.data.dof.keyframe_insert('focus_distance', frame=f); cam.data.dof.keyframe_insert('aperture_fstop', frame=f)
            self.cache[f] = dict(st=st, sh=sh, v=v, shot=shot, cp=cp)

    # ------------------------------------------------------------------------------------ per-frame handler
    def on_frame(self, scene, *a):
        f = scene.frame_current
        d = self.cache.get(f)
        if d is None:
            return
        t = f / C.FPS
        shot = d['shot']; st = d['st']
        # time of day + sun cheat relative to the camera heading
        cp = d['cp']
        if '_hdg' not in shot:                       # one heading per shot: a sun that follows camera shake makes shadows shimmer
            tm = 0.5 * (shot['t0'] + shot['t1'])
            st_m, sh_m, _ = self.rider_states(tm, shot)
            cm_ = self.camera_pose(tm, shot, st_m, sh_m)
            shot['_hdg'] = round(math.degrees(math.atan2(cm_['tgt'].y - cm_['eye'].y, cm_['tgt'].x - cm_['eye'].x)) / 5.0) * 5.0
        az = shot['_hdg'] + min(178.0, max(118.0, shot.get('sun', 150)))   # keep the light on the subjects' faces
        day, night = S.apply_time_of_day(t, self.sun, self.moon, az)
        set_glow('M_windows', 2.4 * night)
        set_glow('M_lamps', 5.0 * night)
        lights = smooth01((t - 198) / 6)
        set_glow('M_bikelight', 25.0 * lights)
        for rid, stt in st.items():
            r = self.riders[rid]
            cad = self.cad[rid]
            theta = -2 * math.pi * (cad / 60.0) * t + self.ph[rid]     # forward pedalling = crank angle decreasing (x fwd, z up)
            # wheels: at 30 fps a truly rolling 20-spoke wheel aliases (wagon-wheel) and looks like it spins BACKWARDS, so the
            # visible rotation is a slow forward spin that still scales with speed (always < half a spoke gap per frame)
            spin = math.radians(18.0) * (0.10 + 0.32 * min(1.0, stt['v'] / 12.0)) * C.FPS * t + self.ph[rid]
            ex = mood_over(t, rid)
            stand = stt['stand']
            if stand > 0.05:
                theta *= 0.7
            roll = 0.09 * stand * math.sin(theta)
            r.pose(theta, spin, stand=stand, roll=roll, mood=ex.get('mood'), stress=ex.get('stress', 0.0), look=ex.get('look', 0.0), lights=lights)
        # props
        for p in self.props:
            p.update(t)
        if self.items is not None:
            an, *_ = self.rider_world('angelo', st['angelo'], t)
            self.items.update(t, an, cp['eye'])
        # route glow for the map scenes (haze would wash the aerial views, so it is off for map and calendar shots)
        mp = shot['kind'] == 'map'
        ng = scene.compositing_node_group
        if ng is not None and 'HazeRange' in ng.nodes:
            ng.nodes['HazeRange'].inputs['To Max'].default_value = 0.0 if shot['kind'] in ('map', 'calendar') else 0.5
        self.route.ob.hide_render = not mp
        self.table.hide_render = not mp
        if mp:
            head, tail = self.map_head(t, shot)
            self.route.set(head, tail)
        # travelling light pools
        pk = st['karim']
        pl, *_ = self.rider_world('karim', pk, t)
        self.packlight.location = pl + Vector((0, 0, 4.0)); self.packlight.data.energy = 900.0 * night
        self.packlight2.location = pl + Vector((-6, 3, 3.0)); self.packlight2.data.energy = 300.0 * night
        # screen-space markers for the overlay
        from bpy_extras.object_utils import world_to_camera_view
        mk = {}
        for rid in K.NAMED:
            pos, yaw, pitch, roll = self.rider_world(rid, st[rid], t)
            p = world_to_camera_view(scene, self.cam, pos + Vector((0.0, 0.0, 1.75)))
            mk[rid] = (p.x, 1 - p.y, p.z)
        self.markers[f] = mk


def smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)
