"""Render settings, sky, sun/moon, camera."""
import math
import bpy
from mathutils import Vector, Euler, Matrix
import common as C


def setup_render(w=1920, h=1080, samples=24, fast=False):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x = w; sc.render.resolution_y = h; sc.render.resolution_percentage = 100
    sc.render.fps = C.FPS
    sc.render.image_settings.file_format = 'JPEG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.image_settings.quality = 95
    sc.render.use_file_extension = False
    ev = sc.eevee
    import os
    E = os.environ
    ev.taa_render_samples = int(E.get('GG_SAMPLES', samples))
    ev.use_shadows = E.get('GG_SHADOWS', '1') == '1'
    ev.use_raytracing = (E.get('GG_RT', '0') == '1') and not fast
    try:
        ev.ray_tracing_method = 'SCREEN'
    except Exception:
        pass
    ev.use_fast_gi = False
    ev.shadow_ray_count = int(E.get('GG_SRAYS', 1))
    ev.shadow_step_count = int(E.get('GG_SSTEPS', 6))
    ev.shadow_resolution_scale = float(E.get('GG_SRES', 1.0))
    if True:
        try:
            ev.shadow_pool_size = E.get('GG_POOL', '512')
        except Exception as e:
            print('pool err', e)
    sc.render.use_motion_blur = (E.get('GG_MB', '1') == '1') and not fast
    sc.render.motion_blur_shutter = 0.4
    ev.motion_blur_steps = 1
    ev.motion_blur_max = 48
    # colour: AgX with a punchy look reads like illustration, not photo
    vs = sc.view_settings
    for name in ('Standard',):
        try:
            vs.view_transform = name; break
        except Exception:
            continue
    try:
        vs.look = 'AgX - Punchy'
    except Exception:
        pass
    vs.exposure = 0.0
    sc.display_settings.display_device = 'sRGB'


SKYK = [  # t, zenith, horizon  (v2's story, a lighter and less magenta palette)
    (0, '#5d93d6', '#ffd9a3'), (45, '#5a8ad0', '#ffc58a'), (90, '#5a82cc', '#ffa982'), (125, '#6377c0', '#f59a86'),
    (155, '#4b55a8', '#e0809c'), (185, '#2a3170', '#8a5a96'), (205, '#101a48', '#3a3768'), (250, '#050818', '#18173a')]


def _lin(h):
    from bl_util import hexc
    return hexc(h)


def setup_world(sc=None):
    sc = sc or bpy.context.scene
    w = bpy.data.worlds.new('sky') if not sc.world else sc.world
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    N = nt.nodes; L = nt.links
    out = N.new('ShaderNodeOutputWorld')
    bg = N.new('ShaderNodeBackground'); bg.name = 'BG'; bg.inputs['Strength'].default_value = 1.0
    tc = N.new('ShaderNodeTexCoord')
    # direction = Generated (unit vector in the world shader)
    d = N.new('ShaderNodeVectorMath'); d.operation = 'MULTIPLY_ADD'
    d.inputs[1].default_value = (1, 1, 1); d.inputs[2].default_value = (0, 0, 0)   # Generated is already the unit direction
    L.new(tc.outputs['Generated'], d.inputs[0])
    sep = N.new('ShaderNodeSeparateXYZ'); L.new(d.outputs['Vector'], sep.inputs['Vector'])
    # elevation factor: clamp(z) ^ 0.5
    clampz = N.new('ShaderNodeMath'); clampz.operation = 'MAXIMUM'; clampz.inputs[1].default_value = 0.0
    L.new(sep.outputs['Z'], clampz.inputs[0])
    powz = N.new('ShaderNodeMath'); powz.operation = 'POWER'; powz.inputs[1].default_value = 0.8
    L.new(clampz.outputs[0], powz.inputs[0])
    ramp3 = N.new('ShaderNodeValToRGB'); ramp3.name = 'SkyRamp'
    cr = ramp3.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[1].position = 1.0
    e = cr.elements.new(0.30)
    L.new(powz.outputs[0], ramp3.inputs['Fac'])
    grad = ramp3
    # below the horizon: a darker ground tone
    neg = N.new('ShaderNodeMath'); neg.operation = 'MULTIPLY'; neg.use_clamp = True; neg.inputs[1].default_value = -5.0
    L.new(sep.outputs['Z'], neg.inputs[0])
    gnd = N.new('ShaderNodeMix'); gnd.data_type = 'RGBA'; gnd.blend_type = 'MULTIPLY'
    gnd.inputs['B'].default_value = (0.35, 0.38, 0.32, 1)
    L.new(neg.outputs[0], gnd.inputs['Factor']); L.new(grad.outputs['Color'], gnd.inputs['A'])
    # sun glow + disc
    sv = N.new('ShaderNodeCombineXYZ'); sv.name = 'SunVec'
    dot = N.new('ShaderNodeVectorMath'); dot.operation = 'DOT_PRODUCT'
    L.new(d.outputs['Vector'], dot.inputs[0]); L.new(sv.outputs['Vector'], dot.inputs[1])
    pos = N.new('ShaderNodeMath'); pos.operation = 'MAXIMUM'; pos.inputs[1].default_value = 0.0; L.new(dot.outputs['Value'], pos.inputs[0])
    halo = N.new('ShaderNodeMath'); halo.operation = 'POWER'; halo.inputs[1].default_value = 28.0; L.new(pos.outputs[0], halo.inputs[0])
    core = N.new('ShaderNodeMath'); core.operation = 'GREATER_THAN'; core.inputs[1].default_value = 0.99935; L.new(dot.outputs['Value'], core.inputs[0])
    halo_w = N.new('ShaderNodeMath'); halo_w.operation = 'MULTIPLY'; halo_w.name = 'HaloGain'; halo_w.inputs[1].default_value = 0.55; L.new(halo.outputs[0], halo_w.inputs[0])
    core_w = N.new('ShaderNodeMath'); core_w.operation = 'MULTIPLY'; core_w.name = 'CoreGain'; core_w.inputs[1].default_value = 3.0; L.new(core.outputs[0], core_w.inputs[0])
    gsum = N.new('ShaderNodeMath'); gsum.operation = 'ADD'; L.new(halo_w.outputs[0], gsum.inputs[0]); L.new(core_w.outputs[0], gsum.inputs[1])
    suncol = N.new('ShaderNodeRGB'); suncol.name = 'SunCol'
    glow = N.new('ShaderNodeVectorMath'); glow.operation = 'SCALE'
    L.new(suncol.outputs[0], glow.inputs[0]); L.new(gsum.outputs[0], glow.inputs[3])
    addg = N.new('ShaderNodeMix'); addg.data_type = 'RGBA'; addg.blend_type = 'ADD'; addg.inputs['Factor'].default_value = 1.0
    L.new(gnd.outputs['Result'], addg.inputs['A']); L.new(glow.outputs['Vector'], addg.inputs['B'])
    # stars
    vor = N.new('ShaderNodeTexVoronoi'); vor.voronoi_dimensions = '3D'; vor.feature = 'F1'; vor.distance = 'EUCLIDEAN'
    vor.inputs['Scale'].default_value = 190.0; vor.inputs['Randomness'].default_value = 1.0
    L.new(d.outputs['Vector'], vor.inputs['Vector'])
    ramp = N.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[1].position = 0.03
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1); ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    L.new(vor.outputs['Distance'], ramp.inputs['Fac'])
    sg = N.new('ShaderNodeMath'); sg.operation = 'MULTIPLY'; sg.name = 'StarGain'; sg.inputs[1].default_value = 0.0
    L.new(ramp.outputs['Color'], sg.inputs[0])
    skyz = N.new('ShaderNodeMath'); skyz.operation = 'MULTIPLY'; skyz.use_clamp = True; skyz.inputs[1].default_value = 3.0; L.new(sep.outputs['Z'], skyz.inputs[0])
    sg2 = N.new('ShaderNodeMath'); sg2.operation = 'MULTIPLY'; L.new(sg.outputs[0], sg2.inputs[0]); L.new(skyz.outputs[0], sg2.inputs[1])
    sc3 = N.new('ShaderNodeCombineColor')
    for k in ('Red', 'Green', 'Blue'):
        L.new(sg2.outputs[0], sc3.inputs[k])
    adds = N.new('ShaderNodeMix'); adds.data_type = 'RGBA'; adds.blend_type = 'ADD'; adds.inputs['Factor'].default_value = 1.0
    L.new(addg.outputs['Result'], adds.inputs['A']); L.new(sc3.outputs['Color'], adds.inputs['B'])
    L.new(adds.outputs['Result'], bg.inputs['Color'])
    L.new(bg.outputs['Background'], out.inputs['Surface'])
    return w


def setup_compositor():
    """Bloom for the emissive lights + a depth haze toward the horizon colour (sky pixels are exempt)."""
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    vl.use_pass_z = True
    ng = bpy.data.node_groups.new('Comp', 'CompositorNodeTree')
    sc.compositing_node_group = ng
    sc.render.use_compositing = True
    ng.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    N = ng.nodes; L = ng.links
    rl = N.new('CompositorNodeRLayers')
    go = N.new('NodeGroupOutput')
    gl = N.new('CompositorNodeGlare'); gl.name = 'Bloom'
    try:
        gl.inputs['Type'].default_value = 'Bloom'
    except Exception:
        pass
    gl.inputs['Threshold'].default_value = 1.1
    gl.inputs['Strength'].default_value = 0.55
    try:
        gl.inputs['Size'].default_value = 0.5
        gl.inputs['Smoothness'].default_value = 0.5
    except Exception:
        pass
    L.new(rl.outputs['Image'], gl.inputs['Image'])
    mr = N.new('ShaderNodeMapRange'); mr.clamp = True; mr.name = 'HazeRange'
    mr.inputs['From Min'].default_value = 70.0; mr.inputs['From Max'].default_value = 1500.0
    mr.inputs['To Min'].default_value = 0.0; mr.inputs['To Max'].default_value = 0.5
    L.new(rl.outputs['Depth'], mr.inputs['Value'])
    lt = N.new('ShaderNodeMath'); lt.operation = 'LESS_THAN'; lt.inputs[1].default_value = 100000.0
    L.new(rl.outputs['Depth'], lt.inputs[0])
    fac = N.new('ShaderNodeMath'); fac.operation = 'MULTIPLY'
    L.new(mr.outputs['Result'], fac.inputs[0]); L.new(lt.outputs[0], fac.inputs[1])
    hz = N.new('CompositorNodeRGB'); hz.name = 'HazeCol'
    mx = N.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MIX'
    L.new(fac.outputs[0], mx.inputs['Factor'])
    L.new(gl.outputs['Image'], mx.inputs['A']); L.new(hz.outputs[0], mx.inputs['B'])
    L.new(mx.outputs['Result'], go.inputs['Image'])
    return ng


def add_sun(coll=None):
    sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 4.0; sun.angle = math.radians(3.0)
    sun.use_shadow = True
    ob = bpy.data.objects.new('sun', sun)
    (coll or bpy.context.scene.collection).objects.link(ob)
    moon = bpy.data.lights.new('moon', 'SUN'); moon.energy = 0.0; moon.angle = math.radians(2.0); moon.color = (0.62, 0.72, 1.0)
    mob = bpy.data.objects.new('moon', moon)
    (coll or bpy.context.scene.collection).objects.link(mob)
    return ob, mob


def sun_state(t):
    """Elevation (deg) of the sun by song time: golden hour -> set -> dark."""
    keys = [(0, 28.0), (60, 22.0), (110, 15.0), (150, 8.0), (172, 1.0), (190, -5.0), (205, -9.0), (260, -12.0)]
    for i in range(1, len(keys)):
        if t <= keys[i][0]:
            a, b = keys[i - 1], keys[i]
            k = C.smooth((t - a[0]) / (b[0] - a[0]))
            return a[1] + (b[1] - a[1]) * k
    return keys[-1][1]


def sky_colors(t):
    ks = SKYK
    i = 1
    while i < len(ks) - 1 and ks[i][0] < t:
        i += 1
    a, b = ks[i - 1], ks[i]
    k = C.smooth((t - a[0]) / (b[0] - a[0]))
    ca, cb = _lin(a[1]), _lin(b[1]); ha, hb = _lin(a[2]), _lin(b[2])
    lp = lambda p, q: tuple(p[j] + (q[j] - p[j]) * k for j in range(3)) + (1.0,)
    return lp(ca, cb), lp(ha, hb)


def apply_time_of_day(t, sun_ob, moon_ob, az_deg):
    """Sun direction from elevation + azimuth (az measured CCW from +x in world); drives lamps + sky nodes."""
    el = sun_state(t)
    elr = math.radians(el); azr = math.radians(az_deg)
    d = Vector((math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr), math.sin(elr)))
    sun_ob.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    day = C.smooth((el + 1.5) / 6.0)
    sun_ob.data.energy = 6.0 * day
    warm = C.smooth((el - 0.5) / 24)
    sun_ob.data.color = (1.0, 0.70 + 0.22 * warm, 0.46 + 0.4 * warm)
    night = C.smooth((-el - 1.0) / 6.0)
    md = Vector((-0.45, -0.35, 0.82)).normalized()
    moon_ob.rotation_euler = (-md).to_track_quat('-Z', 'Y').to_euler()
    moon_ob.data.energy = 0.65 * night
    nt = bpy.context.scene.world.node_tree
    top, hor = sky_colors(t)
    import colorsys
    mid = [top[i] * 0.45 + hor[i] * 0.55 for i in range(3)]
    h_, s_, v_ = colorsys.rgb_to_hsv(*mid)
    mid = colorsys.hsv_to_rgb(h_, min(1.0, s_ * 1.2), min(1.0, v_ * 1.08)) + (1.0,)
    cr = nt.nodes['SkyRamp'].color_ramp
    cr.elements[0].color = hor; cr.elements[1].color = mid; cr.elements[2].color = top
    sv = nt.nodes['SunVec']
    sv.inputs['X'].default_value, sv.inputs['Y'].default_value, sv.inputs['Z'].default_value = d.x, d.y, d.z
    nt.nodes['SunCol'].outputs[0].default_value = (1.0, 0.62 + 0.3 * warm, 0.30 + 0.45 * warm, 1.0)
    vis = day
    nt.nodes['HaloGain'].inputs[1].default_value = 0.65 * vis
    nt.nodes['CoreGain'].inputs[1].default_value = 3.0 * vis
    nt.nodes['StarGain'].inputs[1].default_value = 2.5 * night
    ng = bpy.context.scene.compositing_node_group
    if ng is not None and 'HazeCol' in ng.nodes:
        k = 0.55 + 0.35 * (1 - day)
        ng.nodes['HazeCol'].outputs[0].default_value = (hor[0] * k, hor[1] * k, hor[2] * k, 1.0)
    nt.nodes['BG'].inputs['Strength'].default_value = 0.85 + 0.25 * (1 - day)
    return day, night


def make_camera(coll=None):
    cam = bpy.data.cameras.new('cam')
    cam.lens = 35; cam.sensor_width = 36
    cam.clip_start = 0.05; cam.clip_end = 3000
    ob = bpy.data.objects.new('cam', cam)
    (coll or bpy.context.scene.collection).objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def look_at(ob, eye, target, roll=0.0):
    eye = Vector(eye); target = Vector(target)
    d = target - eye
    q = d.to_track_quat('-Z', 'Y')
    ob.location = eye
    ob.rotation_euler = q.to_euler()
    if roll:
        ob.rotation_euler.rotate(Euler((0, 0, 0)))
        ob.rotation_euler = (q @ Euler((0, 0, roll)).to_quaternion()).to_euler()
