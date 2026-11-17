"""Blender helpers: numpy -> mesh, shared primitive meshes, materials."""
import bpy, bmesh, math
import numpy as np
from mathutils import Matrix, Vector


def hexc(h, a=1.0):
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    # sRGB -> linear
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (f(r), f(g), f(b), a)


def mesh_from_arrays(name, V, F, colors=None, smooth=False):
    """V (n,3) float, F (m,3) int triangles, colors (n,4) linear or None (white)."""
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set('co', V.ravel())
    m = len(F)
    me.loops.add(3 * m)
    me.loops.foreach_set('vertex_index', F.ravel())
    me.polygons.add(m)
    me.polygons.foreach_set('loop_start', np.arange(0, 3 * m, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if smooth:
        me.polygons.foreach_set('use_smooth', np.ones(m, dtype=bool))
    col = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    if colors is None:
        colors = np.ones((len(V), 4), dtype=np.float32)
    col.data.foreach_set('color', np.ascontiguousarray(colors, dtype=np.float32).ravel())
    return me


def add_obj(me, name, collection=None, color=None):
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if color is not None:
        ob.color = color
    return ob


# ----------------------------------------------------------------------------------- primitives (unit size)
def unit_cylinder(seg=10, name='cyl'):
    """Radius 1, z in [-0.5, 0.5]: smooth sides, flat caps (cap rim split so normals don't bleed). Scale (r, r, L)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=1, radius2=1, depth=1)
    rim = [e for e in bm.edges if len(e.link_faces) == 2 and (len(e.link_faces[0].verts) != len(e.link_faces[1].verts))]
    bmesh.ops.split_edges(bm, edges=rim)
    for f in bm.faces:
        f.smooth = len(f.verts) == 4
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    _white(me)
    return me


def unit_sphere(seg=10, rings=7, name='sph'):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    _white(me)
    return me


def unit_box(name='box'):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    _white(me)
    return me


def _white(me):
    col = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    col.data.foreach_set('color', np.ones(len(me.vertices) * 4, dtype=np.float32))


def bm_to_mesh(bm, name, color=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    col = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    c = np.tile(np.array(color if color else (1, 1, 1, 1), dtype=np.float32), (len(me.vertices), 1))
    col.data.foreach_set('color', c.ravel())
    return me


def limb_matrix(a, b, r):
    """Matrix placing a unit cylinder (z axis, centered) between points a and b with radius r."""
    a = Vector(a); b = Vector(b)
    d = b - a
    L = d.length
    if L < 1e-6:
        return Matrix.Translation(a) @ Matrix.Diagonal((r, r, 1e-4, 1))
    z = d / L
    up = Vector((0, 0, 1)) if abs(z.z) < 0.95 else Vector((1, 0, 0))
    x = up.cross(z).normalized()
    y = z.cross(x)
    rot = Matrix(((x.x, y.x, z.x, 0), (x.y, y.y, z.y, 0), (x.z, y.z, z.z, 0), (0, 0, 0, 1)))
    return Matrix.Translation((a + b) / 2) @ rot @ Matrix.Diagonal((r, r, L, 1))


# ----------------------------------------------------------------------------------- materials
def main_material():
    """Vertex colour x object colour, rough dielectric. One material for almost everything."""
    m = bpy.data.materials.get('M_main')
    if m:
        return m
    m = bpy.data.materials.new('M_main')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'Col'; attr.attribute_type = 'GEOMETRY'
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    nt.links.new(attr.outputs['Color'], mul.inputs['A'])
    nt.links.new(oi.outputs['Color'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.55
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return m


def glow_material(name='M_glow', strength=0.0):
    """Emission = vertex colour x strength. Strength is the 'Strength' input of the Emission node (animated from python)."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'Col'; attr.attribute_type = 'GEOMETRY'
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    nt.links.new(attr.outputs['Color'], mul.inputs['A'])
    nt.links.new(oi.outputs['Color'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], em.inputs['Color'])
    em.inputs['Strength'].default_value = strength
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    return m


def window_material(name='M_windows', base='#2f4a63'):
    """Dark glass by day; the vertex colour (warm light) is emitted at night. Strength animated via set_glow."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.name = 'Glass'
    bsdf.inputs['Base Color'].default_value = hexc('#c9ccd2' if name == 'M_lamps' else base)
    bsdf.inputs['Roughness'].default_value = 0.12
    attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'Col'; attr.attribute_type = 'GEOMETRY'
    nt.links.new(attr.outputs['Color'], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = 0.0
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return m


def set_glow(name, strength):
    m = bpy.data.materials.get(name)
    if not m:
        return
    nodes = m.node_tree.nodes
    if 'Emission' in nodes:
        nodes['Emission'].inputs['Strength'].default_value = strength
    elif 'Glass' in nodes:
        nodes['Glass'].inputs['Emission Strength'].default_value = strength
