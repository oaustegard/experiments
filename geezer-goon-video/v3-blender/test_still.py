"""blender -b --factory-startup -P test_still.py -- out.png [s] [t] [cam]  : first look at world + riders."""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector
import common as C, world as W, riders as R, scene_setup as S, cast
argv = sys.argv[sys.argv.index('--') + 1:]
out = argv[0]; s0 = float(argv[1]) if len(argv) > 1 else 6500; t = float(argv[2]) if len(argv) > 2 else 20.0
camname = argv[3] if len(argv) > 3 else 'side'

t_all = time.time()
def mark(m): os.write(2, (m + '\n').encode())
coll = bpy.context.scene.collection
S.setup_render(1280, 720, 16)
S.setup_world()
sun, moon = S.add_sun()
t0 = time.time(); objs = W.build_world(coll); mark('world built'); print('world', len(objs), 'objs', round(time.time() - t0, 1), 's')
rd = W.get_road()
for o in objs:
    if any(o.name.startswith(p) for p in os.environ.get('HIDE','').split(',') if p): o.hide_render = True

order = ['karim', 'phil', 'hulk', 'gary', 'angelo', 'me']
specs = [cast.CAST[k] for k in order] + cast.far_specs(14)
riders = []
t0 = time.time()
for sp in specs:
    riders.append(R.Rider(sp, coll))
mark('riders built'); print('riders', round(time.time() - t0, 1), 's')

x, y, z, tx, ty = rd.at(s0)
yaw = math.atan2(ty, tx)
for i, r in enumerate(riders):
    row = i // 2; lane = i % 2
    along = -row * 2.3
    lat = -1.0 - 1.1 * lane
    sx, sy, sz, ttx, tty = rd.at(s0 + along)
    nx, ny = -tty, ttx
    r.place(sx + nx * lat, sy + ny * lat, sz, math.atan2(tty, ttx))
    theta = 1.3 * i + 0.4
    r.pose(theta, theta * 3, stand=0.0, mood='neutral')

mark('posed'); az = math.degrees(yaw) + 120
S.apply_time_of_day(t, sun, moon, az)
if os.environ.get('SKYGAIN'): bpy.context.scene.world.node_tree.nodes['BG'].inputs['Strength'].default_value = float(os.environ['SKYGAIN'])
if os.environ.get('VT'): bpy.context.scene.view_settings.view_transform = os.environ['VT']
if os.environ.get('LOOK'):
    try: bpy.context.scene.view_settings.look = os.environ['LOOK']
    except Exception as e: mark('look fail ' + str(e))
if os.environ.get('SUN'): sun.data.energy = float(os.environ['SUN'])
if os.environ.get('EXPO'): bpy.context.scene.view_settings.exposure = float(os.environ['EXPO'])
cam = S.make_camera(coll)
cx, cy, cz, ctx, cty = rd.at(s0 - 6)
lx, ly = -cty, ctx
if camname == 'side':
    S.look_at(cam, (cx + lx * 9 + ctx * 2, cy + ly * 9 + cty * 2, cz + 1.4), (cx + ctx * 6, cy + cty * 6, cz + 1.2))
    cam.data.lens = 50
elif camname == 'chase':
    S.look_at(cam, (cx - ctx * 5 - lx * 1.5, cy - cty * 5 - ly * 1.5, cz + 1.7), (cx + ctx * 10, cy + cty * 10, cz + 1.0))
    cam.data.lens = 28
elif camname == 'front':
    S.look_at(cam, (x + tx * 16 + lx * 1.5, y + ty * 16 + ly * 1.5, z + 0.9), (x - tx * 4, y - ty * 4, z + 1.2))
    cam.data.lens = 40
elif camname == 'drone':
    S.look_at(cam, (cx + lx * 14 - ctx * 8, cy + ly * 14 - cty * 8, cz + 22), (cx + ctx * 6, cy + cty * 6, cz))
    cam.data.lens = 35
mark('camera set'); bpy.context.scene.render.filepath = out
t0 = time.time()
bpy.ops.render.render(write_still=True)
print('RENDER', round(time.time() - t0, 1), 's  total', round(time.time() - t_all, 1))
