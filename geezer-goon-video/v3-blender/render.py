"""blender -b --factory-startup --gpu-backend metal -P render.py -- f0 f1 outdir [w h samples]      (range)
   blender ... -P render.py -- a,b,c outdir [w h samples]                                          (explicit frames)

Renders frames to outdir/f_#####.jpg. Finished frames (.jpg or earlier .png) are skipped, so a job can be re-run after a
crash. Roots + camera are keyframed (motion blur); a handler poses limbs/lights. First and last frame of each shot render
without motion blur so it can't smear across a cut. Per-frame rider screen positions go to outdir/markers.jsonl as they are
rendered (a crash loses at most the current frame).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import common as C, scene_setup as S, film as F
from shots import SHOTS, shot_at

argv = sys.argv[sys.argv.index('--') + 1:]
if ',' in argv[0]:
    FRAMES = [int(x) for x in argv[0].split(',')]
    f0, f1 = min(FRAMES), max(FRAMES)
    rest = argv[1:]
else:
    f0, f1 = int(argv[0]), int(argv[1])
    FRAMES = list(range(f0, f1 + 1))
    rest = argv[2:]
outdir = rest[0]
W_, H_, SAMP = (int(rest[1]), int(rest[2]), int(rest[3])) if len(rest) >= 4 else (1280, 720, 10)
os.makedirs(outdir, exist_ok=True)


def have(x):
    return any(os.path.exists(os.path.join(outdir, f'f_{x:05d}.{e}')) for e in ('jpg', 'png'))


FRAMES = [x for x in FRAMES if not have(x)]
if not FRAMES:
    print('DONE 0 frames (nothing to do)', flush=True)
    sys.exit(0)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = min(FRAMES), max(FRAMES)
S.setup_render(W_, H_, SAMP)
sc.render.film_transparent = False
t0 = time.time()
film = F.Film(sc.collection, frames=FRAMES)
for o in sc.collection.objects:
    if any(o.name.startswith(p) for p in os.environ.get('GG_HIDE', '').split(',') if p):
        o.hide_render = True
print(f'build {time.time() - t0:.1f}s ({len(film.world_objs)} world objects)', flush=True)
t0 = time.time()
film.bake(FRAMES)
print(f'bake {time.time() - t0:.1f}s', flush=True)
mk_path = os.path.join(outdir, 'markers.jsonl')
mk_fh = open(mk_path, 'a')


def on_frame(scene, *a):
    film.on_frame(scene, *a)
    f = scene.frame_current
    if f in film.markers:
        mk_fh.write(json.dumps({'f': f, 'm': film.markers[f]}) + '\n')
        mk_fh.flush()


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(on_frame)

segs = []
for x in FRAMES:
    shot = shot_at(x / C.FPS)
    if segs and segs[-1][2] is shot and x == segs[-1][1] + 1:
        segs[-1] = (segs[-1][0], x, shot)
    else:
        segs.append((x, x, shot))

done = 0
for a, b, shot in segs:
    todo = [x for x in FRAMES if a <= x <= b and not have(x)]
    if not todo:
        continue
    ts = time.time()
    film.cam.data.dof.use_dof = film.shot_uses_dof(shot)
    for x in todo:
        sh_first = int(round(shot['t0'] * C.FPS)); sh_last = int(round(shot['t1'] * C.FPS)) - 1
        edge = (x == sh_first or x == sh_last)
        sc.render.use_motion_blur = not edge
        sc.frame_set(x)
        sc.render.filepath = os.path.join(outdir, f'f_{x:05d}.jpg')
        bpy.ops.render.render(write_still=True)
        done += 1
    print(f'shot {shot["t0"]:.1f}-{shot["t1"]:.1f} frames {a}-{b}: {len(todo)} in {time.time() - ts:.1f}s ({(time.time() - ts) / len(todo):.2f}s/f)', flush=True)
print('DONE', done, 'frames', flush=True)
