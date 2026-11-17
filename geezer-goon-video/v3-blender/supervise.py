"""Crash-tolerant render driver:  python3 supervise.py OUT_DIR [W H SAMPLES MAXSEG F0 F1]

One Blender process per shot (a shot longer than MAXSEG frames is split). Each process builds only the part of the world
its shot can see, which keeps GPU memory low; if Blender dies inside Eevee (it does, occasionally, on this 8 GB machine)
the segment is retried and already-rendered frames are skipped. Progress goes to OUT_DIR/status.txt.
"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import shots as SH
out = sys.argv[1]
W, H, SAMP, MAXSEG = (int(x) for x in sys.argv[2:6]) if len(sys.argv) >= 6 else (1280, 720, 10, 150)
F0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
F1 = int(sys.argv[7]) if len(sys.argv) > 7 else C.NF - 1
os.makedirs(out, exist_ok=True)
status = os.path.join(out, 'status.txt')
t_start = time.time()


def have(f):
    return any(os.path.exists(os.path.join(out, f'f_{f:05d}.{e}')) for e in ('jpg', 'png'))


def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(status, 'a') as fh:
        fh.write(line + '\n')


segments = []
for s in SH.SHOTS:
    a = max(F0, int(round(s['t0'] * C.FPS))); b = min(F1, int(round(s['t1'] * C.FPS)) - 1)
    for k in range(a, b + 1, MAXSEG):
        segments.append((k, min(b, k + MAXSEG - 1)))
crashes = 0
for a, b in segments:
    for attempt in range(1, 9):
        missing = [f for f in range(a, b + 1) if not have(f)]
        if not missing:
            break
        lo, hi = min(missing), max(missing)
        cmd = ['caffeinate', '-i', 'blender', '-b', '--factory-startup', '--gpu-backend', 'metal', '-P', os.path.join(HERE, 'render.py'),
               '--', str(lo), str(hi), out, str(W), str(H), str(SAMP)]
        t0 = time.time()
        r = subprocess.run(cmd, stdout=open(os.path.join(out, 'last_run.log'), 'w'), stderr=subprocess.STDOUT)
        miss2 = sum(1 for f in range(a, b + 1) if not have(f))
        if miss2:
            crashes += 1
            log(f'segment {a}-{b}: attempt {attempt} ended with {miss2} frames missing (rc={r.returncode}) after {time.time() - t0:.0f}s')
            time.sleep(2)
    left = [f for f in range(a, b + 1) if not have(f)]
    total_done = sum(1 for f in range(F0, F1 + 1) if have(f))
    log(f'segment {a}-{b} {"OK" if not left else "INCOMPLETE " + str(len(left))}; {total_done}/{F1 - F0 + 1} frames; {(time.time() - t_start) / 60:.1f} min; crashes {crashes}')
log('ALL DONE' if all(have(f) for f in range(F0, F1 + 1)) else 'FINISHED WITH GAPS')
