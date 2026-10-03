"""Bundle route + lyrics + audio analysis into web/data.js for the HTML renderer."""
import json, math, re
import numpy as np

raw = json.load(open('route_raw.json'))
RIDE_START = 9   # samples 0-8 are the roll from home to the ride start on Beach Dr; they are not part of the video
L = np.array(raw['location'])[RIDE_START:]; A = np.array(raw['altitude'])[RIDE_START:]; V = np.array(raw['velocity_smooth'])[RIDE_START:]
full = np.array(raw['location']); R0 = 6371000.0
_s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(np.radians(full[:, 1]) * R0 * math.cos(math.radians(full[:, 0].mean()))), np.diff(np.radians(full[:, 0]) * R0)))])
SHIFT = float(_s[RIDE_START] * 40494.8 / _s[-1])   # metres of the original activity that precede the ride start
R = 6371000.0
lat0, lon0 = L[:, 0].mean(), L[:, 1].mean()
X = np.radians(L[:, 1] - lon0) * R * math.cos(math.radians(lat0))
Y = np.radians(L[:, 0] - lat0) * R
seg = np.hypot(np.diff(X), np.diff(Y))
S = np.concatenate([[0], np.cumsum(seg)])
S *= 40494.8 / _s[-1]                     # same distance scale as the full activity (GPS decimation shortens corners by ~1%)

# resample every 10 m
step = 10.0
sg = np.arange(0, S[-1], step)
def interp(a): return np.interp(sg, S, a)
def gauss(a, sigma_m):
    k = int(sigma_m / step * 3)
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * step / sigma_m) ** 2); w /= w.sum()
    pad = np.pad(a, k, mode='edge'); return np.convolve(pad, w, mode='valid')
alt = gauss(interp(A), 90)
v = V.copy(); bad = v < 2.0; v[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), v[~bad])
spd = gauss(interp(v), 160)
# smooth the map polyline a little (GPS jitter), keep corners
mx = gauss(interp(X), 25); my = gauss(interp(Y), 25)
gain = np.concatenate([[0], np.cumsum(np.clip(np.diff(alt), 0, None))])
gain *= (600 / 3.28084) / gain[-1]       # lyric: six hundred to climb (m)
grade = np.gradient(alt, step)

# lyrics
def parse_srt(path):
    out = []
    for blk in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = blk.strip().split('\n'); m = re.match(r'(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)', ls[1])
        g = [int(x) for x in m.groups()]
        out.append((g[0]*3600+g[1]*60+g[2]+g[3]/1000, g[4]*3600+g[5]*60+g[6]+g[7]/1000, ' '.join(ls[2:]).strip()))
    return out
FIX = {'Kareem': 'Karim', 'lede': 'lead', 'Gro’venor': 'Grosvenor'}
lines, sections = [], []
for t0, t1, s in parse_srt('subs.srt'):
    for a, b in FIX.items(): s = s.replace(a, b)
    if s.startswith('['): sections.append([round(t0, 3), s.strip('[]').split(' —')[0].upper()])
    else: lines.append([round(t0, 3), round(t1, 3), s])

an = json.load(open('analysis.json'))
data = dict(
    dur=an['dur'], beats=[round(b, 3) for b in an['beats']],
    rms=an['rms'], ons=an['onset'], lines=lines, sections=sections,
    route=dict(step=step, n=len(sg), len=float(S[-1]), shift=round(SHIFT, 1),
               alt=[round(float(a), 2) for a in alt], spd=[round(float(a), 2) for a in spd],
               gain=[round(float(a), 1) for a in gain], grade=[round(float(a), 4) for a in grade],
               mx=[round(float(a), 1) for a in mx], my=[round(float(a), 1) for a in my]),
)
open('web/data.js', 'w').write('window.DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('n', len(sg), 'alt', alt.min().round(1), alt.max().round(1), 'spd', spd.min().round(1), spd.max().round(1),
      'lines', len(lines), 'sections', len(sections))
# where are the big climbs / fast stretches (km)?
g = np.gradient(gauss(alt, 120), step)
for k in range(0, len(sg), 50):
    pass
i = np.argmax(np.convolve(g, np.ones(60), 'same'))
print('steepest 600m window centred at km', round(sg[i] / 1000, 2), 'grade', round(float(np.mean(g[i-30:i+30])) * 100, 1), '%')
for km in range(0, 41, 2):
    j = int(km * 100); j = min(j, len(sg) - 1)
    print(f'km {km:2d} mi {km/1.609:4.1f} alt {alt[j]:5.1f} spd {spd[j]*2.237:4.1f}mph')
