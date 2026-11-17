"""Shared data for the Blender cut: route, song-time -> distance mapping, lyrics, beats.

Pure numpy (no bpy) so it can be imported by Blender's python and by the overlay tooling.
The anchor table is the one v2 used (song second -> metres along the real Strava ride).
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'data.json')))
FPS = 30
DUR = D['dur']
NF = int(DUR * FPS)

R = D['route']
STEP = R['step']
MX = np.array(R['mx']); MY = np.array(R['my']); ALT = np.array(R['alt']); SPD = np.array(R['spd'])
N = len(MX)
ELEV_EX = 2.0           # vertical exaggeration of the real terrain
S_GRID = np.arange(N) * STEP
LEN = float(R['len'])

BEATS = np.array(D['beats'])
RMS = np.array(D['rms'])
LINES = [dict(t0=a, t1=b, text=s) for a, b, s in D['lines']]
SECTIONS = D['sections']

# ---- song time -> metres along the ride --------------------------------------------------------------
_ANCH0 = [[0, 0], [19, 0], [26.5, 400], [46.3, 4500], [55.8, 6500], [81.9, 11800], [91, 14500], [96.5, 15300],
          [101.6, 17000], [109.4, 25500], [111.1, 25900], [124.1, 26750], [130.8, 27400], [153.9, 30300], [166.4, 33300],
          [169.6, 34000], [184.2, 35100], [198.5, 37800], [205, 38400], [214, 39500], [232, 40495], [400, 40495]]
SHIFT = R['shift']
_ANCH = [[t, max(0.0, s - SHIFT)] for t, s in _ANCH0]
_ANCH[2][1] = 300.0


def _build_odo():
    n = NF + 90
    rate = np.zeros(n)
    for f in range(n):
        t = f / FPS
        i = 1
        while i < len(_ANCH) - 1 and _ANCH[i][0] < t:
            i += 1
        (ta, sa), (tb, sb) = _ANCH[i - 1], _ANCH[i]
        rate[f] = (sb - sa) / (tb - ta)
    k = 36
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.pad(rate, k, mode='edge')
    r2 = np.convolve(pad, ker, mode='valid')
    odo = np.concatenate([[0], np.cumsum(r2[:-1] / FPS)])
    sc = LEN / odo[int(round(232 * FPS))]
    return odo * sc, r2 * sc


ODO, RATE = _build_odo()


def odo(t):
    f = min(max(t * FPS, 0), len(ODO) - 1.001)
    i = int(f)
    return ODO[i] + (ODO[i + 1] - ODO[i]) * (f - i)


def odo_rate(t):
    f = min(max(int(round(t * FPS)), 0), len(RATE) - 1)
    return RATE[f]


# ---- route sampling ----------------------------------------------------------------------------------
def _interp(arr, s):
    f = min(max(s / STEP, 0), N - 1.001)
    i = int(f)
    return arr[i] + (arr[i + 1] - arr[i]) * (f - i)


def route_xy(s):
    return _interp(MX, s), _interp(MY, s)


def route_alt(s):
    return _interp(ALT, s)


def route_z(s):
    return (route_alt(s) - ALT[0]) * ELEV_EX


def route_heading(s, ahead=6.0):
    """Unit tangent (x, y) of the road at s, from a short chord."""
    x0, y0 = route_xy(max(0, s - ahead)); x1, y1 = route_xy(min(LEN, s + ahead))
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    return dx / L, dy / L


def route_speed(s):
    a = 0.0
    for k in range(-3, 4):
        a += _interp(SPD, s + k * 120)
    return a / 7


def beat_pulse(t, tau=0.13):
    i = int(np.searchsorted(BEATS, t, side='right')) - 1
    return math.exp(-(t - BEATS[i]) / tau) if i >= 0 else 0.0


def energy(t):
    return float(RMS[min(len(RMS) - 1, max(0, int(round(t * FPS))))])


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def section_at(t):
    name = ''
    for t0, n in SECTIONS:
        if t >= t0:
            name = n
    return name
