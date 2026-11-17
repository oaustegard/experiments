"""Shot table. Times are the lyric/section cuts from v2; the camera is described in road coordinates relative to a
reference (a rider, the pack, or a fixed spot): eye/tgt = (ds forward, dl left, dz up) in metres. Numbers may be [a, b]
to ease from a to b across the shot.
"""
import math
import common as C

SHOTS = []
PROPS = []         # set pieces: dict(kind=..., s=..., t0, t1, ...)

# route positions (metres along the real ride) of the places the lyrics name; each is a straight stretch of real road
LM = dict(start=252, gate1=5716, amber=11388, walkers=13756, potholes=15116, gate2=16308, hill=25450, crest=26150,
          speed1=30908, speed2=33368, sprint=35664, red=37572, gate3=39664)
DEFAULT_LEAD = ['karim', 'phil', 'hulk', 'gary', 'angelo', 'me']


def S(t0, t1, kind='ride', **kw):
    d = dict(t0=t0, t1=t1, kind=kind)
    d.update(kw)
    SHOTS.append(d)
    return d


def cam(ref='pack', eye=(0, 8, 1.4), tgt=(0, 0, 1.15), lens=50, **kw):
    d = dict(ref=ref, eye=eye, tgt=tgt, lens=lens)
    d.update(kw)
    return d


def _n(x):
    return [-v for v in x] if isinstance(x, (list, tuple)) else -x


def SIDE(ref='pack', d=8.0, h=1.4, ds=0.0, lens=50, th=1.15, tds=0.0, **kw):
    return cam(ref, (ds, d, h), (tds, 0, th), lens, **kw)


def FRONT(ref='pack', ahead=12.0, side=0.8, h=1.0, lens=35, th=1.25, **kw):
    return cam(ref, (ahead, side, h), (0, 0, th), lens, **kw)


def CHASE(ref='pack', back=6.0, side=1.2, h=1.6, lens=35, ahead=10.0, th=1.0, **kw):
    return cam(ref, (_n(back), side, h), (ahead, 0, th), lens, **kw)


def DRONE(ref='pack', back=8.0, side=12.0, h=18.0, lens=35, tds=4.0, **kw):
    return cam(ref, (_n(back), side, h), (tds, 0, 0), lens, **kw)


def TOP(ref='pack', h=24.0, lens=28, ds=0.0, **kw):
    return cam(ref, (ds - 2.0, 0, h), (ds + 2.5, 0, 0), lens, **kw)


def arrive(t0, stagger=0.14, dur=2.4, dist=46):
    """Riders roll into the pack one after another from behind."""
    def hook(t, rid, k, film):
        a = 1 - min(1, max(0, (t - t0 - stagger * k) / dur))
        a = a * a
        return dict(ds=dist * a * (1 + 0.5 * (k % 2)))
    return hook


def gap_me(t, rid, k, film):
    """'Hanging on the back': Me gets gapped through the bridge and fights back."""
    if rid != 'me':
        return None
    g = 15.0 * C.smooth((t - 160) / 9) * (1 - C.smooth((t - 172.5) / 3.2))
    if 186.0 <= t < 198.5:
        g = 4.0 + 5.0 * math.sin((t - 186) * 0.35)
    return dict(ds=g)


def gate_at(s_gate, t0, t1, v, pass_t):
    """Anchor (head position at shot mid-time) so the head of the pack reaches the gate at pass_t."""
    return s_gate + v * (0.5 * (t0 + t1) - pass_t)


# =============================================================================================== INTRO
S(0, 9.5, 'ride', formation='two', v=3.0, at=LM['start'] + 60, sun=135, lead=['me', 'angelo', 'gary', 'hulk', 'phil', 'karim'],
  computer='me', hook=arrive(4.2, 0.1, 2.6),
  cam=cam('me', eye=(0.30, [-0.10, -0.06], 1.30), tgt=(0.6, 0.0, 1.03), lens=[40, 36], fstop=1.8, focus='tgt', shake=0.2))
S(9.5, 14.68, 'ride', formation='two', v=4.5, hook=arrive(9.5, 0.14, 2.6), sun=140, at=LM['start'] + 40,
  cam=cam('pack', eye=(26, 7.5, 1.1), tgt=(-4, -2, 1.1), lens=[26, 22], static=True, tref='pack', fstop=0))
S(14.68, 19.23, 'ride', sun=120, cam=SIDE('karim', d=[7.0, 5.5], h=1.75, ds=[2.0, 0.6], lens=[45, 60], th=0.95, fstop=3.5, focus='karim'))
S(19.23, 26.49, 'ride', sun=125, cam=cam('karim', eye=([1.0, -16], [3.0, 11], [1.3, 7.5]), tgt=([0, 5], 0, [1.2, 0.6]), lens=[50, 28]))
# =============================================================================================== VERSE 1
S(26.49, 30.08, 'ride', sun=150, cam=FRONT('pack', ahead=[22, 17], side=[0.9, 1.1], h=0.75, lens=[24, 28], th=1.1, shake=0.3), title=True)
S(30.08, 32.95, 'ride', sun=40, cam=FRONT('karim', ahead=[8, 5], side=[2.0, 1.5], h=1.35, lens=[35, 32], th=1.6, shake=0.4))
S(32.95, 36.38, 'ride', sun=105, lead=['hulk', 'karim', 'gary', 'phil', 'me', 'angelo'],
  cam=SIDE('hulk', d=[6.4, 5.4], h=1.75, ds=[0.8, 0.3], lens=[50, 65], th=0.95, fstop=3.5, focus='hulk', shake=0.3))
S(36.38, 39.57, 'ride', sun=60, lead=['hulk', 'karim', 'gary', 'phil', 'me', 'angelo'],
  cam=FRONT('hulk', ahead=[7.5, 5.0], side=[0.6, 1.1], h=[0.45, 0.6], lens=[32, 36], th=1.45))
S(39.57, 42.77, 'ride', sun=110, lead=['phil', 'karim', 'gary', 'hulk', 'angelo', 'me'],
  cam=FRONT('phil', ahead=[10, 7], side=[1.6, 1.0], h=1.3, lens=[40, 50], th=1.2, shake=0.3))
S(42.77, 46.28, 'ride', sun=100, lead=['phil', 'gary', 'karim', 'hulk', 'angelo', 'me'], slots={'phil': (3.0, -2.05), 'gary': (4.55, -2.05)},
  cam=SIDE('gary', d=[6.2, 5.4], h=0.9, ds=[2.2, 1.4], lens=[60, 70], th=0.8, tds=[0.9, 0.7], fstop=3.2, focus='gary'))
# GATE UP! (46.28-47.07) then the funnel
G1 = LM['gate1']
PROPS.append(dict(kind='gate', s=G1, t0=46.28, t1=56.0))
PROPS.append(dict(kind='strollers', s=G1 - 5, t0=46.28, t1=56.0))
S(46.28, 47.07, 'ride', sun=95, formation='zipper', gate_s=G1, v=8.5, at=gate_at(G1, 46.28, 56.0, 8.5, 48.6), lead=DEFAULT_LEAD,
  cam=cam('fixed:%d' % G1, eye=(-26, -1.6, 0.6), tgt=(0, -1.8, 1.2), lens=[26, 30], shake=0.6))
S(47.07, 51.70, 'ride', sun=95, formation='zipper', gate_s=G1, v=8.5, at=gate_at(G1, 46.28, 56.0, 8.5, 48.6),
  cam=cam('fixed:%d' % G1, eye=(-1.5, 0, [20, 11]), tgt=(1.5, 0, 0), lens=[30, 32]))
S(51.70, 55.77, 'ride', sun=95, formation='zipper', gate_s=G1, v=8.5, at=gate_at(G1, 46.28, 56.0, 8.5, 48.6),
  cam=cam('fixed:%d' % G1, eye=(-9, [-6.6, -6.2], 1.0), tgt=([4, 2], -2.0, 1.1), lens=[32, 38], fstop=4))
# =============================================================================================== CHORUS 1
S(55.77, 59.12, 'ride', sun=110, cam=SIDE('pack', d=10, h=1.2, lens=[28, 32], static=True, tref='pack'))
S(59.12, 62.31, 'ride', sun=145, lead=['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'], cam=FRONT('karim', ahead=[5.5, 4.0], side=[0.5, 0.9], h=0.28, lens=[22, 24], th=0.85, shake=0.5))
S(62.31, 66.06, 'ride', sun=95, lead=['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'], cam=SIDE('me', d=[7.5, 6.5], h=1.75, ds=[1.6, 0.8], lens=[50, 60], th=0.95, fstop=3.5, focus='me', shake=0.3))
S(66.06, 71.97, 'ride', sun=170, cam=cam('pack', eye=(30, [1.3, 1.5], 0.55), tgt=(-10, -0.5, 1.1), lens=[30, 38], static=True, tref='pack', fstop=0))
S(71.97, 75.00, 'map', head=[0.0, 1.0], tail=0.0, full=True, sun=120)
S(75.00, 78.40, 'ride', sun=110, cam=CHASE('pack', back=[5, 30], side=[0.6, 8], h=[1.6, 16], lens=[30, 30], ahead=[14, 6], th=[1.0, 0.0]))
S(78.40, 81.94, 'calendar', mode='weeks')
# =============================================================================================== VERSE 2
TL1 = LM['amber']
PROPS.append(dict(kind='light', s=TL1, t0=81.94, t1=88.4, states=[(0, 'green'), (82.4, 'amber'), (85.2, 'red')]))
S(81.94, 85.53, 'ride', sun=100, lead=['phil', 'karim', 'hulk', 'gary', 'angelo', 'me'], v=9.5, at=TL1 - 8,
  cam=cam('phil', eye=(-3, [9, 7], 1.3), tgt=([2, 0], -2, 2.6), lens=[28, 35], static=True, tref='phil', shake=0.2))
S(85.53, 88.40, 'ride', sun=140, v=11.5, cam=CHASE('karim', back=[3.5, 4.5], side=[0.9, 0.3], h=[0.8, 1.0], lens=[24, 28], ahead=[7, 12], th=1.0, shake=1.0))
WK = LM['walkers']
PROPS.append(dict(kind='walkers', s=WK + 6, t0=88.40, t1=95.2))
S(88.40, 91.68, 'ride', sun=140, v=7.5, at=WK + 2, avoid=(WK + 8.0, -1.0, 2.2),
  cam=cam('fixed:%d' % (WK + 15), eye=(0, -5.4, 1.5), tgt=(-9, -1.4, 1.0), lens=[30, 34], fstop=0))
S(91.68, 95.11, 'ride', sun=140, v=7.0, at=WK + 30, lead=['hulk', 'karim', 'gary', 'phil', 'me', 'angelo'], avoid=(WK + 8.0, -1.0, 2.2),
  cam=cam('fixed:%d' % (WK + 16), eye=(0, -5.0, 1.3), tgt=(-7, -1.6, 1.0), lens=[34, 38], fstop=0))
PH = LM['potholes']
PROPS.append(dict(kind='potholes', s=PH, t0=95.11, t1=101.7))
S(95.11, 97.66, 'ride', sun=140, v=7.5, at=PH + 20, cam=cam('karim', eye=([2.2, 1.6], [1.6, 1.2], 0.38), tgt=(-1.0, -0.4, 0.42), lens=[28, 32], shake=0.8))
S(97.66, 101.65, 'ride', sun=115, v=7.0, at=PH + 22, lead=['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'],
  cam=cam('fixed:%d' % (PH + 30), eye=(-4.0, -1.2, [30, 18]), tgt=(2.0, -1.2, 0), lens=[28, 32]))
# GATE 2
G2 = LM['gate2']
PROPS.append(dict(kind='gate', s=G2, t0=101.65, t1=111.2))
PROPS.append(dict(kind='scatter', s=G2 - 14, t0=101.65, t1=111.2))
S(101.65, 102.45, 'ride', sun=105, formation='zipper', gate_s=G2, v=8.0, at=gate_at(G2, 101.65, 111.2, 8.0, 103.8),
  cam=cam('fixed:%d' % G2, eye=(-24, -1.6, 0.6), tgt=(0, -1.8, 1.2), lens=[26, 30], shake=0.6))
S(102.45, 106.83, 'ride', sun=105, formation='zipper', gate_s=G2, v=8.0, at=gate_at(G2, 101.65, 111.2, 8.0, 103.8),
  cam=cam('fixed:%d' % G2, eye=(-1.5, 0.0, [20, 11]), tgt=(1.5, 0.0, 0.0), lens=[30, 32]))
S(106.83, 111.14, 'ride', sun=105, formation='zipper', gate_s=G2, v=8.0, at=gate_at(G2, 101.65, 111.2, 8.0, 103.8),
  cam=cam('fixed:%d' % (G2 + 12), eye=(-7, [3.0, 3.6], 1.0), tgt=([-6, -3], -2.0, 1.2), lens=[30, 36], fstop=4))
# =============================================================================================== VERSE 3: MORMON HILL
HILL = LM['hill']
PROPS.append(dict(kind='hillsign', s=HILL + 70, t0=111.1, t1=117.7))
S(111.14, 114.42, 'ride', sun=95, v=5.2, at=HILL + 110, hook=None,
  cam=cam('pack', eye=(-8, [6.5, 7.5], 2.2), tgt=(12, -1.5, 6.5), lens=[28, 34], static=True, tref='pack', fstop=0))
S(114.42, 117.69, 'ride', sun=80, v=4.8, at=HILL + 190, lead=['angelo', 'karim', 'hulk', 'phil', 'gary', 'me'],
  cam=SIDE('angelo', d=[6.0, 5.0], h=1.6, ds=[1.8, 0.9], lens=[50, 62], th=0.95, fstop=3.5, focus='angelo', shake=0.4))
S(117.69, 120.88, 'ride', sun=80, v=4.8, at=HILL + 260, lead=['angelo', 'karim', 'hulk', 'phil', 'gary', 'me'], items=True,
  cam=SIDE('angelo', d=[6.8, 5.8], h=1.7, ds=[1.0, 0.0], lens=[40, 46], th=1.5, fstop=4.0, focus='angelo', shake=0.3))
S(120.88, 124.15, 'ride', sun=90, v=5.0, at=HILL + 320, lead=['angelo', 'karim', 'hulk', 'phil', 'gary', 'me'],
  cam=DRONE('angelo', back=[8, 14], side=[8, 12], h=[7, 16], lens=34, tds=3))
CR = LM['crest']
S(124.15, 127.10, 'ride', sun=75, v=5.6, at=CR - 8, lead=['angelo', 'karim', 'hulk', 'phil', 'gary', 'me'], slots={'angelo': (-3.0, -1.9)},
  cam=cam('fixed:%d' % CR, eye=(-8, 6.5, 1.4), tgt=(2, -1.6, 1.3), lens=[34, 40], tref='angelo'))
S(127.10, 130.77, 'ride', sun=100, v=4.0, at=CR + 14, mute=True, cam=cam('pack', eye=(-26, [10, 12], 6.0), tgt=(2, 0, 1.4), lens=[60, 70], static=True, tref='pack', fstop=0))
# =============================================================================================== CHORUS 2
S(130.77, 134.04, 'map', tailback=0.05, sun=120)
S(134.04, 137.23, 'ride', sun=160, cam=FRONT('karim', ahead=[7, 5], side=[1.2, 0.6], h=0.5, lens=[26, 28], th=1.25, shake=0.4))
S(137.23, 143.70, 'ride', sun=120, lead=['phil', 'karim', 'hulk', 'gary', 'angelo', 'me'],
  cam=FRONT('phil', ahead=[7, 11], side=[2.2, 1.4], h=1.0, lens=[32, 40], th=1.2, shake=0.4))
S(143.70, 149.76, 'ride', sun=130, cam=cam('pack', eye=(0, 0, 42), tgt=(2.5, 0, 0), lens=34, orbit={'r': 26, 'a': [-80, 140]}))
S(149.76, 153.91, 'ride', sun=105, cam=SIDE('pack', d=[16, 12], h=2.5, lens=28, th=1.0, static=True, tref='pack'))
# =============================================================================================== BRIDGE
S(153.91, 163.16, 'map', tailback=0.07, sun=120)
SP1 = LM['speed1']
S(163.16, 166.36, 'ride', sun=140, v=11.5, at=SP1, cam=CHASE('karim', back=[3.0, 4.2], side=[0.9, 0.5], h=[0.9, 1.1], lens=[24, 28], ahead=[8, 12], th=1.0, shake=1.0))
S(166.36, 169.63, 'ride', sun=135, v=11.8, at=LM['speed2'], computer='me', lead=['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'],
  hook=gap_me, cam=cam('me', eye=(0.30, [-0.10, -0.06], 1.30), tgt=(0.6, 0.0, 1.03), lens=[40, 36], fstop=1.8, focus='tgt', shake=0.9))
S(169.63, 171.00, 'ride', sun=140, v=11.6, at=LM['speed2'] + 40, hook=gap_me, cam=CHASE('me', back=[2.5, 2.0], side=[0.5, 0.2], h=1.45, lens=30, ahead=[12, 14], th=1.0, shake=1.0))
S(171.00, 173.22, 'ride', sun=150, v=11.4, at=LM['speed2'] + 80, hook=gap_me, cam=cam('me', eye=([2.2, 1.7], [1.5, 1.2], 1.5), tgt=(0.1, 0, 1.6), lens=[60, 70], fstop=4.5, focus='me', shake=0.6))
S(173.22, 176.09, 'ride', sun=125, v=11.2, at=LM['speed2'] + 120, hook=gap_me, cam=SIDE('pack', d=[14, 12], h=1.8, lens=[30, 32], th=1.1, ds=[-14, -2], static=True, tref='pack'))
S(176.09, 184.23, 'calendar', mode='tuethu')
# =============================================================================================== VERSE 4
SPR = LM['sprint']
S(184.23, 186.46, 'ride', sun=110, v=9.5, at=SPR - 20, cam=SIDE('pack', d=[12, 10], h=1.6, lens=[30, 34], static=True, tref='pack'))
S(186.46, 192.13, 'ride', sun=140, v=[11.0, 13.5], at=SPR, hook=gap_me, cam=CHASE('karim', back=[3.5, 4.0], side=[0.8, 0.6], h=[1.0, 1.2], lens=[24, 28], ahead=[8, 12], th=1.0, shake=1.0))
S(192.13, 195.32, 'ride', sun=110, v=11.0, at=SPR + 50, hook=gap_me, cam=SIDE('me', d=[6.0, 5.0], h=1.5, ds=[1.3, 0.5], lens=[45, 55], th=0.85, fstop=3.5, focus='me', shake=0.7))
RED = LM['red']
PROPS.append(dict(kind='light', s=RED, t0=195.0, t1=201.9, states=[(0, 'green'), (195.0, 'amber'), (196.5, 'red'), (200.4, 'green')]))
S(195.32, 198.51, 'ride', sun=100, v=6.5, at=RED - 8, hook=gap_me, cam=cam('fixed:%d' % RED, eye=(-14, 7.0, 1.4), tgt=(-2, -2, 1.3), lens=[34, 40], tref='pack'))
# =============================================================================================== FINAL CHORUS (night)
S(198.51, 201.78, 'ride', sun=150, v=9.0, cam=FRONT('pack', ahead=[16, 12], side=[1.0, 0.8], h=0.8, lens=[26, 30], th=1.2, shake=0.4), lights=True)
S(201.78, 204.89, 'ride', sun=110, v=9.0, cam=SIDE('pack', d=[9, 8], h=1.3, lens=[34, 38], th=1.15, ds=[6, -2], static=True, tref='pack'), lights=True)
S(204.89, 211.28, 'ride', sun=140, v=[10.5, 12.0], cam=CHASE('karim', back=[4, 5], side=[0.9, 0.5], h=[1.0, 1.2], lens=[24, 28], ahead=[8, 12], th=1.0, shake=1.0), lights=True)
S(211.28, 214.07, 'ride', sun=130, v=9.5, cam=DRONE('pack', back=[10, 4], side=[16, 10], h=[20, 14], lens=32, tds=4), lights=True)
S(214.07, 216.54, 'ride', sun=135, v=9.0, computer='me', cam=cam('me', eye=(0.30, -0.08, 1.30), tgt=(0.6, 0.0, 1.03), lens=[40, 36], fstop=1.8, focus='tgt', shake=0.4), lights=True)
S(216.54, 224.28, 'map', head=1.0, tail=0.0, sun=120, loop=True)
# =============================================================================================== OUTRO
G3 = LM['gate3']
PROPS.append(dict(kind='gate', s=G3, t0=224.28, t1=229.5))        # gone before the last three shots, which ride the same stretch
S(224.28, 230.50, 'ride', sun=95, formation='zipper', gate_s=G3, v=7.5, at=gate_at(G3, 224.28, 232.0, 7.5, 226.0), lights=True,
  cam=cam('fixed:%d' % G3, eye=(-1.5, 0, [20, 12]), tgt=(1.5, 0, 0), lens=[30, 32]))
S(230.50, 233.94, 'ride', sun=125, v=7.5, at=G3 - 420, cam=FRONT('karim', ahead=[6, 4], side=[1.0, 0.6], h=1.2, lens=[40, 50], th=1.25, shake=0.3), lights=True)
S(233.94, 237.13, 'ride', sun=120, v=7.0, at=G3 - 380, cam=SIDE('pack', d=[10, 9], h=1.2, lens=[32, 34], th=1.1, static=True, tref='pack'), lights=True)
S(237.13, 260.0, 'ride', sun=135, v=[6.0, 3.0], at=G3 - 300, cam=CHASE('pack', back=[8, 40], side=[0.8, 3.0], h=[1.6, 12], lens=[30, 30], ahead=[20, 25], th=[1.0, 0.0]), lights=True, endcard=True)

SHOTS.sort(key=lambda d: d['t0'])


def shot_at(t):
    cur = SHOTS[0]
    for s in SHOTS:
        if t >= s['t0']:
            cur = s
        else:
            break
    return cur


def mood_over(t, rid):
    """Face state overrides (ported from v2's riderOver)."""
    o = {}
    if rid == 'gary' and 42.7 <= t < 46.3:
        o = dict(mood='squint')
    if rid == 'hulk' and 32.9 <= t < 39.6:
        o = dict(mood='grit' if t < 36.4 else 'smirk')
    if rid == 'me':
        if 160 <= t < 184.3:
            o = dict(mood='wide', stress=min(1.0, (t - 160) / 9))
        elif 186 <= t < 198.5:
            o = dict(mood='grit', stress=0.7)
    if rid in ('karim', 'phil') and 205 <= t < 211.3:
        o = dict(mood='grit')
    if rid == 'angelo':
        if 114 <= t < 125:
            o = dict(mood='grit', stress=0.8)
        elif 124.1 <= t < 131:
            o = dict(mood='smirk')
    if rid == 'karim' and 127.1 <= t < 130.8:
        o = dict(mood='neutral')
    return o
