"""Melody metrics for Strudel FM bar dumps (from gen.mjs). usage: python3 metrics.py a.json [b.json ...]"""
import json, re, sys
from collections import defaultdict, Counter
import statistics as S

MODES = dict(ionian=[0,2,4,5,7,9,11], dorian=[0,2,3,5,7,9,10], phrygian=[0,1,3,5,7,8,10], lydian=[0,2,4,6,7,9,11], mixolydian=[0,2,4,5,7,9,10], aeolian=[0,2,3,5,7,8,10])
PC = {'c':0,'d':2,'e':4,'f':5,'g':7,'a':9,'b':11}
def midi(t):
    m = re.fullmatch(r'([a-g])(#|b)?(-?\d)', t)
    return PC[m[1]] + (1 if m[2]=='#' else -1 if m[2]=='b' else 0) + 12*(int(m[3])+1)
def lead_tokens(code):
    if not code: return None
    s = re.match(r'note\("([^"]*)"\)', code)[1]
    if '*' in s or '[' in s: return 'arp'
    return s.split()
def voicing(code):
    if not code: return []
    m = re.search(r'\[([^\]]+)\]', code)
    return [midi(x) for x in m[1].split(',')] if m else []

def notes(bar):
    """[(step, midi, dur)] for a 16-token bar"""
    toks = bar['_lt']; out = []
    for i, t in enumerate(toks):
        if t in ('~', '_'): continue
        d = 1
        while i + d < 16 and toks[i + d] == '_': d += 1
        out.append((i, midi(t), d))
    return out

def run(path):
    rows = json.load(open(path))
    by = defaultdict(list)
    for r in rows: by[(r['style'], r['energy'], r['seed'])].append(r)
    agg = defaultdict(lambda: defaultdict(list))
    for (style, energy, seed), bars in by.items():
        bars.sort(key=lambda r: r['b'])
        A = agg[style]
        prev = None; seqs = []
        for r in bars:
            lt = lead_tokens(r['lead']); r['_lt'] = lt
            A['bars'].append(1)
            A['lead'].append(1 if isinstance(lt, list) else 0)
            if not isinstance(lt, list): prev = None; continue
            ns = notes(r); v = voicing(r['chords']); sc = {(r['key']['tonic'] + x) % 12 for x in MODES[r['key']['mode']]}
            tones = {(r['chord']['root'] + i) % 12 for i in r['chord']['iv']}
            A['notes'].append(len(ns))
            for s, m, d in ns:
                A['inkey'].append(m % 12 in sc)
                A['odd16'].append(s % 2 == 1)
                if s % 4 == 0: A['beat_ct'].append(m % 12 in tones)
                if s % 4 == 0 or d >= 2: A['rub'].append(any(abs(m - x) in (1, 13, 25) for x in v))
                A['dur'].append(d)
            # intervals, continuing across bars of the same phrase
            seq = ([prev] if prev is not None and r['bip'] > 0 else []) + [m for _, m, _ in ns]
            iv = [b - a for a, b in zip(seq, seq[1:])]
            for i, x in enumerate(iv):
                a = abs(x)
                A['step'].append(1 <= a <= 2); A['rep'].append(a == 0); A['leap'].append(a > 4); A['bigleap'].append(a > 7)
                if a > 4 and i + 1 < len(iv): A['recover'].append(iv[i+1] * x < 0 and abs(iv[i+1]) <= 4)
            prev = ns[-1][1] if ns else prev
            # a bar's shape: onsets + intervals from its first note (transposition-free)
            shape = (tuple(s for s, _, _ in ns), tuple(m - ns[0][1] for _, m, _ in ns)) if ns else None
            rhythm = tuple(s for s, _, _ in ns)
            seqs.append((r['b'], shape, rhythm, r))
        # recurrence: a bar's shape (or rhythm) recurs >=4 bars away in the same run
        for i, (b, sh, rh, r) in enumerate(seqs):
            if sh is None: continue
            A['shape_recur'].append(any(sh == sh2 and abs(b - b2) >= 4 for b2, sh2, _, _ in seqs))
            A['rhythm_recur'].append(any(rh == rh2 and abs(b - b2) >= 4 for b2, _, rh2, _ in seqs))
            A['rhythm_inphrase'].append(any(rh == rh2 and b != b2 and b2 // 4 == b // 4 for b2, _, rh2, _ in seqs))
        # phrase range and cadence
        phr = defaultdict(list)
        for b, sh, rh, r in seqs: phr[b // 4].append(r)
        for k, rs in phr.items():
            ms = [m for r in rs for _, m, _ in notes(r)]
            if len(ms) >= 3: A['range'].append(max(ms) - min(ms))
            last = [r for r in rs if r['bip'] == 3]
            if last and notes(last[0]):
                r = last[0]; s, m, d = notes(r)[-1]
                A['cad_root'].append((m - r['chord']['root']) % 12 in (0,))
                A['cad_long'].append(d >= 4)
    return agg

def pct(x): return f"{100*sum(x)/len(x):5.1f}" if x else "   - "
def main():
    cols = [('lead','lead%'),('notes','n/bar'),('odd16','odd16'),('step','step'),('rep','rep'),('leap','>4st'),('bigleap','>7st'),('recover','recov'),
            ('beat_ct','beatCT'),('rub','rub'),('inkey','inkey'),('rhythm_inphrase','rhyPh'),('rhythm_recur','rhyRec'),('shape_recur','shpRec'),('range','range'),('cad_root','cadRt'),('cad_long','cadLg')]
    for path in sys.argv[1:]:
        agg = run(path); print(f"\n== {path}")
        print(f"{'style':10}" + "".join(f"{h:>7}" for _, h in cols))
        tot = defaultdict(list)
        for style, A in agg.items():
            line = f"{style:10}"
            for k, h in cols:
                x = A[k]; tot[k] += x
                line += f"{S.mean(x):7.1f}" if k in ('notes','range') and x else f"{pct(x):>7}"
            print(line)
        print(f"{'ALL':10}" + "".join((f"{S.mean(tot[k]):7.1f}" if k in ('notes','range') and tot[k] else f"{pct(tot[k]):>7}") for k, _ in cols))
main()
