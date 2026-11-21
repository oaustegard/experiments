#!/usr/bin/env python3
"""Join foresight (Haiku, prompt only) and hindsight (Sonnet + Opus, outcome in view) labels with the turn records."""
import json, glob, collections, re, sys, math
C = collections.Counter
E = ['low', 'medium', 'high', 'xhigh']; TI = ['haiku', 'sonnet', 'opus', 'fable']
def load(d):
    out = {}
    for f in sorted(glob.glob(f'data/{d}/out_*.jsonl')):
        for l in open(f):
            l = l.strip()
            if not l: continue
            try: r = json.loads(l); out[r['id']] = r
            except Exception: pass
    return out
T = {t['id']: t for t in map(json.loads, open('data/turns_trace.jsonl')) if t['kind'] == 'human'}
F, S, O = load('fore'), load('hind_sonnet'), load('hind_opus')
main = lambda d: max(d, key=d.get) if d else None
print(f'turns {len(T)}  foresight {len(F)}  sonnet {len(S)}  opus {len(O)}')
ids = [i for i in T if i in S and i in O]
def agree(k): return sum(S[i].get(k) == O[i].get(k) for i in ids) / max(1, len(ids))
def within1(a, b, L=E): return a in L and b in L and abs(L.index(a) - L.index(b)) <= 1
print('\n== judge agreement (Sonnet vs Opus), n=%d' % len(ids))
for k in ('effort', 'tier', 'actual_effort', 'delegate', 'outcome'): print(f'  {k:14s} exact {agree(k):.2f}')
print('  effort within one level %.2f' % (sum(within1(S[i].get('effort'), O[i].get('effort')) for i in ids) / len(ids)))
def tab(pairs, rows, cols, title):
    c = C(pairs); print('\n' + title); print('  %-9s' % '' + ''.join('%8s' % x for x in cols) + '   total')
    for r in rows: print('  %-9s' % r + ''.join('%8d' % c[(r, x)] for x in cols) + '%8d' % sum(c[(r, x)] for x in cols))
gold = {i: S[i]['effort'] for i in ids if S[i].get('effort') == O[i].get('effort') and S[i].get('effort') in E}
print('\ngold effort (both judges agree): n=%d  %s' % (len(gold), dict(C(gold.values()))))
act = {i: main(T[i]['efforts']) for i in T}
tab([(act[i], gold[i]) for i in gold if act[i] in E], E, E, '== ran at (rows) vs needed in hindsight, gold (cols)')
g2 = [i for i in gold if act[i] in E]
d = [E.index(act[i]) - E.index(gold[i]) for i in g2]
print('  ran higher than needed %.2f | matched %.2f | ran lower than needed %.2f   (n=%d)' % (sum(x > 0 for x in d) / len(d), sum(x == 0 for x in d) / len(d), sum(x < 0 for x in d) / len(d), len(d)))
print('  by >=2 levels: higher %.2f lower %.2f' % (sum(x >= 2 for x in d) / len(d), sum(x <= -2 for x in d) / len(d)))
for name, J in (('sonnet', S), ('opus', O)): print(f'  actual_effort verdicts {name}:', dict(C(J[i].get('actual_effort') for i in ids)))
both_low = [i for i in ids if S[i].get('actual_effort') == O[i].get('actual_effort') == 'too_low']
both_high = [i for i in ids if S[i].get('actual_effort') == O[i].get('actual_effort') == 'too_high']
print('  both judges: too_low %d (%.1f%%)  too_high %d (%.1f%%)' % (len(both_low), 100 * len(both_low) / len(ids), len(both_high), 100 * len(both_high) / len(ids)))
# output tokens by need
def med(v): v = sorted(v); return v[len(v) // 2] if v else 0
print('\n== output tokens of the turn by gold need: ' + '  '.join('%s med %d sum %dk (n=%d)' % (e, med([T[i]['out_tokens'] for i in gold if gold[i] == e]), sum(T[i]['out_tokens'] for i in gold if gold[i] == e) / 1000, sum(gold[i] == e for i in gold)) for e in E))
tot = sum(T[i]['out_tokens'] for i in g2)
over = sum(T[i]['out_tokens'] for i in g2 if E.index(act[i]) > E.index(gold[i]))
print('  share of output tokens spent in turns that ran higher than needed: %.2f' % (over / tot))
# foresight
fi = [i for i in gold if i in F and F[i].get('effort') in E]
print('\n== Haiku foresight vs gold effort, n=%d' % len(fi))
tab([(F[i]['effort'], gold[i]) for i in fi], E, E, 'foresight (rows) vs gold (cols)')
acc = sum(F[i]['effort'] == gold[i] for i in fi) / len(fi); w1 = sum(within1(F[i]['effort'], gold[i]) for i in fi) / len(fi)
maj = C(gold[i] for i in fi).most_common(1)[0]
print('  exact %.3f  within-one %.3f  | always-"%s" baseline %.3f | ran-at-setting exact %.3f' % (acc, w1, maj[0], maj[1] / len(fi), sum(act[i] == gold[i] for i in fi) / len(fi)))
def prf(pred, truth, name):
    tp = sum(p and t for p, t in zip(pred, truth)); fp = sum(p and not t for p, t in zip(pred, truth)); fn = sum((not p) and t for p, t in zip(pred, truth))
    print('  %-34s base %.2f  fires %.2f  precision %.2f  recall %.2f  (tp %d fp %d fn %d)' % (name, sum(truth) / len(truth), sum(pred) / len(pred), tp / max(1, tp + fp), tp / max(1, tp + fn), tp, fp, fn))
hi = lambda e: e in ('high', 'xhigh')
print(' binary decisions:')
prf([hi(F[i]['effort']) for i in fi], [hi(gold[i]) for i in fi], 'needs high+ (escalate)')
prf([F[i]['effort'] == 'xhigh' for i in fi], [gold[i] == 'xhigh' for i in fi], 'needs xhigh')
prf([F[i]['effort'] == 'low' for i in fi], [gold[i] == 'low' for i in fi], 'low is enough (de-escalate)')
sub = [i for i in fi if act[i] in ('low', 'medium')]
prf([hi(F[i]['effort']) for i in sub], [hi(gold[i]) for i in sub], 'escalate, among turns run at <=medium')
conf = [i for i in fi if (F[i].get('conf') or 0) >= 0.8]
if conf: print('  conf>=0.8: n=%d exact %.3f' % (len(conf), sum(F[i]['effort'] == gold[i] for i in conf) / len(conf)))
# trivial baselines
def rule(t):
    n = len(t['prompt']); return 'low' if n < 40 else 'medium'
print('  length rule (<40 chars -> low else medium): exact %.3f' % (sum(rule(T[i]) == gold[i] for i in fi) / len(fi)))
# tier
gt = {i: S[i]['tier'] for i in ids if S[i].get('tier') == O[i].get('tier') and S[i].get('tier') in TI}
at = {i: next((t for t in TI if t in (main(T[i]['models']) or '')), None) for i in T}
tab([(at[i], gt[i]) for i in gt if at[i]], TI, TI, '== ran on (rows) vs least tier needed, gold (cols), n=%d' % len(gt))
ft = [i for i in gt if i in F and F[i].get('tier') in TI]
print('  Haiku foresight tier exact %.3f within-one %.3f (n=%d)' % (sum(F[i]['tier'] == gt[i] for i in ft) / len(ft), sum(within1(F[i]['tier'], gt[i], TI) for i in ft) / len(ft), len(ft)))
tab([(F[i]['tier'], gt[i]) for i in ft], TI, TI, 'foresight tier (rows) vs gold (cols)')
# delegation and outcome
print('\n== delegation (hindsight): sonnet', dict(C(S[i].get('delegate') for i in ids)), ' opus', dict(C(O[i].get('delegate') for i in ids)))
miss = [i for i in ids if S[i].get('delegate') == O[i].get('delegate') == 'missed']
print('  both say missed: %d (%.1f%%)' % (len(miss), 100 * len(miss) / len(ids)))
fd = [i for i in ids if i in F]
prf([F[i].get('delegate') in ('subagent', 'fanout') for i in fd], [(S[i].get('delegate') in ('missed', 'used_well')) and (O[i].get('delegate') in ('missed', 'used_well')) for i in fd], 'foresight delegate vs warranted')
print('== outcome: sonnet', dict(C(S[i].get('outcome') for i in ids)), ' opus', dict(C(O[i].get('outcome') for i in ids)))
bad = lambda i: S[i].get('outcome') in ('corrected', 'failed') and O[i].get('outcome') in ('corrected', 'failed')
print('  corrected/failed (both) by ran-vs-needed: ' + '  '.join('%s %d/%d' % (n, sum(bad(i) for i in g2 if f(E.index(act[i]) - E.index(gold[i]))), sum(1 for i in g2 if f(E.index(act[i]) - E.index(gold[i])))) for n, f in (('lower', lambda x: x < 0), ('matched', lambda x: x == 0), ('higher', lambda x: x > 0))))
json.dump({'gold_effort': gold, 'gold_tier': gt, 'too_low': both_low, 'too_high': both_high, 'missed_delegation': miss}, open('data/gold.json', 'w'))
# small-model feasibility: TF-IDF + logistic regression on the prompt text, 5-fold CV, against the same gold
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    X = [(T[i].get('prev_final') or '')[-300:] + ' ||| ' + T[i]['prompt'][:2000] for i in fi]; y = [gold[i] for i in fi]
    for name, yy in (('4-way effort', y), ('needs high+', [str(hi(v)) for v in y])):
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, C=4, class_weight='balanced'))
        p = cross_val_predict(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        print('\n== TF-IDF + logistic regression, 5-fold CV, %s: accuracy %.3f' % (name, float(np.mean(np.array(p) == np.array(yy)))))
        if name != '4-way effort': prf([v == 'True' for v in p], [v == 'True' for v in yy], 'TF-IDF LR escalate')
    hk = [str(hi(F[i]['effort'])) for i in fi]
except Exception as e: print('sklearn step skipped:', e)
