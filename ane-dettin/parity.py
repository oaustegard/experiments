import sys, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
from dettin import Dettin
from ane_model import DettinANE
m = Dettin.load('pack', device='cpu')
cases = json.load(open('pack/cases.json'))
print('temps', m.temps, '| layers', m.enc.config.num_hidden_layers, 'hidden', m.enc.config.hidden_size, 'params %.1fM' % (sum(p.numel() for p in m.enc.parameters()) / 1e6))
items = []            # one (ids, mask positions, options, gold, type) per question
for c in cases:
    for n, q in c['questions'].items():
        ids, pos, opts = m._build(c['state'], q)
        g = c['gold'][n]['probabilities']
        items.append(dict(ids=ids, pos=pos, opts=opts, gold=max(g, key=g.get), type=q['type']))
lens = [len(i['ids']) for i in items]
print('questions', len(items), '| tokens: min %d median %d p95 %d max %d' % (min(lens), np.median(lens), np.percentile(lens, 95), max(lens)))
ref, ts = [], []
with torch.no_grad():
    for it in items:
        t = time.perf_counter()
        hs = m.enc(input_ids=torch.tensor([it['ids']]), attention_mask=torch.ones(1, len(it['ids']), dtype=torch.long)).last_hidden_state
        z = m.head(hs[0, it['pos']]).squeeze(-1) / m.temps.get(it['type'], 1.0)
        ts.append(time.perf_counter() - t); ref.append(torch.softmax(z, -1).numpy())
acc = np.mean([it['opts'][int(p.argmax())] == it['gold'] for it, p in zip(items, ref)])
print('reference (HF ModernBert, torch CPU, unpadded, one question per call): accuracy %.3f, median %.1f ms' % (acc, np.median(ts) * 1000))
S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
net = DettinANE(m.enc, m.head, S).eval()
pad = m.ids['[PAD]']; mx = 0; agree = 0; n = 0
with torch.no_grad():
    for it, r in zip(items, ref):
        if len(it['ids']) > S: continue
        ids = torch.full((1, S), pad, dtype=torch.long); ids[0, :len(it['ids'])] = torch.tensor(it['ids'])
        pm = torch.zeros(1, S); pm[0, :len(it['ids'])] = 1
        z = net(ids, pm)[0, it['pos']] / m.temps.get(it['type'], 1.0)
        p = torch.softmax(z, -1).numpy(); mx = max(mx, float(np.abs(p - r).max())); agree += int(p.argmax() == r.argmax()); n += 1
print('ANE-layout torch model at S=%d vs reference: %d questions fit, max |dp| %.2e, top-answer agreement %d/%d' % (S, n, mx, agree, n))
torch.save({'items': items, 'ref': ref, 'temps': m.temps, 'pad': pad}, 'items.pt')
