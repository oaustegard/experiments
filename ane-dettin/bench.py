import sys, time, numpy as np, torch, warnings, collections, os
warnings.filterwarnings("ignore")
import coremltools as ct
S = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
d = torch.load('items.pt', weights_only=False); items, ref, temps, pad = d['items'], d['ref'], d['temps'], d['pad']
fit = [(it, r) for it, r in zip(items, ref) if len(it['ids']) <= S]
def feed(it):
    ids = np.full((1, S), pad, dtype=np.int32); ids[0, :len(it['ids'])] = it['ids']
    pm = np.zeros((1, S), dtype=np.float16); pm[0, :len(it['ids'])] = 1
    return {"input_ids": ids, "pad_mask": pm}
def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
path = f"dettin17m_S{S}.mlpackage"
size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6
print(f"S={S}: {len(fit)}/{len(items)} questions fit; package {size:.1f} MB")
for name, cu in (("CPU only", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU+Neural Engine", ct.ComputeUnit.CPU_AND_NE), ("All", ct.ComputeUnit.ALL)):
    t = time.time(); ml = ct.models.MLModel(path, compute_units=cu); load = time.time() - t
    f0 = feed(fit[0][0]); [ml.predict(f0) for _ in range(5)]
    ts = []
    for i in range(N):
        f = feed(fit[i % len(fit)][0]); t = time.perf_counter(); ml.predict(f); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    acc = agree = 0; mx = 0.0
    for it, r in fit:
        z = np.asarray(ml.predict(feed(it))["logits"], dtype=np.float32)[0, it['pos']] / temps.get(it['type'], 1.0)
        p = softmax(z); agree += int(p.argmax() == r.argmax()); acc += it['opts'][int(p.argmax())] == it['gold']; mx = max(mx, float(np.abs(p - r).max()))
    print(f"  {name:18s} load {load:5.1f}s | median {np.median(ts):6.2f} ms  p90 {np.percentile(ts, 90):6.2f}  min {ts.min():6.2f} | accuracy {acc / len(fit):.3f}  agrees with fp32 reference {agree}/{len(fit)}  max |dp| {mx:.3f}")
