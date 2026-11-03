# Sanity check for depth_probe.py: Parareal must be exact at K = S (here 4 segments of 16 layers). Output in results/depth_probe.json.
import sys, torch
sys.path.insert(0, '/home/user/experiments/monad-layer-parallel'); sys.path.insert(0, '/home/user/experiments/monad-weight-edit')
from depth_probe import Net, calib, kl, load, MWE
torch.set_num_threads(4)
tok, model = load(MWE / "models/A8p")
(ids, p), = calib(tok, "A8p__gpu_dev.jsonl", n=1, L=128)
with torch.no_grad():
    net = Net(model, torch.tensor([ids])); exact = net.logp(net.seq(net.h0, range(64)), p)
    m, S = 16, 4; seg = [range(s*m, (s+1)*m) for s in range(S)]
    U = [net.h0]
    for s in range(S): U.append(net.par(U[s], seg[s]))
    for K in range(1, 5):
        G = [net.seq(U[s], seg[s]) for s in range(S)]; Co = [net.par(U[s], seg[s]) for s in range(S)]
        V = [net.h0]
        for s in range(S): V.append(net.par(V[s], seg[s]) + G[s] - Co[s])
        U = V
        print('K', K, 'kl', round(kl(exact, net.logp(U[S], p)), 6), 'boundary errs', [round(float((U[s]-net.seq(net.h0, range(s*m))).norm()),1) for s in range(1, S+1)])
