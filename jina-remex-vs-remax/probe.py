#!/usr/bin/env python3
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke
A = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(spoke("remax_kb"))); sys.path.insert(0, str(spoke("remax")/"src"))
from remex import Quantizer
from remax import StackedSignBitQuantizer
from remax_kb._hamming import hamming_scan, top_k

D = np.load(A/".vec_doc.npz")["m"]; Q = np.load(A/".vec_qry.npz")["m"]
print("D", D.shape, "Q", Q.shape, flush=True)

def t(msg, fn):
    s=time.time(); r=fn(); print(f"{msg}: {time.time()-s:.2f}s", flush=True); return r

for bits, dim in [(2,768),(4,768),(8,768),(4,512)]:
    qz = t(f"Quantizer(d={dim},bits={bits}) ctor", lambda: Quantizer(d=dim, bits=bits, seed=0))
    comp = t(f"  encode bits={bits} dim={dim}", lambda: qz.encode(np.ascontiguousarray(D[:,:dim])))
    t(f"  search1 bits={bits} dim={dim}", lambda: qz.search(comp, Q[0,:dim], k=len(D)))
print("--- remax ---", flush=True)
for dim,k in [(256,8),(512,4),(768,2)]:
    mean=D.mean(0).astype(np.float32)
    qz=t(f"remax ctor d{dim}k{k}", lambda: StackedSignBitQuantizer(d=dim,k=k,seed=0))
    dc=t(f"  remax enc d{dim}k{k}", lambda: qz.encode(np.ascontiguousarray((D-mean)[:,:dim])))
    qc=qz.encode(np.ascontiguousarray((Q-mean)[:,:dim]))
    t(f"  remax scan d{dim}k{k}", lambda: top_k(hamming_scan(dc,qc[0]),len(D)))
print("DONE", flush=True)
