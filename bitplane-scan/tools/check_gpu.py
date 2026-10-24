"""Run web/index.html in headless Chromium (SwiftShader WebGPU) against test vectors
exported by `reference.py --export DIR`; checks GPU and JS-CPU top-K and per-pass survivor
counts against numpy. Usage: python3 tools/check_gpu.py web/index.html DIR [demo-json] [chrome]"""
import sys, json, numpy as np
from playwright.sync_api import sync_playwright
CH = sys.argv[4] if len(sys.argv) > 4 else "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"
import os
page_path, tv = os.path.abspath(sys.argv[1]), sys.argv[2]
m=json.load(open(tv+"/meta.json"))
xq=np.fromfile(tv+"/xq.i8",np.int8).tolist(); s=np.fromfile(tv+"/s.f32",np.float32).tolist(); q=np.fromfile(tv+"/q.i8",np.int8).tolist()
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=CH,args=["--no-sandbox","--enable-unsafe-webgpu","--use-angle=swiftshader","--use-webgpu-adapter=swiftshader","--enable-features=Vulkan"])
    pg=b.new_page(); errs=[]
    pg.on("console", lambda msg: errs.append(f"[{msg.type}] {msg.text}") if msg.type in ("error","warning") else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.set_default_timeout(0)
    pg.add_init_script("window.BPS_NO_AUTORUN=true"); pg.goto("file://"+page_path)
    r=pg.evaluate("(a)=>window.BPS.testVectors(a)", dict(xq=xq,s=s,q=q,N=m["N"],D=m["D"],K=m["K"],nq=m["nq"]))
    print("adapter", r["adapter"], "dot4Native", r["dot4Native"])
    ok=True
    for c,o in zip(m["cases"], r["out"]):
        line=dict(py_vs_jscpu_top=c["topk"]==o["cpuTop"], py_vs_jsbrute=c["topk"]==o["cpuBrute"], py_vs_js_counts=c["n_out"]==o["cpuOut"],
                  gpu_top=o.get("gpuTop")==c["topk"], gpu_brute=o.get("gpuBrute")==c["topk"], gpu_counts=o.get("gpuOut")==c["n_out"])
        ok &= all(line.values()); print(line, c["n_out"], o.get("gpuOut"))
    print("ALL OK" if ok else "MISMATCH")
    if len(sys.argv)>3 and sys.argv[3]:
        d=pg.evaluate("(o)=>window.BPS.demo(o)", json.loads(sys.argv[3])); print(d["status"]); print(d["verdict"]); print(d["summary"]); print(d["table"])
    print("\n".join(errs[:20])); b.close()
