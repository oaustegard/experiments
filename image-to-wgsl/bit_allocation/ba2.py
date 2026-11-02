import sys, json, re
sys.path.insert(0, "/tmp/claude-0/-home-user/d2e513c8-16db-55c6-8c70-8ad582e99f7b/scratchpad")
src = open("/tmp/claude-0/-home-user/d2e513c8-16db-55c6-8c70-8ad582e99f7b/scratchpad/ba.py").read().split("res = json.loads")[0]
exec(src)
OUT2 = ROOT / "results/bit_allocation_raw2.json"
def stripshape(sh):
    out = re.sub(r"    for \(var j = 0u; j < u32\(rd\([^\n]*\n", "", sh)
    assert out != sh
    return out
def measure2(run, tgt, drop_loop=False):
    o = {}
    for n in NS:
        img = codec.render_image(run["P"], run["bg"], run["H"], run["W"], run["A"], run["layout"], n=n)
        o[f"psnr{n}"] = codec.psnr8(img.numpy(), tgt)
        sh = codec.emit_wgsl(run, n=n)
        if drop_loop: sh = stripshape(sh)
        o[f"gz{n}"] = codec.rate(sh)["gz"]
    o["bits"] = run["layout"].bits
    return o
VARS = {
 "shape-drop": [("shape", 2)],
 "rec_A": [("shape", 2), ("colour", 2), ("sigma", 1)],
 "rec_B": [("shape", 2), ("colour", 3), ("sigma", 1)],
 "colour-3": [("colour", 3)],
}
res = {}
for im in IMS:
    run, _ = codec.load_run(ROOT / f"results/runs/{im}_prefix1024.json", ROOT / f"data/kodak/{im}.png")
    tgt = run["target"].numpy(); d = {}
    for name, steps in VARS.items():
        r = run
        for g, k in steps: r = coarsen(r, g, k)
        d[name] = measure2(r, tgt, drop_loop=any(g == "shape" for g, _ in steps))
    res[im] = d; OUT2.write_text(json.dumps(res)); print(im, "done", flush=True)
