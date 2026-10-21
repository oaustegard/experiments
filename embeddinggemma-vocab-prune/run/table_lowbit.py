"""3-bit and 2-bit token tables for the #103 cut students, unpacked with standard ONNX ops so
the graph runs on onnxruntime-web's wasm backend.

Usage: python3 -I run/table_lowbit.py --k 4 [--bits 3 2] [--smoke]

1. Exports the 10-epoch Mac build (release eg2-layer-mac-20261020) the way
   eg2-layer-student/run/export.py does: fp32 ONNX, then q4 (MatMulNBits, block 32,
   asymmetric) with the table as GatherBlockQuantized. Into .work/export/mac_k<k>/.
2. Quantizes the fp32 table (41,741 x 512) to b bits, asymmetric min/max per group of 32,
   with fp16 scale and fp16 minimum per group, and stores the codes as bit planes: the low
   two bits four to a byte (128 bytes a row), and for b = 3 the high bit eight to a byte
   (64 bytes a row). 192 bytes a row at 3 bits and 128 at 2, plus 64 bytes of scale and
   minimum. Every unpack op is uint8 Gather / BitShift / BitwiseAnd, then Cast / Mul / Add.
3. Replaces the q4 graph's GatherBlockQuantized with that unpack (model_t<b>w.onnx). The
   transformer stays MatMulNBits, as in eg2-layer-student's model_q4w.onnx.
4. Checks: the unpack alone against a numpy dequantization of the same bytes (must be exact);
   the whole graph on onnxruntime CPU, scored on the #103 bench (ASYM: real_test top-10
   overlap, synth_test hit@10); with --smoke, 32 queries in onnxruntime-web 1.30 wasm in
   headless Chromium against onnxruntime CPU. Writes results/table_lowbit_k<k>.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LS = ROOT.parent / "eg2-layer-student"
sys.path.insert(0, str(LS / "run"))
from bench import TESTS, Bench  # noqa: E402
from export import Graph, onnx_bytes, ort_embed, pack_gather, save  # noqa: E402
from model import Student, tokenize  # noqa: E402
from common import Teacher  # noqa: E402

VOCAB = LS / "out" / "vocab" / "k32000"
G = 32


def export(k, d):
    """fp32 and q4 ONNX of the Mac build, as eg2-layer-student/run/export.py builds them."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    rows = json.loads((VOCAB / "rows.json").read_text())
    s = Student(k, rows)
    sd = torch.load(ROOT / ".work" / "mac" / "out" / "l1" / f"mac_k{k}_v32k" / "final.pt")
    assert not s.load_state_dict(sd, strict=False).unexpected_keys
    s.tm.config._attn_implementation = "eager"
    s.eval()
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("model*"):
        old.unlink()
    fp32 = d / "model.onnx"
    ids = torch.ones((2, 9), dtype=torch.long)
    mask = torch.ones((2, 9), dtype=torch.long)
    mask[1, 6:] = 0
    with torch.no_grad():
        torch.onnx.export(Graph(s), (ids, mask), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["embedding"], opset_version=21, dynamo=False,
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                        "embedding": {0: "b"}})
    save(onnx.load(str(fp32)), fp32)
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=4, block_size=32, is_symmetric=False,
                             op_types_to_quantize=("MatMul", "Gather"), quant_axes=(("MatMul", 0), ("Gather", 1)))
    q.process()
    save(pack_gather(q.model.model), d / "model_q4.onnx")
    (d / "tokenizer.json").write_bytes((VOCAB / "tokenizer.json").read_bytes())


def table_from_fp32(path):
    m = onnx.load(str(path))
    I = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Gather" and n.input[0] in I and len(I[n.input[0]].dims) == 2:
            return numpy_helper.to_array(I[n.input[0]]).astype(np.float32)
    raise ValueError("no table Gather in " + str(path))


def quantize(W, bits):
    """Codes 0 .. 2^bits - 1 per value; fp16 scale and minimum per group of 32. The codes are
    rounded against the fp16 scale and minimum, so dequantizing the stored bytes is exact."""
    V, H = W.shape
    Gw = W.reshape(V, H // G, G)
    lo = Gw.min(2).astype(np.float16)
    s = ((Gw.max(2) - lo.astype(np.float32)) / (2 ** bits - 1)).astype(np.float16)
    s[s == 0] = np.float16(1)
    q = np.clip(np.round((Gw - lo[..., None].astype(np.float32)) / s[..., None].astype(np.float32)), 0, 2 ** bits - 1)
    return q.astype(np.uint8).reshape(V, H), s, lo


def dequant(q, s, lo):
    V, H = q.shape
    return (q.reshape(V, H // G, G).astype(np.float32) * s[..., None].astype(np.float32)
            + lo[..., None].astype(np.float32)).reshape(V, H)


def pack(q, bits):
    """Plane A: bits 0-1, four codes a byte, code j of the byte in bits 2j..2j+1.
    Plane B (3-bit only): bit 2, eight codes a byte, code j in bit j."""
    V, H = q.shape
    a = (q & 3).reshape(V, H // 4, 4)
    planes = [(a[..., 0] | a[..., 1] << 2 | a[..., 2] << 4 | a[..., 3] << 6).astype(np.uint8)]
    if bits == 3:
        b = (q >> 2 & 1).reshape(V, H // 8, 8)
        planes.append(np.bitwise_or.reduce(b << np.arange(8, dtype=np.uint8), axis=2).astype(np.uint8))
    return planes


def unpack_nodes(g, ids, out, planes, s, lo, H, p):
    """Standard-op unpack of the rows of `ids` into `out` [.., H] float32; adds initializers to g."""
    def c(name, arr):
        g.initializer.append(numpy_helper.from_array(np.asarray(arr), p + name))
        return p + name
    nodes = []
    ax = c("ax", np.array([-1], np.int64))
    shH = c("shH", np.array([0, 0, H], np.int64))
    shG = c("shG", np.array([0, 0, H // G, G], np.int64))

    def plane(arr, tag, width, step):
        src = c(tag, arr)
        sh = c(tag + "sh", (np.arange(8 // width, dtype=np.uint8) * width).astype(np.uint8))
        mk = c(tag + "mk", np.uint8(2 ** width - 1))
        t = p + tag
        nodes.extend([helper.make_node("Gather", [src, ids], [t + "g"], axis=0),
                      helper.make_node("Unsqueeze", [t + "g", ax], [t + "u"]),
                      helper.make_node("BitShift", [t + "u", sh], [t + "s"], direction="RIGHT"),
                      helper.make_node("BitwiseAnd", [t + "s", mk], [t + "m"]),
                      helper.make_node("Reshape", [t + "m", shH], [t + "r"]),
                      helper.make_node("Cast", [t + "r"], [t + "f"], to=TensorProto.FLOAT)])
        if step != 1:
            nodes.append(helper.make_node("Mul", [t + "f", c(tag + "k", np.float32(step))], [t + "x"]))
            return t + "x"
        return t + "f"

    v = plane(planes[0], "pa", 2, 1)
    if len(planes) == 2:
        vb = plane(planes[1], "pb", 1, 4)
        nodes.append(helper.make_node("Add", [v, vb], [p + "v"]))
        v = p + "v"

    def grp(arr, tag):
        src = c(tag, arr)
        nodes.extend([helper.make_node("Gather", [src, ids], [p + tag + "g"], axis=0),
                      helper.make_node("Cast", [p + tag + "g"], [p + tag + "f"], to=TensorProto.FLOAT),
                      helper.make_node("Unsqueeze", [p + tag + "f", ax], [p + tag + "u"])])
        return p + tag + "u"
    su, lu = grp(s, "sc"), grp(lo, "lo")
    nodes.extend([helper.make_node("Reshape", [v, shG], [p + "vg"]),
                  helper.make_node("Mul", [p + "vg", su], [p + "vs"]),
                  helper.make_node("Add", [p + "vs", lu], [p + "vl"]),
                  helper.make_node("Reshape", [p + "vl", shH], [out])])
    return nodes


def check_unpack(planes, s, lo, ref, d):
    """The unpack alone, in a two-node-input graph, against numpy on random ids."""
    import onnxruntime as ort
    V, H = ref.shape
    g = helper.make_graph([], "unpack", [helper.make_tensor_value_info("ids", TensorProto.INT64, ["b", "s"])],
                          [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["b", "s", H])])
    g.node.extend(unpack_nodes(g, "ids", "rows", planes, s, lo, H, "t_"))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m)
    ids = np.random.default_rng(0).integers(0, V, (64, 33))
    got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"ids": ids})[0]
    return float(np.abs(got - ref[ids]).max())


def replace_table(q4_path, dst, planes, s, lo):
    m = onnx.load(str(q4_path))
    g = m.graph
    nodes, done, drop = [], False, set()
    for n in g.node:
        if n.op_type == "GatherBlockQuantized":
            assert not done
            H = planes[0].shape[1] * 4
            nodes += unpack_nodes(g, n.input[1], n.output[0], planes, s, lo, H, "tlb_")
            drop |= {n.input[0], n.input[2], n.input[3]}
            done = True
        else:
            nodes.append(n)
    assert done
    del g.node[:]
    g.node.extend(nodes)
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep)
    for old in dst.parent.glob(dst.name + "*"):
        old.unlink()
    save(m, dst)


def smoke(d, fname, enc):
    """32 queries in onnxruntime-web 1.30 wasm (headless Chromium) against onnxruntime CPU."""
    from playwright.sync_api import sync_playwright
    REPO = ROOT.parent
    sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
    from run_demo import CHROME, CHROME_ARGS, free_port, serve_from_cache, wait_for_server
    (d / "smoke_ids.json").write_text(json.dumps(enc))
    ref = ort_embed(d / fname, enc)
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rel = "/" + str(d.relative_to(REPO)) + "/"
    try:
        page_url = f"http://127.0.0.1:{port}/eg2-layer-student/web/smoke.html"
        wait_for_server(page_url)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            page = b.new_page()
            page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
            page.goto(f"{page_url}?model={rel}&ids={rel}smoke_ids.json&ep=wasm&file={fname}")
            page.wait_for_function("window.result.phase !== 'loading'", timeout=900_000)
            st = page.evaluate("window.result")
            b.close()
    finally:
        server.terminate()
    if st["phase"] != "done":
        return {"phase": st["phase"], "error": st.get("error", "")[:2000]}
    V = np.array(st["vectors"], np.float32)
    cos = (V * ref).sum(1) / np.linalg.norm(V, axis=1)
    return {"phase": "done", "n": len(enc), "cos_to_cpu_min": float(cos.min()), "cos_to_cpu_mean": float(cos.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 2])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = ROOT / ".work" / "export" / f"mac_k{a.k}"
    if not (d / "model_q4.onnx").exists():
        export(a.k, d)
    T, B = Teacher(), Bench()
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    resp = ROOT / "results" / f"table_lowbit_k{a.k}.json"
    res = json.loads(resp.read_text()) if resp.exists() else {"k": a.k, "formats": {}}
    q4 = {t: ort_embed(d / "model_q4.onnx", enc[t]) for t in TESTS}
    if "q4" not in res["formats"]:
        r, _ = B.score(q4)
        res["formats"]["q4"] = {"onnx_bytes": onnx_bytes(d / "model_q4.onnx"), **r}
        print("q4", onnx_bytes(d / "model_q4.onnx"), r["real_test"]["overlap@10"][0], r["synth_test"]["hit@10"][0], flush=True)
    W = table_from_fp32(d / "model.onnx")
    for bits in a.bits:
        qc, s, lo = quantize(W, bits)
        planes = pack(qc, bits)
        err = check_unpack(planes, s, lo, dequant(qc, s, lo), d)
        dst = d / f"model_t{bits}w.onnx"
        replace_table(d / "model_q4.onnx", dst, planes, s, lo)
        P = {t: ort_embed(dst, enc[t]) for t in TESTS}
        r, per = B.score(P)
        cos = np.concatenate([(P[t] * q4[t]).sum(1) for t in TESTS])
        f = {"onnx_bytes": onnx_bytes(dst), "table_bytes": int(sum(x.nbytes for x in planes) + s.nbytes + lo.nbytes),
             "unpack_max_abs_err": err, "cos_to_q4": {"mean": float(cos.mean()), "min": float(cos.min())}, **r}
        if a.smoke:
            f["wasm"] = smoke(d, dst.name, enc["synth_test"][:32])
        res["formats"][f"t{bits}w"] = f
        resp.write_text(json.dumps(res, indent=1))
        print(f"t{bits}w", f["onnx_bytes"], "unpack err", err, r["real_test"]["overlap@10"][0],
              r["synth_test"]["hit@10"][0], json.dumps(f.get("wasm", {})), flush=True)


if __name__ == "__main__":
    main()
