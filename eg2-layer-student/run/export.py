"""Ship a student: ONNX export, q4 and int8 quantization, and the bench on the ONNX itself.

Usage: python3 -I run/export.py --k 4 [--vocab out/vocab/k32000] [--weights out/l1/k4/final.pt] --tag NAME

The graph takes input_ids and attention_mask (int64) and returns `embedding` (unit 768-d),
pooling and normalization included. Queries are at most 128 tokens and the sliding window
is 512, so every layer gets the same bidirectional padding mask. q4 = onnxruntime
MatMulNBits / GatherBlockQuantized, 4-bit, block 32, asymmetric (EG2's own q4 format);
int8 = onnxruntime dynamic quantization of the MatMuls and the Gather. Download = ONNX bytes (graph + external data) + gzip tokenizer.json.
Writes out/export/<tag>/ and results/export_<tag>.json.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TESTS, Bench  # noqa: E402
from model import ROOT, Student, embed, tokenize  # noqa: E402
from common import HF, Teacher  # noqa: E402


class Graph(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, input_ids, attention_mask):
        keep = attention_mask[:, None, None, :].bool()
        add = torch.zeros(keep.shape, dtype=torch.float32).masked_fill(~keep, torch.finfo(torch.float32).min)
        h = self.s.tm(input_ids=input_ids, attention_mask={"full_attention": add, "sliding_attention": add}).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(e, dim=-1)


def onnx_bytes(p):
    # model.onnx_data from onnx.save, model.onnx.data from quantize_dynamic
    return p.stat().st_size + sum(x.stat().st_size for x in p.parent.glob(p.name + "?data"))


def save(model, p):
    onnx.save(model, str(p), save_as_external_data=True, all_tensors_to_one_file=True,
              location=p.name + "_data", size_threshold=1024)


def pack_gather(model):
    """Repack the 4-bit token table into EG2's own layout. MatMulNBitsQuantizer stores it
    as UINT4 tensors, which onnxruntime-web's wasm build has no GatherBlockQuantized kernel
    for (WebGPU runs it); EG2's model_q4.onnx stores uint8 bytes, two values per byte with
    the low nibble first, and sets bits=4. Values are unchanged."""
    I = {i.name: i for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type != "GatherBlockQuantized":
            continue
        for name in (n.input[0], n.input[3]):
            a = numpy_helper.to_array(I[name]).astype(np.uint8)
            a = a.reshape(a.shape[0], -1)
            if a.shape[1] % 2:
                a = np.pad(a, ((0, 0), (0, 1)))
            I[name].CopyFrom(numpy_helper.from_array((a[:, 0::2] | (a[:, 1::2] << 4)).astype(np.uint8), name))
        if not any(x.name == "bits" for x in n.attribute):
            n.attribute.append(onnx.helper.make_attribute("bits", 4))
    return model


def unpack_gather(model):
    """Replace each GatherBlockQuantized (no onnxruntime-web wasm kernel) by standard ops on
    the same bytes: Gather the packed rows, zero points and scales, split nibbles with
    BitwiseAnd/BitShift (low nibble = even column), Cast, subtract the zero point, multiply by
    the block scale. Reshape uses 0 to copy the dynamic batch and sequence dims."""
    from onnx import helper, TensorProto
    g = model.graph
    I = {i.name: i for i in g.initializer}
    out_nodes = []
    for n in g.node:
        if n.op_type != "GatherBlockQuantized":
            out_nodes.append(n)
            continue
        q, ids, sc, zp = n.input
        block = next(a.i for a in n.attribute if a.name == "block_size")
        cols = I[q].dims[1] * 2
        nb = cols // block
        p = n.name.replace("/", "_") + "_u"

        def c(name, arr):
            g.initializer.append(numpy_helper.from_array(np.array(arr), p + name))  # returns the message, not None
            return p + name
        k15, k4 = c("15", np.uint8(15)), c("4", np.uint8(4))
        sh_cols, sh_blk, sh_nb = c("shc", np.array([0, 0, cols], np.int64)), \
            c("shb", np.array([0, 0, nb, block], np.int64)), c("shn", np.array([0, 0, nb], np.int64))
        ax = c("ax", np.array([-1], np.int64))

        def nibbles(src, shape, tag):
            g_ = p + tag
            return [helper.make_node("Gather", [src, ids], [g_ + "g"], axis=0),
                    helper.make_node("BitwiseAnd", [g_ + "g", k15], [g_ + "lo"]),
                    helper.make_node("BitShift", [g_ + "g", k4], [g_ + "hi"], direction="RIGHT"),
                    helper.make_node("Unsqueeze", [g_ + "lo", ax], [g_ + "lo1"]),
                    helper.make_node("Unsqueeze", [g_ + "hi", ax], [g_ + "hi1"]),
                    helper.make_node("Concat", [g_ + "lo1", g_ + "hi1"], [g_ + "cat"], axis=-1),
                    helper.make_node("Reshape", [g_ + "cat", shape], [g_ + "r"]),
                    helper.make_node("Cast", [g_ + "r"], [g_ + "f"], to=TensorProto.FLOAT)], g_ + "f"
        nq, vq = nibbles(q, sh_cols, "q")
        nz, vz = nibbles(zp, sh_nb, "z")
        out_nodes += nq + nz + [
            helper.make_node("Reshape", [vq, sh_blk], [p + "vb"]),
            helper.make_node("Unsqueeze", [vz, ax], [p + "z1"]),
            helper.make_node("Sub", [p + "vb", p + "z1"], [p + "d"]),
            helper.make_node("Gather", [sc, ids], [p + "s"], axis=0),
            helper.make_node("Unsqueeze", [p + "s", ax], [p + "s1"]),
            helper.make_node("Mul", [p + "d", p + "s1"], [p + "m"]),
            helper.make_node("Reshape", [p + "m", sh_cols], [n.output[0]])]
    del g.node[:]
    g.node.extend(out_nodes)
    return model


def ort_embed(path, enc, batch=128):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = np.empty((len(enc), 768), np.float32)
    order = np.argsort([len(e) for e in enc])
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        S = max(len(enc[r]) for r in rows)
        ids = np.zeros((len(rows), S), np.int64)
        mask = np.zeros((len(rows), S), np.int64)
        for b, r in enumerate(rows):
            ids[b, :len(enc[r])] = enc[r]
            mask[b, :len(enc[r])] = 1
        out[rows] = sess.run(["embedding"], {"input_ids": ids, "attention_mask": mask})[0]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def wasm_variant(a):
    d = ROOT / "out" / "export" / a.tag
    src, dst = d / "model_q4.onnx", d / "model_q4w.onnx"
    for old in d.glob("model_q4w.onnx*"):
        old.unlink()
    save(unpack_gather(onnx.load(str(src))), dst)
    T, B = Teacher(), Bench()
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    res = json.loads((ROOT / "results" / f"export_{a.tag}.json").read_text())
    enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    P = {t: ort_embed(dst, enc[t]) for t in TESTS}
    ref = {t: ort_embed(src, enc[t]) for t in TESTS}
    cos = np.concatenate([(P[t] * ref[t]).sum(1) for t in TESTS])
    r, _ = B.score(P)
    tok_gz = res["tokenizer_gz_bytes"]
    res["formats"]["q4w"] = {"onnx_bytes": onnx_bytes(dst), "bytes": onnx_bytes(dst) + tok_gz,
                             "cos_to_q4": {"mean": float(cos.mean()), "min": float(cos.min())},
                             "note": "model_q4.onnx with GatherBlockQuantized replaced by standard ops (unpack_gather)", **r}
    (ROOT / "results" / f"export_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(a.tag, "q4w", f"{(onnx_bytes(dst) + tok_gz) / 1e6:.1f} MB", f"cos to q4 min {cos.min():.7f}",
          json.dumps({t: r[t]["overlap@10"][0] for t in TESTS}), "hit", round(r["synth_test"]["hit@10"][0], 4))


def repack(a):
    d = ROOT / "out" / "export" / a.tag
    p = d / "model_q4.onnx"
    before = d / "model_q4_uint4.onnx"
    m = onnx.load(str(p))
    save(m, before)  # keep the old file to compare against
    for old in (p, d / "model_q4.onnx_data"):
        old.unlink()
    save(pack_gather(onnx.load(str(before))), p)
    T, B = Teacher(), Bench()
    enc = tokenize(T, B.texts("real_test"), remap=Tokenizer.from_file(str(d / "tokenizer.json")))
    x, y = ort_embed(before, enc), ort_embed(p, enc)
    cos = (x * y).sum(1)
    print(a.tag, "repacked; cos to the UINT4 file min", f"{cos.min():.7f}", "bytes", onnx_bytes(p), "was", onnx_bytes(before))
    for f in d.glob("model_q4_uint4.onnx*"):
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--vocab")
    ap.add_argument("--weights")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--formats", nargs="+", default=["fp32", "q4", "int8"])
    ap.add_argument("--wasm", action="store_true",
                    help="write out/export/<tag>/model_q4w.onnx: model_q4.onnx with the table unpacked by standard ops")
    ap.add_argument("--repack", action="store_true",
                    help="only repack an existing out/export/<tag>/model_q4.onnx (pack_gather) and check it")
    a = ap.parse_args()
    if a.repack:
        return repack(a)
    if a.wasm:
        return wasm_variant(a)
    torch.set_num_threads(4)
    rows = tok = None
    tok_path = HF / "tokenizer.json"
    if a.vocab:
        rows = json.loads((ROOT / a.vocab / "rows.json").read_text())
        tok_path = ROOT / a.vocab / "tokenizer.json"
        tok = Tokenizer.from_file(str(tok_path))
    s = Student(a.k, rows)
    if a.weights:
        sd = torch.load(ROOT / a.weights)
        w = sd.get("tm.embed_tokens.weight")
        if rows is not None and w is not None and len(w) != len(rows):
            sd["tm.embed_tokens.weight"] = w[torch.as_tensor(rows)]  # full-vocab student: cut its table
        missing, unexpected = s.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
    s.tm.config._attn_implementation = "eager"
    s.eval()
    d = ROOT / "out" / "export" / a.tag
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("model*"):
        old.unlink()  # onnx.save appends to an existing external-data file
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
    paths = {"fp32": fp32}
    if "q4" in a.formats:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=4, block_size=32, is_symmetric=False,
                                 op_types_to_quantize=("MatMul", "Gather"),
                                 quant_axes=(("MatMul", 0), ("Gather", 1)))
        q.process()
        paths["q4"] = d / "model_q4.onnx"
        save(pack_gather(q.model.model), paths["q4"])
    if "int8" in a.formats:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        paths["int8"] = d / "model_int8.onnx"
        quantize_dynamic(str(fp32), str(paths["int8"]), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gather"], use_external_data_format=True)
    tok_gz = len(gzip.compress(tok_path.read_bytes()))
    (d / "tokenizer.json").write_bytes(tok_path.read_bytes())
    T, B = Teacher(), Bench()
    enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    res = {"k": a.k, "vocab": a.vocab, "weights": a.weights, "tokenizer_gz_bytes": tok_gz, "formats": {}}
    ref = {t: embed(s, enc[t]) for t in TESTS}
    for f, p in paths.items():
        P = {t: ort_embed(p, enc[t]) for t in TESTS}
        r, _ = B.score(P)
        cos = np.concatenate([(P[t] * ref[t]).sum(1) for t in TESTS])
        res["formats"][f] = {"onnx_bytes": onnx_bytes(p), "bytes": onnx_bytes(p) + tok_gz,
                             "cos_to_pytorch": {"mean": float(cos.mean()), "min": float(cos.min())}, **r}
        print(a.tag, f, f"{(onnx_bytes(p) + tok_gz) / 1e6:.1f} MB", f"cos-to-pt min {cos.min():.5f}",
              json.dumps({t: r[t]["overlap@10"][0] for t in TESTS}), "hit", round(r["synth_test"]["hit@10"][0], 4),
              flush=True)
        (ROOT / "results" / f"export_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
