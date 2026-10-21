"""Export a sentence-transformers model (Ettin/ModernBERT or BERT) to one fp32 ONNX graph that
outputs the pooled embedding, then build the compression ladder from it.

Usage: .venv/bin/python -I run/export.py HF_ID NAME [--ks 16000 32000]

Graph: input_ids, attention_mask (int64 [B,S]) -> sentence_embedding [B,D], i.e. transformer,
the model's own ST pooling (mean or CLS) and any Dense modules, NOT normalized (the harness
normalizes). Variants written to out/models/NAME/<variant>/{model.onnx, tokenizer.json}:

  fp32        reference
  int8        MatMuls int8 (ORT dynamic, per-channel), token table int8 one scale per row
  t4          MatMuls int8, token table int4 symmetric, groups of 32 (custom nibble unpack)
  k<K>-int8   table pruned to top-K corpus/training tokens (BPE-closed), then as int8
  k<K>-t4     pruned, then as t4

The token table is handled by hand rather than by ORT's quantizer so every variant runs on
onnxruntime-web's wasm backend: Gather on the packed rows and their scales, then standard ops
(BitShift/BitwiseAnd/Cast/Mul) to dequantize only the looked-up rows. GatherBlockQuantized,
which ORT's own 4-bit path uses, has no wasm kernel (eg2-layer-student finding 8).
Token frequencies come from the 61,686 post texts and #102's training texts (items.json
queries); test queries are never counted.
"""
import argparse
import collections
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
POSTS = EXP / "eg2-query-student" / "out" / "posts.jsonl"
ITEMS = EXP / "eg2-query-student" / "web" / "items.json"
TEST = EXP / "eg2-native-small" / "out" / "q"
GENERAL = True  # mix general-English token counts into the pruning frequencies


class Wrap(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        f = {"input_ids": input_ids, "attention_mask": attention_mask}
        for mod in self.st:
            if type(mod).__name__ == "Normalize":
                continue
            f = mod(f)
        return f["sentence_embedding"]


def export_fp32(hf, dst):
    st = SentenceTransformer(hf, device="cpu", model_kwargs={"attn_implementation": "eager"})
    st.eval()
    ids = torch.tensor([[50281, 100, 200, 300, 50282], [50281, 100, 200, 50282, 0]])
    if "modernbert" not in st[0].auto_model.config.model_type:
        ids = torch.tensor([[101, 100, 200, 300, 102], [101, 100, 200, 102, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    dst.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrap(st), (ids, attn), str(dst / "model.onnx"), input_names=["input_ids", "attention_mask"],
                          output_names=["sentence_embedding"], opset_version=18, dynamo=False,
                          dynamic_axes={"input_ids": {0: "B", 1: "S"}, "attention_mask": {0: "B", 1: "S"},
                                        "sentence_embedding": {0: "B"}})
    m = onnx.load(str(dst / "model.onnx"))  # re-save as a single file
    onnx.save(m, str(dst / "model.onnx"))
    for p in (dst).glob("*.data"):
        p.unlink()
    snap = Path(st.tokenizer.name_or_path)
    tj = st.tokenizer.backend_tokenizer.to_str() if hasattr(st.tokenizer, "backend_tokenizer") else None
    (dst / "tokenizer.json").write_text(tj)
    meta = {"hf": hf, "prompts": st.prompts, "max_seq_length": st.max_seq_length,
            "model_type": st[0].auto_model.config.model_type}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    return st


def table_init(m):
    """The token table: the initializer feeding a Gather whose indices are input_ids."""
    inits = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Gather" and n.input[0] in inits and n.input[1] == "input_ids":
            return n, inits[n.input[0]]
    raise RuntimeError("no token-table Gather on input_ids")


def frequencies(tok):
    posts = [json.loads(l)["text"] for l in POSTS.open()]
    items = [x for x in json.load(ITEMS.open())["items"] if isinstance(x, str)]
    test = {json.loads(l)["q"].strip().lower() for f in ("synth_test", "real_test") for l in (TEST / f"{f}.jsonl").open()}
    items = [q for q in items if q.strip().lower() not in test]
    c = collections.Counter()
    for enc in tok.encode_batch([p for p in posts if p] + items):
        c.update(enc.ids)
    # General English (wasm #19: domain-only frequencies break out-of-domain queries): the
    # MS MARCO / NQ rows of run/teacher.py's train.jsonl, counted equally with the domain text.
    gen_path = ROOT / "out" / "data" / "train.jsonl"
    if GENERAL and gen_path.exists():
        dom = set(posts) | set(items)
        gen = [r["t"] for r in map(json.loads, gen_path.open()) if r["t"] not in dom]
        g = collections.Counter()
        for enc in tok.encode_batch(gen):
            g.update(enc.ids)
        scale = sum(c.values()) / max(1, sum(g.values()))
        for i, n in g.items():
            c[i] += n * scale
    return c


def kept_set(tj, freq, k):
    model = tj["model"]
    vocab = model["vocab"]
    always = {a["id"] for a in tj["added_tokens"]}
    if model["type"] == "BPE":
        always |= {i for t, i in vocab.items() if len(t) == 1}  # byte-level alphabet
    else:  # WordPiece: keep every single character and its ## form
        always |= {i for t, i in vocab.items() if len(t) == 1 or (t.startswith("##") and len(t) == 3)}
    keep = always | {i for i, _ in freq.most_common(k)}
    if model["type"] == "BPE":
        inv = {i: t for t, i in vocab.items()}
        built = {}
        for mg in model["merges"]:
            a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
            built.setdefault(a + b, (a, b))
        stack = [inv[i] for i in keep if i in inv]
        while stack:
            t = stack.pop()
            for part in built.get(t, ()):
                if vocab[part] not in keep:
                    keep.add(vocab[part])
                    stack.append(part)
    return sorted(keep)


def prune_tokenizer(tj, keep):
    tj = copy.deepcopy(tj)
    new_id = {o: n for n, o in enumerate(keep)}
    old = tj["model"]["vocab"]
    inv = {i: t for t, i in old.items()}
    tj["model"]["vocab"] = {inv[o]: new_id[o] for o in keep if o in inv}
    kv = tj["model"]["vocab"]
    if tj["model"]["type"] == "BPE":
        ms = []
        for mg in tj["model"]["merges"]:
            a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
            if a in kv and b in kv and a + b in kv:
                ms.append(mg)
        tj["model"]["merges"] = ms
    for a in tj["added_tokens"]:
        a["id"] = new_id[a["id"]]
    pp = tj.get("post_processor")
    if pp:  # TemplateProcessing special-token ids
        for v in (pp.get("special_tokens") or {}).values():
            v["ids"] = [new_id[i] for i in v["ids"]]
        for p in pp.get("processors", []) or []:
            for v in (p.get("special_tokens") or {}).values():
                v["ids"] = [new_id[i] for i in v["ids"]]
        for key in ("sep", "cls"):
            if key in pp and isinstance(pp[key], list):
                pp[key][1] = new_id[pp[key][1]]
    if tj.get("padding"):
        tj["padding"]["pad_id"] = new_id.get(tj["padding"]["pad_id"], 0)
    return tj


def prune_table(m, keep):
    _, init = table_init(m)
    W = numpy_helper.to_array(init)[np.array(keep)]
    init.CopyFrom(numpy_helper.from_array(W, init.name))


def quant_table(m, bits, group=32):
    """Replace the table Gather with Gather(codes) + Gather(scales) + dequant, standard ops."""
    node, init = table_init(m)
    W = numpy_helper.to_array(init).astype(np.float32)
    V, H = W.shape
    out, name = node.output[0], init.name
    g = m.graph
    nodes = []
    if bits == 8:
        s = np.abs(W).max(1, keepdims=True) / 127 + 1e-12
        q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
        inits = [numpy_helper.from_array(q, name + "_q8"), numpy_helper.from_array(s.astype(np.float32), name + "_s")]
        nodes += [helper.make_node("Gather", [name + "_q8", "input_ids"], ["tq"], axis=0),
                  helper.make_node("Gather", [name + "_s", "input_ids"], ["ts"], axis=0),
                  helper.make_node("Cast", ["tq"], ["tf"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", ["tf", "ts"], [out])]
    else:
        assert bits == 4 and H % group == 0
        G = H // group
        Wg = W.reshape(V, G, group)
        s = np.abs(Wg).max(2, keepdims=True) / 7 + 1e-12
        q = (np.clip(np.round(Wg / s), -8, 7).astype(np.int16) + 8).astype(np.uint8).reshape(V, H)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)  # lo nibble = even column
        c = lambda n, v, dt: numpy_helper.from_array(np.array(v, dt), n)
        inits = [numpy_helper.from_array(packed, name + "_q4"),
                 numpy_helper.from_array(s.reshape(V, G, 1).astype(np.float16), name + "_s16"),
                 c(name + "_m15", 15, np.uint8), c(name + "_4", 4, np.uint8), c(name + "_8", 8.0, np.float32),
                 c(name + "_shape", [0, 0, G, group], np.int64), c(name + "_shape2", [0, 0, H], np.int64),
                 c(name + "_shape3", [0, 0, H // 2, 1], np.int64)]
        nodes += [helper.make_node("Gather", [name + "_q4", "input_ids"], ["tp"], axis=0),     # [B,S,H/2] u8
                  helper.make_node("BitwiseAnd", ["tp", name + "_m15"], ["tlo"]),
                  helper.make_node("BitShift", ["tp", name + "_4"], ["thi"], direction="RIGHT"),
                  helper.make_node("Reshape", ["tlo", name + "_shape3"], ["tlo1"]),
                  helper.make_node("Reshape", ["thi", name + "_shape3"], ["thi1"]),
                  helper.make_node("Concat", ["tlo1", "thi1"], ["tcat"], axis=3),             # [B,S,H/2,2]
                  helper.make_node("Reshape", ["tcat", name + "_shape2"], ["tu"]),            # [B,S,H]
                  helper.make_node("Cast", ["tu"], ["tf"], to=TensorProto.FLOAT),
                  helper.make_node("Sub", ["tf", name + "_8"], ["tc"]),
                  helper.make_node("Reshape", ["tc", name + "_shape"], ["tg"]),               # [B,S,G,32]
                  helper.make_node("Gather", [name + "_s16", "input_ids"], ["ts16"], axis=0),
                  helper.make_node("Cast", ["ts16"], ["ts"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", ["tg", "ts"], ["tm"]),
                  helper.make_node("Reshape", ["tm", name + "_shape2"], [out])]
    idx = list(g.node).index(node)
    g.node.remove(node)
    for k, n in enumerate(nodes):
        g.node.insert(idx + k, n)
    g.initializer.remove(init)
    g.initializer.extend(inits)


def weight_only_int8(m, L=127):
    """MatMul weights -> int8 per output column (codes in +-L) + DequantizeLinear; activations stay fp32.
    L < 127 lowers code entropy so brotli shrinks the file (wasm #49)."""
    inits = {i.name: i for i in m.graph.initializer}
    done = set()
    for n in m.graph.node:
        if n.op_type == "MatMul" and n.input[1] in inits and n.input[1] not in done:
            done.add(n.input[1])
            init = inits[n.input[1]]
            W = numpy_helper.to_array(init).astype(np.float32)
            if W.ndim != 2:
                continue
            s = (np.abs(W).max(0) / L + 1e-12).astype(np.float32)
            q = np.clip(np.round(W / s), -L, L).astype(np.int8)
            base = init.name
            m.graph.initializer.remove(init)
            m.graph.initializer.extend([numpy_helper.from_array(q, base + "_w8"), numpy_helper.from_array(s, base + "_w8s")])
            m.graph.node.insert(0, helper.make_node("DequantizeLinear", [base + "_w8", base + "_w8s"], [base], axis=1))


def to_fp16_storage(m):
    inits = {i.name: i for i in m.graph.initializer}
    done = set()
    for n in m.graph.node:
        if n.op_type == "MatMul" and n.input[1] in inits and n.input[1] not in done:
            done.add(n.input[1])
            init = inits[n.input[1]]
            W = numpy_helper.to_array(init)
            if W.dtype != np.float32 or W.ndim != 2:
                continue
            base = init.name
            m.graph.initializer.remove(init)
            m.graph.initializer.append(numpy_helper.from_array(W.astype(np.float16), base + "_h"))
            m.graph.node.insert(0, helper.make_node("Cast", [base + "_h"], [base], to=TensorProto.FLOAT))


def build(src_dir, dst, keep=None, table_bits=8, matmul="w8"):
    """matmul: dyn8 (ORT dynamic, activations quantized too), w8 (weight-only int8 per column),
    nb4 / nb8 (MatMulNBits weight-only, block 32, symmetric), f16 (fp16 storage), f32."""
    dst.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(src_dir / "model.onnx"))
    tj = json.loads((src_dir / "tokenizer.json").read_text())
    if keep is not None:
        prune_table(m, keep)
        tj = prune_tokenizer(tj, keep)
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    shutil.copy(src_dir / "meta.json", dst / "meta.json")
    node, _ = table_init(m)
    if matmul == "dyn8":
        tmp = dst / "tmp.onnx"
        onnx.save(m, str(tmp))
        quantize_dynamic(str(tmp), str(tmp), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul"], nodes_to_exclude=[node.name])
        m = onnx.load(str(tmp))
        tmp.unlink()
    elif matmul in ("nb4", "nb8"):
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        qz = MatMulNBitsQuantizer(m, bits=int(matmul[2]), block_size=32, is_symmetric=True, accuracy_level=None)
        qz.process()
        m = qz.model.model
    elif matmul.startswith("w8"):
        weight_only_int8(m, int(matmul[3:]) if matmul.startswith("w8L") else 127)
    elif matmul == "f16":
        to_fp16_storage(m)
    if table_bits in (8, 4):
        quant_table(m, table_bits)
    elif table_bits in (2, 3):
        # eg2-layer-student's asymmetric n-bit table (vendored verbatim); its table_nbit looks for
        # a GatherBlockQuantized node, so stand one in for our plain Gather and let it rebuild.
        from lowbit_vendor import table_nbit
        node, init = table_init(m)
        W = numpy_helper.to_array(init).astype(np.float32)
        node.op_type = "GatherBlockQuantized"
        node.input.extend([init.name, init.name])
        m, _ = table_nbit(m, W, table_bits)
    elif table_bits == 16:
        _, init = table_init(m)
        W = numpy_helper.to_array(init)
        name = init.name
        m.graph.initializer.remove(init)
        m.graph.initializer.append(numpy_helper.from_array(W.astype(np.float16), name + "_h"))
        m.graph.node.insert(0, helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT))
    onnx.checker.check_model(m)
    onnx.save(m, str(dst / "model.onnx"))


def recipe(v):
    """'k16000-t4-nb4' -> (16000, 4, 'nb4'); parts: k<K> (optional), t<bits>, matmul mode."""
    k, tb, mm = None, 32, "f32"
    for part in v.split("-"):
        if part.startswith("k") and part[1:].isdigit():
            k = int(part[1:])
        elif part.startswith("t") and part[1:].isdigit():
            tb = int(part[1:])
        else:
            mm = part
    return k, tb, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf")
    ap.add_argument("name")
    ap.add_argument("variants", nargs="*", default=["t8-w8", "t4-w8", "t4-nb4", "k16000-t4-nb4", "k32000-t4-nb4"],
                    help="k<K>-t<table bits>-<matmul mode>, e.g. k16000-t4-nb4")
    a = ap.parse_args()
    base = ROOT / "out" / "models" / a.name
    fp = base / "fp32"
    if not (fp / "model.onnx").exists():
        export_fp32(a.hf, fp)
    tok = Tokenizer.from_file(str(fp / "tokenizer.json"))
    tj = json.loads((fp / "tokenizer.json").read_text())
    freq = None
    for v in a.variants:
        k, tb, mm = recipe(v)
        keep = None
        if k:
            freq = freq or frequencies(tok)
            keep = kept_set(tj, freq, k)
        build(fp, base / v, keep, tb, mm)
        print(a.name, v, "vocab", len(keep) if keep else tok.get_vocab_size(),
              round((base / v / "model.onnx").stat().st_size / 1e6, 2), "MB", flush=True)


if __name__ == "__main__":
    main()
