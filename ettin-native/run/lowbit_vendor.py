"""Verbatim copy of quant_table / pack_bits / table_nbit from eg2-layer-student/run/export.py
@ origin/main 53fad07b (coord #24: reuse, do not build a second sub-4-bit unpack). Do not edit here."""
import numpy as np
from onnx import numpy_helper


def quant_table(W, bits, block=32):
    """Asymmetric min/max quantization of a [V, H] table in groups of `block` along H.
    Returns codes (uint8, one per value), fp16 scales [V, H/block], uint8 zero points, and
    the dequantized table the graph will reproduce."""
    V, H = W.shape
    G = W.reshape(V, H // block, block)
    lo, hi = G.min(-1, keepdims=True), G.max(-1, keepdims=True)
    qmax = 2 ** bits - 1
    scale = np.maximum((hi - lo) / qmax, 1e-12).astype(np.float16)
    sc = scale.astype(np.float32)
    zp = np.clip(np.round(-lo / sc), 0, qmax).astype(np.uint8)
    q = np.clip(np.round(G / sc) + zp, 0, qmax).astype(np.uint8)
    deq = ((q.astype(np.float32) - zp) * sc).reshape(V, H)
    return q.reshape(V, H), scale[..., 0], zp[..., 0], deq


def pack_bits(q, bits):
    """[V, H] codes < 2**bits -> [V, H*bits/8] bytes: each run of 8 values is one
    little-endian (8*bits)-bit word, value j at bit offset j*bits."""
    V, H = q.shape
    g = q.reshape(V, H // 8, 8).astype(np.uint32)
    word = (g << (np.arange(8, dtype=np.uint32) * bits)).sum(-1, dtype=np.uint64).astype(np.uint32)
    return np.stack([(word >> (8 * b)) & 255 for b in range(bits)], -1).astype(np.uint8).reshape(V, -1)


def table_nbit(model, W, bits, block=32):
    """Replace the GatherBlockQuantized token table with a `bits`-bit table (2, 3 or 4)
    quantized from W and unpacked with standard ops (runs on onnxruntime-web wasm):
    Gather packed bytes -> int32, weight each byte by 256^b, ReduceSum to one word per 8
    values, Cast uint32, BitShift right by j*bits, BitwiseAnd mask, Cast float, subtract the
    gathered zero point, multiply by the gathered fp16 scale."""
    from onnx import helper, TensorProto
    g = model.graph
    node = next(n for n in g.node if n.op_type == "GatherBlockQuantized")
    ids, out = node.input[1], node.output[0]
    drop = set(node.input[i] for i in (0, 2, 3))
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep)
    V, H = W.shape
    q, scale, zp, deq = quant_table(W, bits, block)
    p = "/tm/embed_tokens/t%d_" % bits

    def c(name, arr):
        g.initializer.append(numpy_helper.from_array(np.asarray(arr), p + name))
        return p + name
    codes, scl, zpt = c("codes", pack_bits(q, bits)), c("scales", scale), c("zp", zp)
    ax = c("ax", np.array([-1], np.int64))
    nodes = [
        helper.make_node("Gather", [codes, ids], [p + "g"], axis=0),
        helper.make_node("Reshape", [p + "g", c("sh8", np.array([0, 0, H // 8, bits], np.int64))], [p + "gb"]),
        helper.make_node("Cast", [p + "gb"], [p + "gi"], to=TensorProto.INT32),
        helper.make_node("Mul", [p + "gi", c("bw", (256 ** np.arange(bits)).astype(np.int32))], [p + "gw"]),
        helper.make_node("ReduceSum", [p + "gw", ax], [p + "word"], keepdims=1),
        helper.make_node("Cast", [p + "word"], [p + "wu"], to=TensorProto.UINT32),
        helper.make_node("BitShift", [p + "wu", c("off", (np.arange(8) * bits).astype(np.uint32))], [p + "sh"],
                         direction="RIGHT"),
        helper.make_node("BitwiseAnd", [p + "sh", c("mask", np.uint32(2 ** bits - 1))], [p + "v"]),
        helper.make_node("Reshape", [p + "v", c("shg", np.array([0, 0, H // block, block], np.int64))], [p + "vg"]),
        helper.make_node("Cast", [p + "vg"], [p + "vf"], to=TensorProto.FLOAT),
        helper.make_node("Gather", [zpt, ids], [p + "z"], axis=0),
        helper.make_node("Cast", [p + "z"], [p + "zf"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", [p + "zf", ax], [p + "z1"]),
        helper.make_node("Sub", [p + "vf", p + "z1"], [p + "d"]),
        # Cast the fp16 scale table before the Gather, not after: ORT folds a Cast of an
        # initializer at load, so no fp16 kernel runs (WebGPU's Gather needs shader-f16).
        helper.make_node("Cast", [scl], [p + "sf32"], to=TensorProto.FLOAT),
        helper.make_node("Gather", [p + "sf32", ids], [p + "sf"], axis=0),
        helper.make_node("Unsqueeze", [p + "sf", ax], [p + "s1"]),
        helper.make_node("Mul", [p + "d", p + "s1"], [p + "m"]),
        helper.make_node("Reshape", [p + "m", c("shh", np.array([0, 0, H], np.int64))], [out])]
    i = list(g.node).index(node)
    rest = [n for n in g.node if n is not node]
    del g.node[:]
    g.node.extend(rest[:i] + nodes + rest[i:])
    return model, deq
