"""Build smaller Kokoro-82M ONNX files for the browser from kokoro-onnx's model-files-v1.1 export.

Every variant first writes the torch-matching STFT weights into the graph (the fix kinescript-kokoro applies at
load time; see experiments/kokoro-onnx-kinescript Findings 4-6), so a browser needs no initializer override.
Large weights are then stored smaller and turned back into fp32 by a node in front of their consumer, so every
consumer node stays as it was:

    fp32      the export with the STFT fixed
    fp16      every large weight fp16 + Cast
    mixed     decoder (vocoder) weights int8 per output channel + DequantizeLinear; the rest fp16 + Cast
    mixed63   the same with codes held to +-63 (ettin-native's range)
    mixed2    int8 for everything that only shapes the sound: the vocoder, Kokoro's acoustic text encoder
              (/text_encoder/cnn, /text_encoder/lstm) and the F0/N predictors with their shared LSTM; fp16 for the
              path that sets durations (BERT, the duration encoder /text_encoder/lstms.*, /lstm, /duration_proj)

The duration path (BERT, text encoder, LSTMs, duration/F0/N predictors) stays fp16, because int8 there moves
word times (kokoro-onnx-kinescript round 5).

    python build.py SRC.onnx OUTDIR [variant ...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

STFT = "kmodel.decoder.generator.stft."
MIN_SIZE = 4096  # smaller tensors (biases, norms, Snake alphas, the STFT rows) stay fp32
# Consumers that only shape the sound: in KModel.forward_with_tokens, durations come from bert -> bert_encoder ->
# predictor.text_encoder (exported as /text_encoder/lstms.*) -> predictor.lstm -> duration_proj; F0/N (via
# predictor.shared) and Kokoro's own text_encoder (/text_encoder/cnn, /text_encoder/lstm) feed only the decoder.
GROUPS = {"te": re.compile(r"^/text_encoder/(cnn|lstm/)"), "f0": re.compile(r"^/F0"),
          "n": re.compile(r"^/N[._]"), "shared": re.compile(r"^/shared/")}
SOUND_ONLY = re.compile(r"^/(decoder/|text_encoder/(cnn|lstm/)|shared/|F0|N[._])")


def stft_weights(n_fft=20, hop=5):
    """CustomSTFT weights that match torch.stft/istft (same formula as kinescript_kokoro/onnx.py)."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    k, n = np.arange(n_fft // 2 + 1), np.arange(n_fft)
    angle = 2 * np.pi * np.outer(k, n) / n_fft
    edge = (k == 0) | (k == n_fft // 2)
    forward_imag = np.where(edge[:, None], 0.0, -np.sin(angle)) * window
    rows = window / n_fft * (np.where(edge, 1.0, 2.0) / ((window ** 2).sum() / hop))[:, None]
    f32 = lambda a: np.ascontiguousarray(a[:, None, :], dtype=np.float32)  # noqa: E731
    return {STFT + "weight_forward_imag": f32(forward_imag),
            STFT + "weight_backward_real": f32(np.cos(angle) * rows),
            STFT + "weight_backward_imag": f32(np.sin(angle) * rows)}


def consumers(graph):
    out = defaultdict(list)
    for node in graph.node:
        for i, name in enumerate(node.input):
            out[name].append((node, i))
    return out


def channel_axis(node, weight):
    """The weight axis that indexes output channels for this consumer."""
    if node.op_type == "Conv":
        return 0  # (out, in, k)
    if node.op_type == "ConvTranspose":
        return 1  # (in, out, k)
    if node.op_type == "Gemm":
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b else 1
    if node.op_type == "MatMul":
        return weight.ndim - 1
    if node.op_type == "LSTM":
        return 1  # W and R are (directions, 4 * hidden, in): one scale per gate row, shared by both directions
    return None


def int8_per_channel(w, axis, levels):
    moved = np.moveaxis(w, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    scale = np.abs(flat).max(1) / levels
    scale[scale == 0] = 1.0
    q = np.clip(np.round(flat / scale[:, None]), -levels, levels).astype(np.int8)
    return np.moveaxis(q.reshape(moved.shape), 0, axis), scale.astype(np.float32)


def build(src, variant):
    model = onnx.load(src)
    g = model.graph
    fixed = stft_weights()
    for t in g.initializer:
        if t.name in fixed:
            t.CopyFrom(numpy_helper.from_array(fixed[t.name], t.name))
    if variant == "fp32":
        return model, {}
    cons = consumers(g)
    keep, new_nodes, report = [], [], defaultdict(int)
    for t in g.initializer:
        w = numpy_helper.to_array(t)
        users = cons.get(t.name, [])
        if w.dtype != np.float32 or w.size < MIN_SIZE or not users:
            keep.append(t)
            continue
        if variant == "mixed2":
            int8_ok = all(SOUND_ONLY.match(n.name) for n, _ in users)
        elif variant.startswith("mixed+"):  # mixed plus named groups: te (acoustic text encoder), f0, n, shared
            extra = [GROUPS[g] for g in variant[6:].split("+")]
            int8_ok = all(n.name.startswith("/decoder/") or any(r.match(n.name) for r in extra) for n, _ in users)
        else:
            int8_ok = all(n.name.startswith("/decoder/") for n, _ in users)
        axes = {channel_axis(n, w) for n, i in users if i == 1 or (n.op_type == "LSTM" and i == 2)}
        weight_slot = all(i == 1 or (n.op_type == "LSTM" and i == 2) for n, i in users)
        if variant.startswith("mixed") and int8_ok and weight_slot and len(axes) == 1 and None not in axes:
            levels = 63 if variant == "mixed63" else 127
            q, scale = int8_per_channel(w, axes.pop(), levels)
            axis = channel_axis(users[0][0], w)
            keep += [numpy_helper.from_array(q, t.name + "__q"), numpy_helper.from_array(scale, t.name + "__s"),
                     numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), t.name + "__z")]
            new_nodes.append(helper.make_node("DequantizeLinear", [t.name + "__q", t.name + "__s", t.name + "__z"],
                                              [t.name], name=t.name + "__dq", axis=axis))
            report["int8"] += w.size
        else:
            keep.append(numpy_helper.from_array(w.astype(np.float16), t.name + "__h"))
            new_nodes.append(helper.make_node("Cast", [t.name + "__h"], [t.name], name=t.name + "__cast",
                                              to=onnx.TensorProto.FLOAT))
            report["fp16"] += w.size
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)  # producers before every consumer keeps the graph topologically sorted
    return model, dict(report)


if __name__ == "__main__":
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in sys.argv[3:] or ["fp32", "fp16", "mixed", "mixed63", "mixed2"]:
        model, report = build(src, variant)
        path = out / f"kokoro-{variant}.onnx"
        onnx.save(model, str(path))
        onnx.checker.check_model(str(path))
        summary[variant] = {"bytes": path.stat().st_size, "params": report}
        print(variant, summary[variant], flush=True)
    old = json.loads((out / "build.json").read_text()) if (out / "build.json").exists() else {}
    (out / "build.json").write_text(json.dumps({**old, **summary}, indent=1))
