"""Replace the fixed DFT tables in the tiny-voca graph with nodes that compute them at load time.

The source-filter vocoder carries STFT analysis kernels, inverse-STFT kernels and cepstrum-to-spectrum matrices:
2.8 MB of float32 cosines and sines (times a Hann window) that gzip cannot shrink. Each has the form

    K[i, j] = amp[i] * win[j] * trig(2*pi/N * ((r[i] * c[j]) mod N))

so the file keeps only amp, win, r and c (a few KB) and a Range/Mul/Mod/Cos chain that onnxruntime folds into a
constant when the session starts. Every replaced table is checked against the stored one (max abs error < 1e-6)
before it is swapped, and a table that matches no form is left alone.

    python compute_constants.py IN.onnx OUT.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

MIN_BYTES = 32768


def hann(n):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float64)


def table(amp, win, r, c, n, trig):
    angle = 2 * np.pi / n * ((np.asarray(r)[:, None] * np.asarray(c)[None, :]) % n)
    return np.asarray(amp)[:, None] * np.asarray(win)[None, :] * (np.cos(angle) if trig == "cos" else np.sin(angle))


def candidates(shape):
    """(description, parts) for the forms a table of this shape could take; parts is a list of
    (amp, win, r, c, n, trig) blocks stacked along axis 0, then reshaped to `shape`."""
    out = []
    if len(shape) == 3 and shape[1] == 1:
        rows, n = shape[0], shape[2]
        bins = n // 2 + 1
        w = hann(n)
        if rows == bins:          # analysis: rfft(eye * window) -> real = w cos, imag = -w sin
            out.append(("analysis real", [(np.ones(bins), w, np.arange(bins), np.arange(n), n, "cos")]))
            out.append(("analysis imag", [(-np.ones(bins), w, np.arange(bins), np.arange(n), n, "sin")]))
        if rows == 2 * bins:      # synthesis: windowed irfft of unit real and unit imaginary bins
            scale = np.full(bins, 2.0 / n)
            scale[0] = scale[-1] = 1.0 / n
            imag = -scale.copy()
            imag[0] = imag[-1] = 0.0
            out.append(("synthesis", [(scale, w, np.arange(bins), np.arange(n), n, "cos"),
                                      (imag, w, np.arange(bins), np.arange(n), n, "sin")]))
    if len(shape) == 2:
        rows, bins = shape
        n = 2 * (bins - 1)
        for causal in range(rows + 1):  # cepstrum: rfft of unit coefficients at 0..causal-1 and n-1-i
            positions = list(range(causal)) + [n - 1 - i for i in range(rows - causal)]
            out.append((f"cepstrum real ({causal} causal)", [(np.ones(rows), np.ones(bins), positions, np.arange(bins), n, "cos")]))
            out.append((f"cepstrum imag ({causal} causal)", [(-np.ones(rows), np.ones(bins), positions, np.arange(bins), n, "sin")]))
    return out


def build(name, parts, shape, nodes, inits):
    """Nodes that compute the table named `name`; small constants go to `inits`."""
    blocks = []
    for k, (amp, win, r, c, n, trig) in enumerate(parts):
        p = f"{name}__gen{k}"
        inits += [numpy_helper.from_array(np.asarray(r, np.int64)[:, None], p + "_r"),
                  numpy_helper.from_array(np.asarray(c, np.int64)[None, :], p + "_c"),
                  numpy_helper.from_array(np.array(n, np.int64), p + "_n"),
                  numpy_helper.from_array(np.array(2 * np.pi / n, np.float32), p + "_w"),
                  numpy_helper.from_array(np.asarray(amp, np.float32)[:, None], p + "_amp"),
                  numpy_helper.from_array(np.asarray(win, np.float32)[None, :], p + "_win")]
        nodes += [helper.make_node("Mul", [p + "_r", p + "_c"], [p + "_rc"]),
                  helper.make_node("Mod", [p + "_rc", p + "_n"], [p + "_mod"]),
                  helper.make_node("Cast", [p + "_mod"], [p + "_f"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", [p + "_f", p + "_w"], [p + "_angle"]),
                  helper.make_node("Cos" if trig == "cos" else "Sin", [p + "_angle"], [p + "_trig"]),
                  helper.make_node("Mul", [p + "_trig", p + "_amp"], [p + "_a"]),
                  helper.make_node("Mul", [p + "_a", p + "_win"], [p + "_out"])]
        blocks.append(p + "_out")
    joined = name + "__gen_cat"
    if len(blocks) > 1:
        nodes.append(helper.make_node("Concat", blocks, [joined], axis=0))
    else:
        joined = blocks[0]
    inits.append(numpy_helper.from_array(np.array(shape, np.int64), name + "__gen_shape"))
    nodes.append(helper.make_node("Reshape", [joined, name + "__gen_shape"], [name]))


def match(w):
    """(label, parts) of the first form `w` matches, or None."""
    if w.dtype != np.float32 or w.nbytes < MIN_BYTES:
        return None
    for label, parts in candidates(w.shape):
        stacked = np.concatenate([table(*part) for part in parts], axis=0).reshape(w.shape)
        err = float(np.abs(stacked - w).max())
        if err < 1e-6:
            return label, parts, err
    return None


def main(src, dst):
    model = onnx.load(src)
    g = model.graph
    keep, new_nodes, new_inits, report = [], [], [], []
    for t in g.initializer:                      # tables stored as initializers
        w = numpy_helper.to_array(t)
        found = match(w)
        if found:
            build(t.name, found[1], w.shape, new_nodes, new_inits)
            report.append((t.name, found[0], w.shape, w.nbytes, found[2]))
        else:
            keep.append(t)
    kept_nodes = []
    for node in g.node:                          # tables stored as Constant nodes (torch.onnx's folded buffers)
        value = next((a.t for a in node.attribute if a.name == "value"), None) if node.op_type == "Constant" else None
        found = match(numpy_helper.to_array(value)) if value is not None and value.dims else None
        if found:
            w = numpy_helper.to_array(value)
            build(node.output[0], found[1], w.shape, new_nodes, new_inits)
            report.append((node.output[0], found[0], w.shape, w.nbytes, found[2]))
        else:
            kept_nodes.append(node)
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes + kept_nodes)
    onnx.save(model, dst)
    onnx.checker.check_model(dst)
    for name, label, shape, nbytes, err in report:
        print(f"{name:28} {label:26} {str(shape):18} {nbytes / 1e6:6.3f} MB  max err {err:.1e}")
    print(f"replaced {sum(r[3] for r in report) / 1e6:.3f} MB in {len(report)} tables")


if __name__ == "__main__":
    main(*sys.argv[1:3])
