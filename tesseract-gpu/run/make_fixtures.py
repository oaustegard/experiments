"""Packed-batch fixtures for the GPU kernels: every layer's input/output from tessgpu.ref.

    python3 run/make_fixtures.py            # -> data/fixtures/<case>/

A case is one model and a batch of line images, packed the way the GPU engines hold them:
lines sorted by width, longest first, each line's rows concatenated after the previous
line's. manifest.json gives per-line widths and offsets (in elements of the packed axis):

  in     f32 [sum H*W_i]          normalised input (int8 models: integers -127..127)
  pads   f32 [sum nb_i, 9]        Convolve padding, border raster order, slot (dx+1)*3+(dy+1)
  conv   f32 [sum H*W_i, 16]      Conv 3x3 -> FC 9+1 -> tanh (int8 models: quantised)
  pool   f32 [sum Hp*T_i, 16]     max over 3x3 blocks, Hp = H/3, T_i = W_i // 3
  ylstm  f32 [sum T_i, nsy]       y-summarising LSTM, last step of each column
  x1     f32 [sum T_i, ns1]       forward LSTM along x
  x2     f32 [sum T_i, ns2]       reversed LSTM along x (outputs at original positions)
  x3     f32 [sum T_i, ns3]       forward LSTM along x
  probs  f32 [sum T_i, C]         softmax

Weights, f32: w_conv [16, 10]; w_y, w_x1, w_x2, w_x3 [4*ns, ni+ns+1] with the gate blocks
in the order CI (tanh), GI, GF1, GO (logistic), each block [ns, ni+ns+1], input columns
first, then the previous output, then the bias; w_out [C, ns3+1]. int8 models store the
integer codes in w_* and the per-row scales in s_* (dot = (sum q*x + q_bias*127) * s).
tanh.f32 / logistic.f32 are Tesseract's 4096-entry tables (x = i/256).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tessgpu import bench, decode, ref  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

OUT = HERE / "data" / "fixtures"


def lines_for(lang, n, height):
    rows = [r for r in bench.rows("synth")
            if r["lang"] == lang and r["cond"] in ("clean", "noise", "shade")]
    pick = rows[:: max(1, len(rows) // n)][:n]
    imgs = []
    for r in pick:
        g = np.asarray(Image.open(r["path"]).convert("L"))
        imgs.append(ref.scale_to_height(g, height))
    return imgs


def gate_matrix(node, int_mode):
    blocks = [node.gates[k] for k in ("CI", "GI", "GF1", "GO")]
    if int_mode:
        return (np.concatenate([b.q for b in blocks]).astype(np.float32),
                np.concatenate([b.scale for b in blocks]).astype(np.float32))
    return np.concatenate([b.w for b in blocks]).astype(np.float32), None


def fc_matrix(wts, int_mode):
    if int_mode:
        return wts.q.astype(np.float32), wts.scale.astype(np.float32)
    return wts.w.astype(np.float32), None


def make_case(name, model_path, imgs):
    m = load(model_path)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    imgs = sorted(imgs, key=lambda g: -g.shape[1])
    H = m.input_height
    per = [ref.layer_outputs(m, g) for g in imgs]
    lines, offs = [], {"in": 0, "pad": 0, "pool": 0, "t": 0}
    for g, o in zip(imgs, per):
        W = g.shape[1]
        T = W // 3
        nb = o["pad_table"].shape[0]
        lines.append({"W": W, "T": T, "nb": nb, "in_off": offs["in"], "pad_off": offs["pad"],
                      "pool_off": offs["pool"], "t_off": offs["t"]})
        offs["in"] += H * W
        offs["pad"] += nb
        offs["pool"] += (H // 3) * T
        offs["t"] += T
    files = {}

    def save(key, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        arr.tofile(d / f"{key}.f32")
        files[key] = {"file": f"{key}.f32", "shape": list(arr.shape)}

    save("in", np.concatenate([o["in"].ravel() for o in per]))
    save("gray", np.concatenate([g.ravel() for g in imgs]).astype(np.float32))
    save("pads", np.concatenate([o["pad_table"] for o in per]))
    save("conv", np.concatenate([o["conv"].reshape(-1, o["conv"].shape[-1]) for o in per]))
    save("pool", np.concatenate([o["pool"].reshape(-1, o["pool"].shape[-1]) for o in per]))
    for k in ("ylstm", "x1", "x2", "x3", "probs"):
        save(k, np.concatenate([o[k] for o in per]))
    kids = m.network.children
    conv_fc = kids[1].children[1]
    w, s = fc_matrix(conv_fc.weights, m.int_mode)
    save("w_conv", w)
    if s is not None:
        save("s_conv", s)
    lstm_nodes = {"y": kids[3].children[0], "x1": kids[4], "x2": kids[5].children[0], "x3": kids[6]}
    for k, node in lstm_nodes.items():
        w, s = gate_matrix(node, m.int_mode)
        save(f"w_{k}", w)
        if s is not None:
            save(f"s_{k}", s)
    w, s = fc_matrix(kids[7].weights, m.int_mode)
    save("w_out", w)
    if s is not None:
        save("s_out", s)
    save("tanh", ref.TANH_T)
    save("logistic", ref.LOGISTIC_T)
    man = {"case": name, "model": Path(model_path).name, "variant": Path(model_path).parent.name,
           "int_mode": m.int_mode, "H": H, "Hp": H // 3, "C": m.num_classes,
           "null_char": m.null_char,
           "ns": {k: n.ns for k, n in lstm_nodes.items()}, "lines": lines,
           "total": {"in": offs["in"], "pad": offs["pad"], "pool": offs["pool"], "t": offs["t"]},
           "files": files,
           "texts": [decode.greedy(o["probs"], decode.Codec(m)) for o in per]}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    print(name, len(imgs), "lines, T total", offs["t"])


def main():
    parity = [np.asarray(Image.open(HERE / f"data/parity/l{i}.png")) for i in range(3)]
    eng = parity + lines_for("eng", 3, 36)
    make_case("best_eng", HERE / "models/best/eng.traineddata", eng)
    make_case("fast_eng", HERE / "models/fast/eng.traineddata", eng)
    make_case("best_grc", HERE / "models/best/grc.traineddata", lines_for("grc", 3, 48))


if __name__ == "__main__":
    main()
