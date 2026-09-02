"""Export a trained inverter to ONNX for the browser (Tier 2 demo), and verify it.

  python3 export_onnx.py --cond float                 # zero + corrector, fp32 + 8-bit weight-only
  python3 export_onnx.py --cond float --verify 32     # also decode 32 dev items through ONNX

Per model two graphs:
  <tag>_enc.onnx   e (B,384) [+ e_hyp, hyp_ids, hyp_mask for the corrector] -> encoder states
  <tag>_dec.onnx   (ids, enc_h, enc_mask) -> logits for the last position (no KV cache;
                   sequences are <= 48 tokens, recomputing is cheap)

Measured on the 40k-pair model (2026-09-02): fp32 export is exact (32/32 greedy
decodes match torch); onnxruntime `quantize_dynamic` int8 BREAKS the decoder
(degenerate loops, 0/32); MatMulNBits weight-only 8-bit keeps it (23/32, verifier
cosine 0.511 vs 0.525). So the shipped form is fp32 activations with 8-bit weights.
The export also drops initializers no node references (the zero-step encoder
carries the 66 MB token table it never uses).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch

from model import Inverter, encode_labels, tokenizer

HERE = Path(__file__).resolve().parent
CKPT, OUT, DATA = HERE / "ckpt", HERE / "onnx", HERE / "data"


class Enc(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, e, e_hyp=None, hyp_ids=None, hyp_mask=None):
        x, mask = self.m.encoder_inputs(e, e_hyp, hyp_ids, hyp_mask)
        return self.m.t5.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state


class Dec(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ids, enc_h, enc_mask):
        return self.m.t5(decoder_input_ids=ids, encoder_outputs=(enc_h,), attention_mask=enc_mask).logits[:, -1, :]


def slim(path: Path) -> tuple[int, int]:
    """Drop dead nodes (the T5 encoder's own token Gather survives export even
    though inputs_embeds bypasses it), then unreferenced initializers, then
    merge byte-identical initializers (the corrector references t5.shared twice)."""
    g = onnx.load(path)
    outs = {o.name for o in g.graph.output}
    dead = 0
    while True:
        consumed = {i for n in g.graph.node for i in n.input} | outs
        rm = [n for n in g.graph.node if not any(o in consumed for o in n.output)]
        if not rm:
            break
        for n in rm:
            g.graph.node.remove(n)
        dead += len(rm)
    used = {i for n in g.graph.node for i in n.input}
    for t in [t for t in g.graph.initializer if t.name not in used]:
        g.graph.initializer.remove(t)
    seen, rename, dup = {}, {}, []
    for t in g.graph.initializer:
        key = (t.data_type, tuple(t.dims), hash(t.raw_data))
        if key in seen:
            rename[t.name] = seen[key]; dup.append(t)
        else:
            seen[key] = t.name
    for t in dup:
        g.graph.initializer.remove(t)
    for n in g.graph.node:
        for i, name in enumerate(n.input):
            if name in rename:
                n.input[i] = rename[name]
    onnx.save(g, path)
    return dead, len(dup)


def gather_int8(src: Path, dst: Path) -> None:
    """int8 the token tables (Gather) too; MatMulNBits leaves them fp32 (66 MB each)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])


def weight_only_8bit(src: Path, dst: Path) -> None:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(src)
    q = MatMulNBitsQuantizer(model, block_size=128, is_symmetric=True, bits=8)
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)


def export(mode: str, cond: str, k: int) -> dict:
    tag = f"{mode}_{cond}"
    m = Inverter(mode, k=k)
    m.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location="cpu")); m.eval()
    OUT.mkdir(exist_ok=True)
    e = torch.randn(1, 384)
    if mode == "zero":
        args, names, dyn = (e,), ["e"], {"e": {0: "b"}}
    else:
        _, hid, hm = encode_labels(tokenizer(), ["a hypothesis to correct"])
        args, names = (e, torch.randn(1, 384), hid, hm), ["e", "e_hyp", "hyp_ids", "hyp_mask"]
        dyn = {"e": {0: "b"}, "e_hyp": {0: "b"}, "hyp_ids": {0: "b", 1: "H"}, "hyp_mask": {0: "b", 1: "H"}}
    enc_p, dec_p = OUT / f"{tag}_enc.onnx", OUT / f"{tag}_dec.onnx"
    torch.onnx.export(Enc(m), args, enc_p, input_names=names, output_names=["enc_h"],
                      dynamic_axes={**dyn, "enc_h": {0: "b", 1: "S"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        enc_h = Enc(m)(*args)
    ids = torch.tensor([[0, 363, 19]]); mask = torch.ones(1, enc_h.shape[1], dtype=torch.long)
    torch.onnx.export(Dec(m), (ids, enc_h, mask), dec_p, input_names=["ids", "enc_h", "enc_mask"], output_names=["logits"],
                      dynamic_axes={"ids": {0: "b", 1: "L"}, "enc_h": {0: "b", 1: "S"}, "enc_mask": {0: "b", 1: "S"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    slim_stats = [slim(enc_p), slim(dec_p)]
    for src in (enc_p, dec_p):
        w8 = src.with_suffix(".w8.onnx")
        weight_only_8bit(src, w8)
        gather_int8(w8, src.with_suffix(".w8g.onnx"))
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(OUT.glob(f"{tag}_*.onnx"))}
    print(f"{tag}: slimmed (dead nodes, dup initializers) enc {slim_stats[0]} dec {slim_stats[1]}; MB {sizes}", flush=True)
    return sizes


def verify(cond: str, k: int, n: int) -> None:
    """Greedy zero-step decode of n dev items through ONNX (fp32 and w8) vs torch."""
    import onnxruntime as ort
    from encoder import BekkoEncoder, SignBits, condition
    tok = tokenizer()
    m = Inverter("zero", k=k); m.load_state_dict(torch.load(CKPT / f"zero_{cond}.pt", map_location="cpu")); m.eval()
    sb = SignBits.load(DATA / "signbits_mu.npy") if cond == "bin1" else None
    texts = json.loads((DATA / "splits.json").read_text())["dev"][:n]
    emb = condition(cond, np.load(DATA / "emb_dev.npy")[:n], sb)
    enc = BekkoEncoder()
    with torch.no_grad():
        ref = tok.batch_decode(m.generate(torch.from_numpy(emb), num_beams=1), skip_special_tokens=True)
    se = ort.InferenceSession(str(OUT / f"zero_{cond}_enc.onnx"))
    h = se.run(None, {"e": emb})[0]
    hq = ort.InferenceSession(str(OUT / f"zero_{cond}_enc.w8g.onnx")).run(None, {"e": emb})[0]
    print(f"encoder w8g vs fp32: max |dh| {np.abs(hq - h).max():.4f} on |h| max {np.abs(h).max():.2f}")

    def greedy(sess, max_new=48):
        ids = np.zeros((n, 1), dtype=np.int64); done = np.zeros(n, bool)
        for _ in range(max_new):
            lg = sess.run(None, {"ids": ids, "enc_h": h, "enc_mask": np.ones(h.shape[:2], dtype=np.int64)})[0]
            nxt = lg.argmax(-1); nxt[done] = 0
            ids = np.concatenate([ids, nxt[:, None]], 1); done |= nxt == 1
            if done.all():
                break
        return tok.batch_decode(ids, skip_special_tokens=True)

    def vcos(hyps):
        return float((condition(cond, enc.encode(hyps), sb) * emb).sum(1).mean())

    print(f"torch greedy: verifier cosine {vcos(ref):.3f}")
    for suffix in ["", ".w8", ".w8g"]:
        og = greedy(ort.InferenceSession(str(OUT / f"zero_{cond}_dec{suffix}.onnx")))
        agree = sum(a == b for a, b in zip(ref, og))
        print(f"onnx{suffix or ' fp32'}: {agree}/{n} decodes identical to torch, verifier cosine {vcos(og):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["float", "bin1"], default="float")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--verify", type=int, default=0, help="decode this many dev items through ONNX and compare")
    a = ap.parse_args()
    for mode in ["zero", "correct"]:
        if (CKPT / f"{mode}_{a.cond}.pt").exists():
            export(mode, a.cond, a.k)
        else:
            print(f"no checkpoint for {mode}_{a.cond}, skipped")
    if a.verify:
        verify(a.cond, a.k, a.verify)


if __name__ == "__main__":
    main()
