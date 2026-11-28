"""Convert Qwen3-0.6B (Hugging Face safetensors) into the kernel's flat model blob,
and compute PyTorch reference outputs for checking the C forward pass.

Every matrix, the embedding table included, is quantized to Q4_0 with gguf-py
(llama.cpp's Python quantizers), the equivalent of `llama-quantize --pure`.
Norm weights stay f32. RoPE cos/sin are precomputed for MAX_CTX positions the
way transformers computes them, so the kernel needs no trigonometry.

The reference model is transformers' Qwen3 in float32 with the same
dequantized Q4_0 weights, so a difference between it and the C code is a bug
in the C code or the Q8_0 activations and f16 KV cache the C code adds. The
original bf16 weights give the quantization's own cost.

Writes work/qwen3-0.6b-q4_0.bin, work/ref.npz and results/convert.json.
"""
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType as T
from gguf.quants import quantize, dequantize
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "Qwen3-0.6B"
OUT = HERE / "work" / "qwen3-0.6b-q4_0.bin"
MAX_CTX = 512
PROMPT = "The capital of France is Paris. The capital of Germany is"
N_EXPECT = 16
CHECK_K = 8
EVAL_TEXT = (
    "Ravens are large black birds found across the Northern Hemisphere. They live in forests, "
    "mountains, deserts and cities, and they eat almost anything: carrion, insects, grain, eggs "
    "and food left by people. Ravens are among the most intelligent birds. In experiments they "
    "plan ahead, use tools and remember the faces of people who treated them badly. Pairs often "
    "stay together for life and defend a large territory. In Norse mythology the god Odin kept "
    "two ravens, Huginn and Muninn, who flew over the world each day and brought him news."
)
ALIGN = 64


def bytes_to_unicode():
    """GPT-2's byte <-> printable-unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(tok, vocab_size):
    dec = {c: b for b, c in bytes_to_unicode().items()}
    added = {i for i in tok.added_tokens_decoder}
    out = []
    for i in range(vocab_size):
        s = tok.convert_ids_to_tokens(i) if i < len(tok) else None
        if s is None:
            out.append(b"")
        elif i in added:
            out.append(s.encode("utf-8"))
        else:
            out.append(bytes(dec[c] for c in s))
    return out


def rope_table(cfg):
    hd, theta = cfg["head_dim"], float(cfg["rope_theta"])
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = torch.arange(MAX_CTX, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]                  # [ctx, hd/2], float32 as in transformers
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1).numpy().astype(np.float32)   # [ctx, hd/2, 2]


def greedy(model, ids, n):
    ids = list(ids)
    for _ in range(n):
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0, -1]
        ids.append(int(logits.argmax()))
    return ids


def logits_of(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float().numpy()


def main():
    t0 = time.time()
    cfg = json.loads((SRC / "config.json").read_text())
    assert cfg["tie_word_embeddings"], "the blob assumes the output layer shares the embedding table"
    tok = AutoTokenizer.from_pretrained(SRC)
    sd = load_file(SRC / "model.safetensors")
    nl, dim, nh, nkv, hd, ffn, vocab = (cfg[k] for k in (
        "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "vocab_size"))

    parts, deq = [], {}
    cursor = [4096]  # header and layer table live in the first 4 KB

    def add(data: bytes) -> int:
        off = (cursor[0] + ALIGN - 1) // ALIGN * ALIGN
        parts.append((off, data))
        cursor[0] = off + len(data)
        return off

    def q4(name):
        w = sd[name].float().numpy()
        q = quantize(w, T.Q4_0)
        deq[name] = torch.from_numpy(dequantize(q, T.Q4_0).reshape(w.shape).astype(np.float32))
        return add(q.tobytes())

    def f32(name):
        return add(sd[name].float().numpy().astype(np.float32).tobytes())

    off_tok = q4("model.embed_tokens.weight")
    off_norm = f32("model.norm.weight")
    layers = []
    for i in range(nl):
        p = f"model.layers.{i}."
        layers.append([
            f32(p + "input_layernorm.weight"),
            q4(p + "self_attn.q_proj.weight"), q4(p + "self_attn.k_proj.weight"), q4(p + "self_attn.v_proj.weight"),
            f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"),
            q4(p + "self_attn.o_proj.weight"),
            f32(p + "post_attention_layernorm.weight"),
            q4(p + "mlp.gate_proj.weight"), q4(p + "mlp.up_proj.weight"), q4(p + "mlp.down_proj.weight"),
        ])
    off_rope = add(rope_table(cfg).tobytes())
    tb = token_bytes(tok, vocab)
    idx = np.zeros(vocab + 1, dtype=np.uint32)
    idx[1:] = np.cumsum([len(b) for b in tb])
    off_vidx = add(idx.tobytes())
    off_vbytes = add(b"".join(tb))
    t_quant = time.time() - t0

    # References.
    prompt_ids = tok(PROMPT)["input_ids"]
    eval_ids = tok(EVAL_TEXT)["input_ids"]
    model = Qwen3ForCausalLM.from_pretrained(SRC, torch_dtype=torch.float32).eval()
    orig_gen = greedy(model, prompt_ids, N_EXPECT)
    orig_eval = logits_of(model, eval_ids)
    with torch.no_grad():
        for name, w in deq.items():
            mod_name = name[: -len(".weight")]
            model.get_submodule(mod_name).weight.copy_(w)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "lm_head not tied"
    q4_gen = greedy(model, prompt_ids, N_EXPECT)
    q4_gen_logits = logits_of(model, q4_gen)
    q4_eval = logits_of(model, eval_ids)

    off_prompt = add(np.array(prompt_ids, dtype=np.uint32).tobytes())
    off_expect = add(np.array(q4_gen[len(prompt_ids):], dtype=np.uint32).tobytes())
    off_expect_dev = add(np.zeros(N_EXPECT, dtype=np.uint32).tobytes())   # filled by `host_run BLOB gen --write-expect-dev`
    off_check = add(np.zeros((len(prompt_ids) + N_EXPECT) * CHECK_K * 2, dtype=np.uint32).tobytes())   # filled by host_run too
    off_layers = 512   # inside the first 4 KB
    total = cursor[0]

    blob = bytearray(total)
    hdr = struct.pack("<8s8If2I12I", b"QW3M0001", nl, dim, nh, nkv, hd, ffn, vocab, MAX_CTX,
                      float(cfg["rms_norm_eps"]), len(prompt_ids), N_EXPECT,
                      off_tok, off_norm, off_rope, off_vidx, off_vbytes, off_prompt, off_expect, off_expect_dev, off_check, CHECK_K, off_layers, total)
    blob[: len(hdr)] = hdr
    lt = np.array(layers, dtype=np.uint32).tobytes()
    assert off_layers + len(lt) <= 4096
    blob[off_layers: off_layers + len(lt)] = lt
    for off, data in parts:
        blob[off: off + len(data)] = data
    OUT.write_bytes(blob)

    np.savez(HERE / "work" / "ref.npz",
             gen_ids=np.array(q4_gen, dtype=np.uint32), gen_logits=q4_gen_logits,
             eval_ids=np.array(eval_ids, dtype=np.uint32), eval_logits=q4_eval, eval_logits_orig=orig_eval)

    top1_q4 = q4_eval.argmax(-1)
    top1_orig = orig_eval.argmax(-1)
    summary = {
        "blob_bytes": total,
        "max_ctx": MAX_CTX,
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "q4_greedy_text": tok.decode(q4_gen[len(prompt_ids):]),
        "orig_greedy_text": tok.decode(orig_gen[len(prompt_ids):]),
        "greedy_tokens_same_as_bf16": int(sum(a == b for a, b in zip(q4_gen, orig_gen)) - len(prompt_ids)),
        "eval_tokens": len(eval_ids),
        "eval_top1_agreement_q4_vs_bf16": float((top1_q4 == top1_orig).mean()),
        "seconds_quantize": round(t_quant, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "convert.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
