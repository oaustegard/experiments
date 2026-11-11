"""Pack a GLiNER-on-Ettin checkpoint into the browser format.

    python3 run/gpack.py CKPT [--variant fp32|t4-w8L63|t4-w8L31|...] [--label-variant V] [--name NAME]
                         [--out DIR] [--head int8] [--fixtures N] [--fixture-bias B] [--no-sizes]

CKPT is a gliner save (gliner_config.json + pytorch_model.bin or model.safetensors + tokenizer files),
or a hub id already in the local HF cache (HF_HOME, default data/hf), e.g. knowledgator/gliner-bi-edge-v2.0.

Variant names: `fp32`, or `t<T>-<L>` with T the token-table bits (4: groups of 32 with an fp16 scale and
codes 0..15 packed two per byte; 8: groups of 32, fp16 scale, int8 codes in +-127; 32: fp32) and L either
`w8L<lv>` (every matmul int8 per output column, codes in +-lv) or `f32`. t4-w8L63 is browser-decider's build.

Bi-encoder checkpoints (labels_encoder set, e.g. knowledgator/gliner-bi-edge-v2.0) also pack the BERT label
encoder as `label_encoder` (format: see pack_label_encoder) with --label-variant (default: --variant), copy
its tokenizer as labels_tokenizer.json, and drop the tensors the library never uses: the BERT pooler,
token-type row 1, and the BiLSTM (`rnn.lstm.*`), which BiEncoderSpanModel builds but never calls. Writes DIR (default web/models/<name>-<variant>[-h8]/) with
  weights.bin     encoder in browser-decider's packed layout (4-bit token table in groups of 32 with an
                  fp16 scale, all 50371 rows including [FLERT] <<ENT>> <<SEP>>; int8 matmuls per output
                  column with codes in +-63 or +-31; fp32 for the fp32 variant), then the GLiNER head
  manifest.json   browser-decider's encoder schema at the top level (the WGSL engine reads it as is),
                  plus `gliner` (everything inference needs) and `ghead` / `lstm` / `projection`
  tokenizer.json  the checkpoint's tokenizer (with the three added tokens), its ByteLevel
                  add_prefix_space set to the checkpoint's prefix_space flag
  fixtures.json   with --fixtures N: N inputs with token ids, words_mask, logits and entities from GRef

Head weights are fp32 by default; --head int8 stores every head and BiLSTM matrix as int8 per output
column with codes in +-127 (biases stay fp32). Everything is driven by the checkpoint's own
gliner_config.json; a configuration GRef does not implement, or a state-dict tensor the packer did not
place, stops the pack with an error instead of producing a model.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gref  # noqa: E402
from gref import bd  # noqa: E402

D = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(D / "data" / "hf"))
ENC = "token_rep_layer.bert_layer.model."
LENC = "token_rep_layer.labels_encoder.model."
WORD_REGEX = r"\w+(?:[-_]\w+)*|\S"     # gliner data_processing/tokenizer.py WhitespaceTokenSplitter


def resolve_ckpt(ck):
    """A local gliner directory, or a hub id resolved to its snapshot in the local HF cache."""
    p = Path(ck)
    if (p / "gliner_config.json").exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(ck), local_files_only=True))


def ckpt_name(ck, cfg):
    """Short name for output directories: the hub repo name for a cache snapshot, else the config name."""
    parts = Path(ck).resolve().parts
    for x in parts:
        if x.startswith("models--"):
            return x.split("--")[-1]
    return cfg.get("name") or Path(ck).parent.name


def load_checkpoint(ck):
    ck = resolve_ckpt(ck)
    cfg = json.loads((ck / "gliner_config.json").read_text())
    if (ck / "model.safetensors").exists():
        from safetensors.numpy import load_file
        sd = load_file(str(ck / "model.safetensors"))
    else:
        import torch
        sd = {k: v.float().numpy() for k, v in torch.load(ck / "pytorch_model.bin", map_location="cpu",
                                                         weights_only=True).items()}
    return cfg, sd


def is_bi(cfg):
    return bool(cfg.get("labels_encoder"))


def check_supported(cfg):
    """Refuse every configuration GRef does not reproduce. Returns the list of reasons (empty = ok)."""
    bad = []
    ec = cfg.get("encoder_config") or {}
    if ec.get("model_type") != "modernbert":
        bad.append(f"encoder model_type {ec.get('model_type')!r} (only modernbert)")
    if ec.get("hidden_activation", "gelu") != "gelu":
        bad.append(f"hidden_activation {ec.get('hidden_activation')!r}")
    if any(ec.get(k) for k in ("attention_bias", "mlp_bias", "norm_bias")):
        bad.append("encoder biases")
    if cfg.get("span_mode") not in gref.SPAN_MODES:
        bad.append(f"span_mode {cfg.get('span_mode')!r}")
    if cfg.get("span_mode") == "token_level" and cfg.get("represent_spans"):
        bad.append("token_level with represent_spans")
    if cfg.get("subtoken_pooling", "first") not in gref.POOLING:
        bad.append(f"subtoken_pooling {cfg.get('subtoken_pooling')!r}")
    if cfg.get("words_splitter_type", "whitespace") != "whitespace":
        bad.append(f"words_splitter_type {cfg.get('words_splitter_type')!r}")
    for k in ("fuse_layers", "post_fusion_schema", "use_calibrator", "precomputed_prompts_mode",
              "span_encoder_config", "labels_decoder", "relations_layer"):
        if cfg.get(k):
            bad.append(f"{k}={cfg.get(k)!r}")
    if is_bi(cfg):
        lc = cfg.get("labels_encoder_config") or {}
        if lc.get("model_type") != "bert":
            bad.append(f"labels encoder model_type {lc.get('model_type')!r} (only bert)")
        if lc.get("hidden_act") != "gelu" or lc.get("position_embedding_type", "absolute") != "absolute":
            bad.append(f"labels encoder act {lc.get('hidden_act')!r} / positions {lc.get('position_embedding_type')!r}")
        if lc.get("hidden_size") != cfg.get("hidden_size"):
            bad.append("labels_projection (label encoder width != hidden_size)")
        if cfg.get("span_mode") != "markerV0":
            bad.append(f"bi-encoder span_mode {cfg.get('span_mode')!r} (only markerV0)")
    if cfg.get("model_type") not in (None, "gliner"):
        bad.append(f"model_type {cfg.get('model_type')!r}")
    return bad


def parse_variant(v):
    """'fp32' | 't<4|8|32>-<w8L<lv>|f32>' -> (table_bits, matmul code range, 0 = fp32)."""
    if v == "fp32":
        return 32, 0
    mt = re.fullmatch(r"t(4|8|32)-(?:w8L(\d+)|(f32))", v)
    if not mt or (mt.group(2) and not 1 <= int(mt.group(2)) <= 127):
        raise ValueError(f"variant {v!r}: expected fp32 or t<4|8|32>-<w8L<1..127>|f32>")
    return int(mt.group(1)), int(mt.group(2) or 0)


def q_lin(P, W, lv):
    """torch [N, K] -> fp32 record, or int8 [K, N] per output column with codes in +-lv."""
    W = W.astype(np.float32)
    N, K = W.shape
    if not lv:
        return {"f32_t": P.add(W, np.float32), "K": K, "N": N}
    Wk = W.T
    s = (np.abs(Wk).max(0) / lv + 1e-12).astype(np.float32)
    q = np.clip(np.round(Wk / s), -lv, lv).astype(np.int8)
    return {"q": P.add(q, np.int8), "s": P.add(s, np.float32), "K": K, "N": N}


def q_table(P, T, bits):
    """Token table [V, H]: fp32, or groups of 32 per row with an fp16 scale (4-bit: browser-decider's
    codes 0..15 = q+8, two per byte, low nibble first; 8-bit: int8 codes in +-127)."""
    T = T.astype(np.float32)
    V, H = T.shape
    if bits == 32:
        return {"bits": 32, "V": V, "H": H, "f32": P.add(T, np.float32)}
    G = H // 32
    Tg = T.reshape(V, G, 32)
    if bits == 4:
        s = np.abs(Tg).max(2, keepdims=True) / 7 + 1e-12
        q = (np.clip(np.round(Tg / s), -8, 7).astype(np.int16) + 8).astype(np.uint8).reshape(V, H)
        codes = P.add((q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), np.uint8)
    else:
        s = np.abs(Tg).max(2, keepdims=True) / 127 + 1e-12
        codes = P.add(np.clip(np.round(Tg / s), -127, 127).astype(np.int8).reshape(V, H), np.int8)
    return {"bits": bits, "V": V, "H": H, "codes": codes, "scales": P.add(s.reshape(V, G, 1), np.float16)}


def pack_encoder(P, sd, ec, variant, used):
    """browser-decider pack.pack() lines 44-87 with the GLiNER key prefix and without its head."""
    H, I = ec["hidden_size"], ec["intermediate_size"]
    tbits, lv = parse_variant(variant)

    def g(k):
        used.add(ENC + k)
        return sd[ENC + k]

    def lin(W):
        return q_lin(P, W, lv)

    table = q_table(P, g("embeddings.tok_embeddings.weight"), tbits)
    rp = ec.get("rope_parameters")
    rope = ({"global": rp["full_attention"]["rope_theta"], "local": rp["sliding_attention"]["rope_theta"]} if rp else
            {"global": ec["global_rope_theta"], "local": ec["local_rope_theta"] or ec["global_rope_theta"]})
    m = {"arch": "modernbert", "hidden": H, "heads": ec["num_attention_heads"], "inter": I,
         "eps": ec["norm_eps"], "window": ec["local_attention"], "global_every": ec["global_attn_every_n_layers"],
         "theta_global": rope["global"], "theta_local": rope["local"],
         "table": table, "emb_norm": P.add(g("embeddings.norm.weight"), np.float32), "layers": []}
    for i in range(ec["num_hidden_layers"]):
        p = f"layers.{i}."
        m["layers"].append({
            "attn_norm": P.add(g(p + "attn_norm.weight"), np.float32) if ENC + p + "attn_norm.weight" in sd else None,
            "Wqkv": lin(g(p + "attn.Wqkv.weight")), "Wo": lin(g(p + "attn.Wo.weight")),
            "mlp_norm": P.add(g(p + "mlp_norm.weight"), np.float32),
            "Wi": lin(g(p + "mlp.Wi.weight")), "Wo_mlp": lin(g(p + "mlp.Wo.weight")),
            "global": i % ec["global_attn_every_n_layers"] == 0})
    m["final_norm"] = P.add(g("final_norm.weight"), np.float32)
    return m


def pack_label_encoder(P, sd, lc, variant, used):
    """BERT label encoder (HF BertModel, post-LN). Record layout, all in the same weights.bin:
      table      word embeddings, same formats as the text table (bits 32 / 8 / 4, groups of 32)
      pos        fp32 [max_pos, H] absolute position embeddings; type0: fp32 [H], token-type row 0
                 (the library passes no token_type_ids, so row 1 is never read and is not stored)
      emb_ln_w/b fp32 embedding LayerNorm
      layers[i]  Wqkv (q, k, v stacked on the output axis: N = 3H) + bqkv, Wo + bo, ln1_w/b (attention
                 output LayerNorm), Wi + bi (H -> inter, gelu erf), Wo2 + bo2, ln2_w/b; matmuls as q_lin
                 (fp32 [N, K] or int8 [K, N] with per-column fp32 scales), biases and norms fp32
    The pooler is unused (the library takes last_hidden_state and mean-pools it) and is not stored."""
    tbits, lv = parse_variant(variant)
    H = lc["hidden_size"]

    def g(k):
        used.add(LENC + k)
        return sd[LENC + k]

    def f(k):
        return P.add(g(k), np.float32)

    used.update({LENC + "pooler.dense.weight", LENC + "pooler.dense.bias"})
    m = {"arch": "bert", "hidden": H, "heads": lc["num_attention_heads"], "inter": lc["intermediate_size"],
         "eps": lc["layer_norm_eps"], "max_pos": lc["max_position_embeddings"], "act": "gelu (erf)",
         "variant": variant,
         "table": q_table(P, g("embeddings.word_embeddings.weight"), tbits),
         "pos": f("embeddings.position_embeddings.weight"),
         "type0": P.add(g("embeddings.token_type_embeddings.weight")[0], np.float32),
         "emb_ln_w": f("embeddings.LayerNorm.weight"), "emb_ln_b": f("embeddings.LayerNorm.bias"), "layers": []}
    for i in range(lc["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        a = p + "attention."
        Wqkv = np.concatenate([g(a + f"self.{x}.weight") for x in ("query", "key", "value")], 0)
        bqkv = np.concatenate([g(a + f"self.{x}.bias") for x in ("query", "key", "value")], 0)
        m["layers"].append({
            "Wqkv": q_lin(P, Wqkv, lv), "bqkv": P.add(bqkv, np.float32),
            "Wo": q_lin(P, g(a + "output.dense.weight"), lv), "bo": f(a + "output.dense.bias"),
            "ln1_w": f(a + "output.LayerNorm.weight"), "ln1_b": f(a + "output.LayerNorm.bias"),
            "Wi": q_lin(P, g(p + "intermediate.dense.weight"), lv), "bi": f(p + "intermediate.dense.bias"),
            "Wo2": q_lin(P, g(p + "output.dense.weight"), lv), "bo2": f(p + "output.dense.bias"),
            "ln2_w": f(p + "output.LayerNorm.weight"), "ln2_b": f(p + "output.LayerNorm.bias")})
    if H % 32:
        raise ValueError("label encoder hidden size must be a multiple of 32 for the grouped tables")
    return m


def pack(cfg, sd, variant="t4-w8L63", head="fp32", label_variant=None):
    """-> (manifest, weights bytes). label_variant (bi-encoder only) defaults to variant."""
    bad = check_supported(cfg)
    if bad:
        raise NotImplementedError("unsupported checkpoint configuration: " + "; ".join(bad))
    ec = dict(cfg["encoder_config"])
    P = bd.Packer()
    used = set()
    m = pack_encoder(P, sd, ec, variant, used)
    Hc = cfg["hidden_size"]
    hq = head == "int8"
    bi = is_bi(cfg)
    parts = {"encoder": P.off}

    def g(k):
        if k not in sd:
            raise KeyError(f"checkpoint lacks {k} (config says it should exist)")
        used.add(k)
        return sd[k]

    def lin(W):
        return q_lin(P, W, 127 if hq else 0)

    def affine(prefix):
        return {"l": lin(g(prefix + ".weight")), "b": P.add(g(prefix + ".bias"), np.float32)}

    def mlp(prefix):   # create_projection_layer: Sequential index 0 and 3 are the Linears
        return {"l0": lin(g(prefix + ".0.weight")), "b0": P.add(g(prefix + ".0.bias"), np.float32),
                "l1": lin(g(prefix + ".3.weight")), "b1": P.add(g(prefix + ".3.bias"), np.float32)}

    m["projection"] = affine("token_rep_layer.projection") if "token_rep_layer.projection.weight" in sd else None
    if (m["projection"] is None) != (Hc == ec["hidden_size"]):
        raise ValueError(f"hidden_size {Hc} vs encoder {ec['hidden_size']} disagrees with the projection's presence")
    parts["projection"] = P.off
    dropped = []
    if bi:
        label_variant = label_variant or variant
        m["label_encoder"] = pack_label_encoder(P, sd, dict(cfg["labels_encoder_config"]), label_variant, used)
        parts["label_encoder"] = P.off
        dropped += [f"{LENC}pooler.* (unused: the library mean-pools last_hidden_state)",
                    f"{LENC}embeddings.token_type_embeddings row 1 (no token_type_ids are passed)"]
        rnn = sorted(k for k in sd if k.startswith("rnn."))
        if rnn:   # BiEncoderSpanModel builds self.rnn but get_representations never calls it
            used.update(rnn)
            dropped.append(f"rnn.lstm.* ({sum(sd[k].size for k in rnn)} params; built but never applied by the "
                           "bi-encoder forward in gliner 0.2.30 or 0.2.22)")
        nl = 0
    else:
        nl = int(cfg.get("num_rnn_layers", 1) or 0)
    m["lstm"] = []
    for li in range(nl):
        layer = {}
        for d, suf in (("fwd", ""), ("bwd", "_reverse")):
            p = f"rnn.lstm.%s_l{li}{suf}"
            layer[d] = {"w_ih": lin(g(p % "weight_ih")), "w_hh": lin(g(p % "weight_hh")),
                        "b_ih": P.add(g(p % "bias_ih"), np.float32), "b_hh": P.add(g(p % "bias_hh"), np.float32),
                        "hidden": Hc // 2}
        m["lstm"].append(layer)
    parts["lstm"] = P.off
    if cfg["span_mode"] == "markerV0":
        sp = "span_rep_layer.span_rep_layer."
        m["ghead"] = {"prompt": mlp("prompt_rep_layer"), "start": mlp(sp + "project_start"),
                      "end": mlp(sp + "project_end"), "out": mlp(sp + "out_project")}
    else:
        m["ghead"] = {"proj_token": affine("scorer.proj_token"), "proj_label": affine("scorer.proj_label"),
                      "out_mlp": mlp("scorer.out_mlp")}
    parts["ghead"] = P.off
    left = sorted(set(sd) - used)
    if left:
        raise NotImplementedError(f"{len(left)} checkpoint tensors not placed by the packer: {left[:8]}")
    m["variant"] = variant
    m["head_format"] = head
    if bi:
        m["label_variant"] = label_variant
        m["dropped"] = dropped
    m["gliner"] = {
        "span_mode": cfg["span_mode"], "max_width": cfg["max_width"], "max_len": cfg["max_len"],
        "max_types_train": cfg.get("max_types"), "hidden_size": Hc, "num_rnn_layers": nl,
        "class_token_index": cfg["class_token_index"], "vocab_size": cfg["vocab_size"],
        "ent_token": cfg.get("ent_token", "[ENT]"), "sep_token": cfg.get("sep_token", "[SEP]"),
        "embed_ent_token": bool(cfg.get("embed_ent_token", True)),
        "subtoken_pooling": cfg.get("subtoken_pooling", "first"),
        "prefix_space": bool(cfg.get("prefix_space", False)),
        "word_regex": WORD_REGEX, "word_offsets": "Unicode code points, end exclusive",
        "tokenizer": "tokenizer.json", "model_max_length": None,
        "logit_temperature": float(cfg.get("logit_temperature") or 1.0),
        "logit_bias": float(cfg.get("logit_bias") or 0.0),
        "threshold": 0.5, "flat_ner": True, "multi_label": False,
        "labels": "deduplicated first-wins, case kept; class c = c-th <<ENT>>; empty label adds no word",
        "special": {}, "parts_end": parts | {"total": P.off},
    }
    if bi:
        m["gliner"].update({
            "encoder_kind": "bi", "class_token_index": None, "ent_token": None, "sep_token": None,
            "embed_ent_token": None, "prefix_space": None,    # set from the saved tokenizer by write_tokenizer
            "labels_encoder": cfg["labels_encoder"], "label_tokenizer": "labels_tokenizer.json",
            "label_max_length": None, "label_pooling": "mean over all label tokens, [CLS] and [SEP] included",
            "labels": "deduplicated first-wins, case kept (the label tokenizer lowercases); class c = c-th label; "
                      "each label is encoded alone: [CLS] label [SEP] -> BERT -> mean -> prompt_rep_layer",
            "text_input": "[CLS] + subtokens of the text words only + [SEP]; no label prompt in the text"})
    else:
        m["gliner"]["encoder_kind"] = "uni"
    return m, P.bytes()


def write_tokenizer(ck, out, g):
    """Copy the checkpoint tokenizer, set add_prefix_space from the config, fill special ids.
    Bi-encoder: the text tokenizer is used as saved (its add_prefix_space is what the library applies; the
    library also adds [FLERT] <<ENT>> <<SEP>> to it, which a bi-encoder never feeds, so they are not copied),
    and the label encoder's tokenizer is copied as labels_tokenizer.json with the library's call settings
    (truncation to model_max_length, no fixed padding)."""
    tj = json.loads((Path(ck) / "tokenizer.json").read_text())
    tc = json.loads((Path(ck) / "tokenizer_config.json").read_text())
    pt = tj["pre_tokenizer"]
    if pt.get("type") != "ByteLevel":
        raise NotImplementedError(f"pre-tokenizer {pt.get('type')}")
    saved = bool(pt.get("add_prefix_space"))
    bi = g.get("encoder_kind") == "bi"
    if bi:
        g["prefix_space"] = saved
    if saved != g["prefix_space"]:
        raise ValueError(f"checkpoint tokenizer add_prefix_space={saved} but gliner_config prefix_space="
                         f"{g['prefix_space']}; the library would use the tokenizer's, refusing to guess")
    vocab = {a["content"]: a["id"] for a in tj["added_tokens"]}
    vocab.update(tj["model"]["vocab"])
    sp = {k: vocab[tc[k + "_token"]] for k in ("cls", "sep", "pad", "unk", "mask")}
    if not bi:
        sp["prompt_ent"] = vocab[g["ent_token"]]
        sp["prompt_sep"] = vocab[g["sep_token"]]
        sp["flert"] = vocab.get("[FLERT]")
        if sp["prompt_ent"] != g["class_token_index"]:
            raise ValueError(f"class_token_index {g['class_token_index']} != id of {g['ent_token']} "
                             f"({sp['prompt_ent']})")
    g["special"] = sp
    g["model_max_length"] = int(tc.get("model_max_length", 8192))
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    if bi:
        from huggingface_hub import hf_hub_download
        lj = json.loads(Path(hf_hub_download(g["labels_encoder"], "tokenizer.json", local_files_only=True)).read_text())
        lc = json.loads(Path(hf_hub_download(g["labels_encoder"], "tokenizer_config.json",
                                             local_files_only=True)).read_text())
        if lj["model"]["type"] != "WordPiece" or lj["normalizer"]["type"] != "BertNormalizer":
            raise NotImplementedError(f"label tokenizer {lj['model']['type']} / {lj['normalizer']['type']}")
        ml = int(lc["model_max_length"])
        lj["truncation"] = {"direction": "Right", "max_length": ml, "strategy": "LongestFirst", "stride": 0}
        lj["padding"] = None
        g["label_max_length"] = ml
        lv = lj["model"]["vocab"]
        g["label_special"] = {k: lv[lc[k + "_token"]] for k in ("cls", "sep", "pad", "unk")}
        (out / g["label_tokenizer"]).write_text(json.dumps(lj, ensure_ascii=False))


def part_bounds(m):
    """[(name, start, end)] byte ranges of weights.bin by part, token tables split out of their encoders.
    Uni-encoder names are the original ones (token_table, encoder_layers, projection, lstm, ghead)."""
    pe = m["gliner"]["parts_end"]
    bi = m["gliner"].get("encoder_kind") == "bi"
    names = ({"encoder": ("text_token_table", "text_layers"), "label_encoder": ("label_token_table", "label_layers"),
              "projection": "projection", "lstm": "bilstm", "ghead": "span_head"} if bi else
             {"encoder": ("token_table", "encoder_layers"), "projection": "projection", "lstm": "lstm",
              "ghead": "ghead"})

    def tab_end(tab):
        recs = [tab["codes"], tab["scales"]] if tab["bits"] in (4, 8) else [tab["f32"]]
        return max(r["offset"] + r["length"] for r in recs)
    res, a = [], 0
    for k in ("encoder", "projection", "label_encoder", "lstm", "ghead"):
        if k not in pe:
            continue
        b = pe[k]
        if k in ("encoder", "label_encoder"):
            e = tab_end(m["table"] if k == "encoder" else m["label_encoder"]["table"])
            res += [(names[k][0], a, e), (names[k][1], e, b)]
        else:
            res.append((names[k], a, b))
        a = b
    return res


def sizes(m, data, out, quality=None):
    import brotli
    bounds = part_bounds(m)
    # brotli-11 runs at well under 1 MB/s; the fp32 builds (over 30 MB) are measured at quality 9 (stated)
    qual = quality or (11 if len(data) < 30e6 else 9)
    res = {"brotli_quality": qual}

    def br(b):
        return len(brotli.compress(b, quality=qual)) if b else 0
    for name, a, b in bounds:
        res[name] = {"raw": b - a, "brotli": br(data[a:b])}
    res["weights_total"] = {"raw": len(data), "brotli": sum(res[k]["brotli"] for k, _, _ in bounds)}
    files = ["tokenizer.json"] + ([m["gliner"]["label_tokenizer"]] if m["gliner"].get("encoder_kind") == "bi" else [])
    for f in files + ["manifest.json"]:
        b = (out / f).read_bytes()
        res[f] = {"raw": len(b), "brotli": br(b)}
    res["download_total"] = {k: res["weights_total"][k] + sum(res[f][k] for f in files + ["manifest.json"])
                             for k in ("raw", "brotli")}
    return res


FIXTURE_INPUTS = [
    ("Typical generative model approaches include naive Bayes classifiers , Gaussian mixture models and others .",
     ["algorithm", "product", "researcher"]),
    ("John Smith works at Google in Zürich, isn't it?", ["person", "organization name", "city"]),
    ("Björk Guðmundsdóttir released Homogenic in 1997 — produced with Mark Bell (LFO).",
     ["musical artist", "album", "person", "band", "year"]),
    ("The U.S. Senate passed the state-of-the-art bill 52-48 on 3/4/2021.", ["organization", "country", "event"]),
    ("Barack Obama met Angela Merkel in Berlin.", ["person", "person", "location", ""]),
    ("Marie Curie discovered polonium and radium at the University of Paris .",
     ["scientist", "chemical element", "university", "award", "theory", "discipline", "country", "person",
      "location", "organization", "event", "chemical compound", "protein", "enzyme", "astronomical object",
      "academic journal", "else", "misc", "title", "date", "genre", "rating", "price", "dish", "cuisine",
      "amenity", "hours", "director"]),
    ("show me a 1990s comedy starring jim carrey", ["genre", "year", "actor", "title"]),
    ("any cheap thai places open after 10 pm near downtown", ["price", "cuisine", "hours", "location", "amenity"]),
    ("東京 is the capital of Japan; 大阪 is not.", ["city", "country"]),
    ("Emoji 🙂 test: Apple's iPhone 15 Pro", ["product", "organization"]),
    ("A", ["letter"]),
    ("Lisp, Prolog and Python are programming languages used in AI research at MIT .",
     ["program language", "field", "university"]),
]


def source_info(arg, ck, name):
    ck = Path(ck)
    if "snapshots" in ck.parts:
        return {"checkpoint": f"{arg}@{ck.name}", "name": name}
    return {"checkpoint": ck.name if ck.name != "best" else f"{ck.parent.name}/best", "name": name}


def write_fixtures(out, n, bias):
    """GRef fp32-path fixtures: ids, words_mask, logits (valid spans), entities at 0.5 / 0.3, and the
    same decode with logit_bias=bias so the decoder is exercised even when the checkpoint predicts nothing."""
    g = gref.load(out)
    cases = []
    lab_fx = {}
    for text, labels in FIXTURE_INPUTS[:n]:
        ents, info = g.predict_entities(text, labels, 0.5, return_logits=True)
        labs = g.norm_labels(labels)
        z = info["z"]
        nw = len(info["words"])
        if g.g["span_mode"] == "markerV0":
            zl = [[[float(v) for v in z[s, k]] if s + k < nw else None for k in range(z.shape[1])] for s in range(nw)]
        else:
            zl = z.tolist()
        case = {"text": text, "labels_in": labels, "labels": labs,
                "words": [list(w) for w in g.split(text)][: g.g["max_len"]],
                "input_ids": info["ids"], "words_mask": info["words_mask"], "prompt_len": info["prompt_len"],
                "label_pos": info["label_pos"], "logits": zl, "entities": {}}
        if g.bi:
            case["label_input_ids"] = info["label_ids"]
            for lab, ids in zip(labs, info["label_ids"]):
                if lab not in lab_fx:
                    lab_fx[lab] = {"input_ids": ids, "pooled": [float(v) for v in g.lenc.pooled(ids)]}
        for t in (0.5, 0.3):
            case["entities"][str(t)] = g.predict_entities(text, labels, t)
        b0 = g.b
        g.b = b0 + bias
        case["entities_bias"] = {str(t): g.predict_entities(text, labels, t) for t in (0.5, 0.3)}
        g.b = b0
        cases.append(case)
    fx = {"variant": g.m["variant"], "head_format": g.m["head_format"], "logit_shape": (
        "markerV0: [n_words][max_width][C], null where start+width >= n_words" if g.g["span_mode"] == "markerV0"
        else "token_level: [n_words][C][3] (start, end, inside)"),
        "bias_test": {"extra_logit_bias": bias, "note": "entities_bias decoded with logit_bias += extra_logit_bias"},
        "tolerance": {"logits_abs": 1e-4, "score_abs": 1e-4}, "cases": cases}
    if g.bi:
        fx["label_variant"] = g.m["label_variant"]
        fx["label_encoder"] = {
            "note": "per distinct label: label-tokenizer ids ([CLS] ... [SEP]) and the mean-pooled BERT output "
                    "(before prompt_rep_layer); `first_label_hidden` is the full last_hidden_state of the first",
            "labels": lab_fx}
        first = next(iter(lab_fx.values()), None)
        if first:
            fx["label_encoder"]["first_label_hidden"] = g.lenc.encode(first["input_ids"]).tolist()
    (out / "fixtures.json").write_text(json.dumps(fx, ensure_ascii=False))
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="gliner checkpoint directory or a cached hub id")
    ap.add_argument("--variant", default="t4-w8L63", help="fp32 | t<4|8|32>-<w8L<lv>|f32> (text encoder)")
    ap.add_argument("--label-variant", default=None, help="bi-encoder label encoder, same syntax (default: --variant)")
    ap.add_argument("--head", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--name", default=None, help="default: hub repo name, else the checkpoint's config name")
    ap.add_argument("--no-sizes", action="store_true", help="skip the brotli size table")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fixtures", type=int, default=0)
    ap.add_argument("--fixture-bias", type=float, default=0.0,
                    help="extra logit bias for the fixtures' second decode (choose so some spans pass)")
    a = ap.parse_args()
    parse_variant(a.variant)
    ck = resolve_ckpt(a.ckpt)
    cfg, sd = load_checkpoint(ck)
    name = a.name or ckpt_name(ck, cfg)
    lvar = a.label_variant if is_bi(cfg) else None
    if lvar:
        parse_variant(lvar)
    suffix = (f"-l{lvar}" if lvar and lvar != a.variant else "") + ("-h8" if a.head == "int8" else "")
    out = a.out or D / "web" / "models" / f"{name}-{a.variant}{suffix}"
    m, data = pack(cfg, sd, a.variant, a.head, lvar)
    out.mkdir(parents=True, exist_ok=True)
    write_tokenizer(ck, out, m["gliner"])
    m["gliner"]["source"] = source_info(a.ckpt, ck, name)
    (out / "weights.bin").write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(m))
    sz = {} if a.no_sizes else sizes(m, data, out)
    if sz:
        (out / "sizes.json").write_text(json.dumps(sz, indent=1))
    print(f"packed {out.name} ({cfg['span_mode']}, rnn {m['gliner']['num_rnn_layers']}, prefix_space "
          f"{m['gliner']['prefix_space']}, head {a.head})")
    for k, v in sz.items():
        if not isinstance(v, dict):
            continue
        print(f"  {k:16} {v['raw'] / 1e6:9.3f} MB raw  {v['brotli'] / 1e6:9.3f} MB brotli")
    if a.fixtures:
        fx = write_fixtures(out, a.fixtures, a.fixture_bias)
        ne = sum(len(c["entities_bias"]["0.5"]) for c in fx["cases"])
        print(f"fixtures: {len(fx['cases'])} cases, {ne} entities at 0.5 with bias +{a.fixture_bias}")


if __name__ == "__main__":
    main()
