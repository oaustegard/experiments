"""Pack a GLiNER-on-Ettin checkpoint into the browser format.

    python3 run/gpack.py CKPT_DIR [--variant fp32|t4-w8L63|t4-w8L31] [--name NAME] [--out DIR]
                         [--head int8] [--fixtures N] [--fixture-bias B]

CKPT_DIR is a gliner save (gliner_config.json + pytorch_model.bin or model.safetensors + tokenizer
files). Writes DIR (default web/models/<name>-<variant>[-h8]/) with
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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gref  # noqa: E402
from gref import GRef, bd  # noqa: E402

D = Path(__file__).resolve().parents[1]
ENC = "token_rep_layer.bert_layer.model."
WORD_REGEX = r"\w+(?:[-_]\w+)*|\S"     # gliner data_processing/tokenizer.py WhitespaceTokenSplitter


def load_checkpoint(ck):
    ck = Path(ck)
    cfg = json.loads((ck / "gliner_config.json").read_text())
    if (ck / "model.safetensors").exists():
        from safetensors.numpy import load_file
        sd = load_file(str(ck / "model.safetensors"))
    else:
        import torch
        sd = {k: v.float().numpy() for k, v in torch.load(ck / "pytorch_model.bin", map_location="cpu",
                                                         weights_only=True).items()}
    return cfg, sd


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
              "span_encoder_config", "labels_encoder", "labels_decoder", "relations_layer"):
        if cfg.get(k):
            bad.append(f"{k}={cfg.get(k)!r}")
    if cfg.get("model_type") not in (None, "gliner"):
        bad.append(f"model_type {cfg.get('model_type')!r}")
    return bad


def pack_encoder(P, sd, ec, variant, used):
    """browser-decider pack.pack() lines 44-87 with the GLiNER key prefix and without its head."""
    H, I = ec["hidden_size"], ec["intermediate_size"]
    q8 = variant != "fp32"
    lv = int(variant.rsplit("L", 1)[1]) if q8 else 0

    def g(k):
        used.add(ENC + k)
        return sd[ENC + k]

    def lin(W):
        W = W.astype(np.float32)
        N, K = W.shape
        if not q8:
            return {"f32_t": P.add(W, np.float32), "K": K, "N": N}
        Wk = W.T
        s = (np.abs(Wk).max(0) / lv + 1e-12).astype(np.float32)
        q = np.clip(np.round(Wk / s), -lv, lv).astype(np.int8)
        return {"q": P.add(q, np.int8), "s": P.add(s, np.float32), "K": K, "N": N}

    T = g("embeddings.tok_embeddings.weight").astype(np.float32)
    V = T.shape[0]
    if q8:
        G = H // 32
        Tg = T.reshape(V, G, 32)
        s = np.abs(Tg).max(2, keepdims=True) / 7 + 1e-12
        q = (np.clip(np.round(Tg / s), -8, 7).astype(np.int16) + 8).astype(np.uint8).reshape(V, H)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        table = {"bits": 4, "V": V, "H": H, "codes": P.add(packed, np.uint8),
                 "scales": P.add(s.reshape(V, G, 1), np.float16)}
    else:
        table = {"bits": 32, "V": V, "H": H, "f32": P.add(T, np.float32)}
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


def pack(cfg, sd, variant="t4-w8L63", head="fp32"):
    bad = check_supported(cfg)
    if bad:
        raise NotImplementedError("unsupported checkpoint configuration: " + "; ".join(bad))
    ec = dict(cfg["encoder_config"])
    P = bd.Packer()
    used = set()
    m = pack_encoder(P, sd, ec, variant, used)
    Hc = cfg["hidden_size"]
    hq = head == "int8"
    parts = {"encoder": P.off}

    def g(k):
        if k not in sd:
            raise KeyError(f"checkpoint lacks {k} (config says it should exist)")
        used.add(k)
        return sd[k]

    def lin(W):
        W = W.astype(np.float32)
        N, K = W.shape
        if not hq:
            return {"f32_t": P.add(W, np.float32), "K": K, "N": N}
        Wk = W.T
        s = (np.abs(Wk).max(0) / 127 + 1e-12).astype(np.float32)
        q = np.clip(np.round(Wk / s), -127, 127).astype(np.int8)
        return {"q": P.add(q, np.int8), "s": P.add(s, np.float32), "K": K, "N": N}

    def affine(prefix):
        return {"l": lin(g(prefix + ".weight")), "b": P.add(g(prefix + ".bias"), np.float32)}

    def mlp(prefix):   # create_projection_layer: Sequential index 0 and 3 are the Linears
        return {"l0": lin(g(prefix + ".0.weight")), "b0": P.add(g(prefix + ".0.bias"), np.float32),
                "l1": lin(g(prefix + ".3.weight")), "b1": P.add(g(prefix + ".3.bias"), np.float32)}

    m["projection"] = affine("token_rep_layer.projection") if "token_rep_layer.projection.weight" in sd else None
    if (m["projection"] is None) != (Hc == ec["hidden_size"]):
        raise ValueError(f"hidden_size {Hc} vs encoder {ec['hidden_size']} disagrees with the projection's presence")
    parts["projection"] = P.off
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
    return m, P.bytes()


def write_tokenizer(ck, out, g):
    """Copy the checkpoint tokenizer, set add_prefix_space from the config, fill special ids."""
    tj = json.loads((Path(ck) / "tokenizer.json").read_text())
    tc = json.loads((Path(ck) / "tokenizer_config.json").read_text())
    pt = tj["pre_tokenizer"]
    if pt.get("type") != "ByteLevel":
        raise NotImplementedError(f"pre-tokenizer {pt.get('type')}")
    saved = bool(pt.get("add_prefix_space"))
    if saved != g["prefix_space"]:
        raise ValueError(f"checkpoint tokenizer add_prefix_space={saved} but gliner_config prefix_space="
                         f"{g['prefix_space']}; the library would use the tokenizer's, refusing to guess")
    vocab = {a["content"]: a["id"] for a in tj["added_tokens"]}
    vocab.update(tj["model"]["vocab"])
    sp = {k: vocab[tc[k + "_token"]] for k in ("cls", "sep", "pad", "unk", "mask")}
    sp["prompt_ent"] = vocab[g["ent_token"]]
    sp["prompt_sep"] = vocab[g["sep_token"]]
    sp["flert"] = vocab.get("[FLERT]")
    if sp["prompt_ent"] != g["class_token_index"]:
        raise ValueError(f"class_token_index {g['class_token_index']} != id of {g['ent_token']} ({sp['prompt_ent']})")
    g["special"] = sp
    g["model_max_length"] = int(tc.get("model_max_length", 8192))
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))


def sizes(m, data, out):
    import brotli
    pe = m["gliner"]["parts_end"]
    bounds = [("encoder", 0, pe["encoder"]), ("projection", pe["encoder"], pe["projection"]),
              ("lstm", pe["projection"], pe["lstm"]), ("ghead", pe["lstm"], pe["ghead"])]
    tab = m["table"]
    tab_end = max(r["offset"] + r["length"] for r in ([tab["codes"], tab["scales"]] if tab["bits"] == 4 else [tab["f32"]]))
    bounds = [("token_table", 0, tab_end), ("encoder_layers", tab_end, pe["encoder"])] + bounds[1:]
    # brotli-11 runs at well under 1 MB/s; the 77 MB fp32 build is measured at quality 9 (stated in the output)
    qual = 11 if len(data) < 30e6 else 9
    res = {"brotli_quality": qual}

    def br(b):
        return len(brotli.compress(b, quality=qual)) if b else 0
    for name, a, b in bounds:
        res[name] = {"raw": b - a, "brotli": br(data[a:b])}
    res["weights_total"] = {"raw": len(data), "brotli": sum(res[k]["brotli"] for k, _, _ in bounds)}
    for f in ("tokenizer.json", "manifest.json"):
        b = (out / f).read_bytes()
        res[f] = {"raw": len(b), "brotli": br(b)}
    res["download_total"] = {k: res["weights_total"][k] + res["tokenizer.json"][k] + res["manifest.json"][k]
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


def write_fixtures(out, n, bias):
    """GRef fp32-path fixtures: ids, words_mask, logits (valid spans), entities at 0.5 / 0.3, and the
    same decode with logit_bias=bias so the decoder is exercised even when the checkpoint predicts nothing."""
    g = GRef(out)
    cases = []
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
    (out / "fixtures.json").write_text(json.dumps(fx, ensure_ascii=False))
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--variant", default="t4-w8L63", choices=["t4-w8L63", "t4-w8L31", "fp32"])
    ap.add_argument("--head", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--name", default=None, help="default: the checkpoint's config name")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fixtures", type=int, default=0)
    ap.add_argument("--fixture-bias", type=float, default=0.0,
                    help="extra logit bias for the fixtures' second decode (choose so some spans pass)")
    a = ap.parse_args()
    cfg, sd = load_checkpoint(a.ckpt)
    name = a.name or cfg.get("name") or a.ckpt.parent.name
    out = a.out or D / "web" / "models" / f"{name}-{a.variant}{'-h8' if a.head == 'int8' else ''}"
    m, data = pack(cfg, sd, a.variant, a.head)
    out.mkdir(parents=True, exist_ok=True)
    write_tokenizer(a.ckpt, out, m["gliner"])
    m["gliner"]["source"] = {"checkpoint": a.ckpt.name if a.ckpt.name != "best" else f"{a.ckpt.parent.name}/best",
                             "name": name}
    (out / "weights.bin").write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(m))
    sz = sizes(m, data, out)
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
