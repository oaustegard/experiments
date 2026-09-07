"""Export the four graphs the browser demo needs, in fp32 / fp16 / int8.

    python3 export_onnx.py [--outdir model] [--opset 18] [--no-int8]

Graphs
    lower.onnx       embeddings + decoder layers 0..16  -> hidden16 + present kv
    upper.onnx       decoder layers 17..29 + norm + lm_head -> logits + kv
    query_head.onnx  AttnQueryHead: hidden16 + mask -> op logits, slot logits
    encoder.onnx     ResultEncoder: symbols[1,14] + step[1] -> vec[1,576]

Hosting shape.  GitHub release assets are not fetchable cross-origin, so the
weights have to sit on Hugging Face `resolve` URLs or on raw.githubusercontent,
and the latter caps a file at 100 MB.  So every graph is written as a small
.onnx plus external-data files of at most SHARD_LIMIT bytes, and any single
file still over the cap is additionally cut into `<file>.000`, `.001`, ...
pieces the browser concatenates.  meta.json carries the manifest.

SmolLM2 ties lm_head to the input embeddings.  The 49152x576 table is written
ONCE as `embed[.variant].data` and both graphs' initializers point at it, which
is why llama_min.Upper keeps the weight in [vocab, hidden] orientation (the
dynamo exporter then emits it verbatim instead of pre-transposing it).

Every graph is validated against its torch reference with onnxruntime on CPU.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_utils as mu
import query_head as QH

import llama_min as LM

K = 16
OPSET = 18
SHARD_LIMIT = 90_000_000        # bytes; hosts cap a file at 100 MB
MIN_EXTERNAL = 1_000_000        # smaller initializers stay inside the .onnx
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ exporting
def try_export(mod, args, path, input_names, output_names, dynamic_axes,
               opset=OPSET):
    """dynamo first, legacy exporter as the fallback.  Returns which worked."""
    errs = {}
    try:
        torch.onnx.export(mod, args, path, input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=opset,
                          dynamo=True, external_data=False)
        onnx.checker.check_model(path, full_check=False)
        return "dynamo", errs
    except Exception as e:                                   # noqa: BLE001
        errs["dynamo"] = f"{type(e).__name__}: {str(e)[:200]}"
    torch.onnx.export(mod, args, path, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=opset, dynamo=False)
    return "legacy", errs


def sess(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (float(np.abs(a - b).max()),
            float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9)))


# --------------------------------------------------- external data + sharding
def _bytes_of(t):
    return onnx.numpy_helper.to_array(t).tobytes()


def _set_external(t, loc, off, ln):
    t.ClearField("raw_data")
    for f in ("float_data", "int32_data", "int64_data", "double_data",
              "uint64_data", "string_data"):
        del getattr(t, f)[:]
    t.data_location = onnx.TensorProto.EXTERNAL
    del t.external_data[:]
    for k, v in (("location", loc), ("offset", str(off)), ("length", str(ln))):
        e = t.external_data.add()
        e.key, e.value = k, v


def find_shared_table(model, dims):
    """The [vocab, hidden] tied table, as (tensor, bytes)."""
    for t in model.graph.initializer:
        if list(t.dims) == list(dims):
            return t, _bytes_of(t)
    return None, None


def retie_transposed(model, dims, dtype):
    """Fallback when the legacy exporter pre-transposed the lm_head table.

    It stores [hidden, vocab], whose bytes no longer match the embedding
    table, so the graph is rewritten to hold the [vocab, hidden] tensor plus a
    Transpose in front of the MatMul.  ORT constant-folds that Transpose at
    session load, so this costs load-time memory, not download bytes.
    Returns True if a rewrite happened.
    """
    v, h = dims
    for t in list(model.graph.initializer):
        if list(t.dims) != [h, v] or t.data_type != dtype:
            continue
        arr = onnx.numpy_helper.to_array(t)
        vh = np.ascontiguousarray(arr.T)
        name = t.name + "__vh"
        model.graph.initializer.remove(t)
        model.graph.initializer.append(onnx.numpy_helper.from_array(vh, name))
        model.graph.node.insert(0, onnx.helper.make_node(
            "Transpose", [name], [t.name], perm=[1, 0], name=t.name + "__tp"))
        return True
    return False


def externalize(model, outdir, prefix, shared, shared_shapes=None):
    """Move initializers >= MIN_EXTERNAL into files of <= SHARD_LIMIT bytes.

    `shared` maps sha1(bytes) -> location for tensors another graph already
    wrote (the tied embedding table); `shared_shapes` maps (dims, dtype) ->
    location for the same table when the two copies are numerically equal but
    not byte-identical (the fp16 converter does not round the Gather data and
    the MatMul weight to exactly the same bytes).  Returns (locations written
    here, shared locations referenced).
    """
    shared_shapes = shared_shapes or {}
    bufs, order, used_shared = {}, [], set()
    cur = None
    for t in model.graph.initializer:
        b = _bytes_of(t)
        if len(b) < MIN_EXTERNAL:
            continue
        h = hashlib.sha1(b).hexdigest()
        key = (tuple(t.dims), t.data_type)
        loc = shared.get(h) or shared_shapes.get(key)
        if loc:
            _set_external(t, loc, 0, len(b))
            used_shared.add(loc)
            continue
        assert len(b) <= SHARD_LIMIT, (
            f"{t.name} is {len(b) / 1e6:.0f} MB, over the shard limit and not "
            "shared -- it would need its own split location")
        if cur is None or len(bufs[cur]) + len(b) > SHARD_LIMIT:
            cur = f"{prefix}.w{len(order):03d}"
            order.append(cur)
            bufs[cur] = bytearray()
        _set_external(t, cur, len(bufs[cur]), len(b))
        bufs[cur] += b
    for loc in order:
        with open(os.path.join(outdir, loc), "wb") as f:
            f.write(bufs[loc])
    return order, sorted(used_shared)


def shard_file(outdir, name, limit=SHARD_LIMIT):
    """Cut a file over the host cap into `<name>.000`, `.001`, ... pieces."""
    path = os.path.join(outdir, name)
    if os.path.getsize(path) <= limit:
        return [name]
    names = []
    with open(path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(limit)
            if not chunk:
                break
            nm = f"{name}.{i:03d}"
            with open(os.path.join(outdir, nm), "wb") as g:
                g.write(chunk)
            names.append(nm)
            i += 1
    return names


def manifest_for(outdir, model_name, locations):
    return {
        "model": model_name,
        "model_bytes": os.path.getsize(os.path.join(outdir, model_name)),
        "external": [
            {"location": loc,
             "bytes": os.path.getsize(os.path.join(outdir, loc)),
             "shards": shard_file(outdir, loc)}
            for loc in locations],
    }


def total_mb(man):
    return (man["model_bytes"] + sum(e["bytes"] for e in man["external"])) / 1e6


# ------------------------------------------------------------------ variants
def make_fp16(src, dst):
    """fp16 weights, keeping the graph's inputs and outputs fp32.

    onnxconverter_common's converter leaves this graph type-inconsistent -- the
    fp32 RMSNorm (`x.float() ... .to(dtype)`) becomes an Add whose two operands
    end up float and float16, and the session then refuses to build.  ORT's own
    converter inserts the casts correctly, so it is the primary here and the
    other one is only the fallback.  Either way emit_variant loads the result
    with onnxruntime before it is kept.
    """
    errs = []
    from onnxruntime.transformers import float16 as ortf16
    for name, fn in (
        ("onnxruntime.transformers",
         lambda: ortf16.convert_float_to_float16(onnx.load(src),
                                                 keep_io_types=True)),
        ("onnxconverter_common", _occ_fp16(src)),
    ):
        try:
            m16 = fn()
            onnx.save(m16, dst)
            ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
            return name
        except Exception as e:                               # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__} {str(e)[:160]}")
    raise RuntimeError("no working fp16 conversion; " + " | ".join(errs))


def _occ_fp16(src):
    def go():
        from onnxconverter_common import float16
        return float16.convert_float_to_float16(onnx.load(src),
                                                keep_io_types=True,
                                                disable_shape_infer=True)
    return go


def make_int8(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], extra_options={})


def emit_variant(raw_lo, raw_up, outdir, cfg, variant, report):
    """Write one weight variant of the two halves, deduplicating the table."""
    suf = "" if variant == "fp32" else "." + variant
    tmp = os.path.join(outdir, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    src_lo, src_up = raw_lo, raw_up
    if variant == "fp16":
        src_lo, src_up = f"{tmp}/lo16.onnx", f"{tmp}/up16.onnx"
        how = make_fp16(raw_lo, src_lo)
        make_fp16(raw_up, src_up)
        print(f"  fp16 conversion via {how}", flush=True)
    elif variant == "int8":
        src_lo, src_up = f"{tmp}/lo8.onnx", f"{tmp}/up8.onnx"
        make_int8(raw_lo, src_lo)
        make_int8(raw_up, src_up)

    dims = [cfg["vocab_size"], cfg["hidden_size"]]
    m_lo, m_up = onnx.load(src_lo), onnx.load(src_up)
    _, tb = find_shared_table(m_lo, dims)
    assert tb is not None, "no [vocab, hidden] table in lower"
    lo_tab = find_shared_table(m_lo, dims)[0]
    retied = False
    if find_shared_table(m_up, dims)[0] is None:
        retied = retie_transposed(m_up, dims, lo_tab.data_type)
        print(f"  {variant}: upper's lm_head was pre-transposed; "
              f"rewritten to share the table: {retied}", flush=True)
    up_tab = find_shared_table(m_up, dims)[0]
    close = None
    if up_tab is not None and up_tab.data_type == lo_tab.data_type:
        a = onnx.numpy_helper.to_array(lo_tab).astype(np.float32)
        c = onnx.numpy_helper.to_array(up_tab).astype(np.float32)
        close = bool(np.allclose(a, c, rtol=1e-3, atol=1e-3))
        assert close, "the two copies of the tied table are not the same weights"
    embed_loc = f"embed{suf}.data"
    with open(os.path.join(outdir, embed_loc), "wb") as f:
        f.write(tb)
    shared = {hashlib.sha1(tb).hexdigest(): embed_loc}
    shapes = ({(tuple(lo_tab.dims), lo_tab.data_type): embed_loc}
              if close else {})

    locs_lo, sh_lo = externalize(m_lo, outdir, f"lower{suf}", shared, shapes)
    locs_up, sh_up = externalize(m_up, outdir, f"upper{suf}", shared, shapes)
    tied = bool(sh_lo) and bool(sh_up)
    onnx.save(m_lo, os.path.join(outdir, f"lower{suf}.onnx"))
    onnx.save(m_up, os.path.join(outdir, f"upper{suf}.onnx"))

    man = {
        "lower": manifest_for(outdir, f"lower{suf}.onnx", locs_lo + sh_lo),
        "upper": manifest_for(outdir, f"upper{suf}.onnx", locs_up + sh_up),
        "shared": [embed_loc],
        "embedding_shared_between_halves": tied,
        "upper_table_retransposed": retied,
        "table_numerically_equal": close,
    }
    embed_bytes = os.path.getsize(os.path.join(outdir, embed_loc))
    man["total_download_mb"] = round(
        (total_mb(man["lower"]) + total_mb(man["upper"])
         - (embed_bytes / 1e6 if tied else 0)), 1)
    man["largest_file_mb"] = round(max(
        [os.path.getsize(os.path.join(outdir, s)) / 1e6
         for half in ("lower", "upper") for e in man[half]["external"]
         for s in e["shards"]]
        + [man["lower"]["model_bytes"] / 1e6,
           man["upper"]["model_bytes"] / 1e6]), 1)
    man["validation"] = validate_variant(outdir, suf, cfg)
    report.setdefault("variants", {})[variant] = man
    print(f"{variant:5} lower {total_mb(man['lower']):7.1f} MB  "
          f"upper {total_mb(man['upper']):7.1f} MB  "
          f"table shared: {tied}  download {man['total_download_mb']:.1f} MB  "
          f"largest file {man['largest_file_mb']:.1f} MB", flush=True)
    return man


def validate_variant(outdir, suf, cfg):
    """Load the emitted files with onnxruntime and compare against fp32.

    A weight variant that will not build a session, or whose hidden16 /
    argmax drifts from the fp32 graphs, is reported here rather than shipped
    on the strength of its file size.
    """
    out = {}
    kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    n_lo, n_up = K + 1, cfg["num_hidden_layers"] - K - 1
    T = 7
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg["vocab_size"], (1, T)).astype(np.int64)
    pos = np.arange(T, dtype=np.int64)[None, :]
    mask = np.tril(np.ones((T, T), np.float32))[None, None]
    ref = {}
    for tag in ("", suf):
        try:
            lo = sess(os.path.join(outdir, f"lower{tag}.onnx"))
            up = sess(os.path.join(outdir, f"upper{tag}.onnx"))
        except Exception as e:                               # noqa: BLE001
            out["session_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            return out
        h, _, _ = lo.run(None, {
            "input_ids": ids, "position_ids": pos, "attn_mask": mask,
            "past_k": np.zeros((n_lo, 1, kv, 0, hd), np.float32),
            "past_v": np.zeros((n_lo, 1, kv, 0, hd), np.float32)})
        lg, _, _ = up.run(None, {
            "hidden": h.astype(np.float32), "position_ids": pos,
            "attn_mask": mask,
            "past_k": np.zeros((n_up, 1, kv, 0, hd), np.float32),
            "past_v": np.zeros((n_up, 1, kv, 0, hd), np.float32)})
        ref[tag] = (h, lg)
        if tag == "":
            out["sessions_build"] = True
    h0, l0 = ref[""]
    h1, l1 = ref[suf]
    out["sessions_build"] = True
    out["hidden16_max_abs_diff_vs_fp32"] = float(np.abs(h1 - h0).max())
    out["logits_max_abs_diff_vs_fp32"] = float(np.abs(l1 - l0).max())
    out["argmax_agrees_with_fp32"] = bool(
        (l1[0].argmax(-1) == l0[0].argmax(-1)).all())
    return out


# ------------------------------------------------------------------ graphs
def half_example_inputs(cfg, half):
    """Deterministic tracing inputs, identical in the worker and the parent."""
    n = K + 1 if half == "lower" else cfg["num_hidden_layers"] - K - 1
    kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    T, P = 5, 3
    torch.manual_seed(0 if half == "lower" else 2)
    first = (torch.randint(0, cfg["vocab_size"], (1, T)) if half == "lower"
             else torch.randn(1, T, cfg["hidden_size"]))
    return (first, torch.arange(P, P + T).view(1, T),
            torch.ones(1, 1, T, P + T),
            torch.randn(n, 1, kv, P, hd) * 0.1,
            torch.randn(n, 1, kv, P, hd) * 0.1)


def half_io(half):
    dyn = {"position_ids": {1: "T"}, "attn_mask": {2: "T", 3: "S"},
           "past_k": {3: "P"}, "past_v": {3: "P"},
           "present_k": {3: "S"}, "present_v": {3: "S"}}
    if half == "lower":
        dyn.update({"input_ids": {1: "T"}})
        ins = ["input_ids", "position_ids", "attn_mask", "past_k", "past_v"]
        outs = ["hidden", "present_k", "present_v"]
    else:
        dyn.update({"hidden": {1: "T"}, "logits": {1: "T"}})
        ins = ["hidden", "position_ids", "attn_mask", "past_k", "past_v"]
        outs = ["logits", "present_k", "present_v"]
    return ins, outs, dyn


def export_one_half(half, rawdir, opset):
    """Export a single half.  Called in a FRESH process: exporting `lower`
    with the dynamo path first leaves state behind that makes the dynamo
    export of `upper` fall back to the legacy exporter, and the legacy one
    pre-transposes the lm_head table to [hidden, vocab], which destroys the
    byte-sharing with the embedding table."""
    hf, _ = mu.load_model("smol")
    lo, up, cfg = LM.split_from_hf(hf, K)
    mod = lo if half == "lower" else up
    ins, outs, dyn = half_io(half)
    p = os.path.join(rawdir, f"{half}.onnx")
    how, errs = try_export(mod, half_example_inputs(cfg, half), p,
                           ins, outs, dyn, opset)
    with open(os.path.join(rawdir, f"{half}.json"), "w") as f:
        json.dump({"exporter": how, "errors": errs}, f)
    print(f"  exported {half} with the {how} exporter", flush=True)


def export_halves(lo, up, cfg, rawdir, opset, report, reuse=False):
    """Run both half-exports in their own processes, then validate here."""
    for half in ("lower", "upper") if not reuse else ():
        cmd = [sys.executable, os.path.abspath(__file__), "--half", half,
               "--outdir", os.path.dirname(rawdir), "--opset", str(opset)]
        subprocess.run(cmd, check=True)
    for half, mod in (("lower", lo), ("upper", up)):
        p = os.path.join(rawdir, f"{half}.onnx")
        with open(os.path.join(rawdir, f"{half}.json")) as f:
            info = json.load(f)
        args = half_example_inputs(cfg, half)
        with torch.no_grad():
            ref = mod(*args)
        names = half_io(half)[0]
        got = sess(p).run(None, {n: a.numpy() for n, a in zip(names, args)})
        d = [maxdiff(g, r.numpy()) for g, r in zip(got, ref)]
        report[half] = {"exporter": info["exporter"], "errors": info["errors"],
                        "max_abs_diff": max(x[0] for x in d),
                        "max_rel_diff": max(x[1] for x in d)}


def export_heads(qh, enc, cfg, outdir, opset, report):
    T = 9
    torch.manual_seed(1)
    hs = torch.randn(1, T, cfg["hidden_size"])
    mask = torch.ones(1, T, dtype=torch.int64)
    p = os.path.join(outdir, "query_head.onnx")
    how, errs = try_export(qh, (hs, mask), p, ["hidden", "mask"],
                           ["op_logits", "slot_logits"],
                           {"hidden": {1: "T"}, "mask": {1: "T"}}, opset)
    with torch.no_grad():
        ref = qh(hs, mask)
    got = sess(p).run(None, {"hidden": hs.numpy(), "mask": mask.numpy()})
    d = [maxdiff(g, r.numpy()) for g, r in zip(got, ref)]
    report["query_head"] = {"exporter": how, "errors": errs,
                            "max_abs_diff": max(x[0] for x in d),
                            "max_rel_diff": max(x[1] for x in d),
                            "bytes": os.path.getsize(p)}

    syms = mu.result_symbols(["12345"], align="left")
    step = torch.tensor([2], dtype=torch.int64)
    p = os.path.join(outdir, "encoder.onnx")
    how, errs = try_export(enc, (syms, step), p, ["symbols", "step"], ["vec"],
                           {}, opset)
    with torch.no_grad():
        ref = enc(syms, step)
    got = sess(p).run(None, {"symbols": syms.numpy(), "step": step.numpy()})
    a, r = maxdiff(got[0], ref.numpy())
    report["encoder"] = {"exporter": how, "errors": errs, "max_abs_diff": a,
                         "max_rel_diff": r, "bytes": os.path.getsize(p)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "model"))
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="skip the torch exports, reuse model/_raw/*.onnx")
    ap.add_argument("--half", choices=["lower", "upper"],
                    help="internal: export just this half and exit")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rawdir = os.path.join(args.outdir, "_raw")
    os.makedirs(rawdir, exist_ok=True)
    if args.half:
        export_one_half(args.half, rawdir, args.opset)
        return

    t0 = time.time()
    hf, _tok = mu.load_model("smol")
    lo, up, cfg = LM.split_from_hf(hf, K)

    qh = mu.AttnQueryHead(cfg["hidden_size"])
    qh.load_state_dict(torch.load(QH.ckpt_file("smol", "attn"))["state_dict"])
    qh.eval()
    enc = mu.ResultEncoder(cfg["hidden_size"], n_steps=mu.N_STREAM_STEPS)
    enc.load_state_dict(torch.load(os.path.join(
        mu.repo_dir(), "ckpt", "smol_stream-left_final.pt"))["state_dict"])
    enc.eval()

    report = {"k": K, "opset": args.opset, "config": cfg,
              "shard_limit_bytes": SHARD_LIMIT}
    export_halves(lo, up, cfg, rawdir, args.opset, report,
                  reuse=args.reuse_raw)
    export_heads(qh, enc, cfg, args.outdir, args.opset, report)
    for name in ("lower", "upper", "query_head", "encoder"):
        r = report[name]
        print(f"{name:11} {r['exporter']:7} max|diff| {r['max_abs_diff']:.3e} "
              f"(rel {r['max_rel_diff']:.2e})", flush=True)

    raw_lo = os.path.join(rawdir, "lower.onnx")
    raw_up = os.path.join(rawdir, "upper.onnx")
    variants = (["fp32"] + ([] if args.no_fp16 else ["fp16"])
                + ([] if args.no_int8 else ["int8"]))
    for v in variants:
        emit_variant(raw_lo, raw_up, args.outdir, cfg, v, report)

    shutil.rmtree(os.path.join(args.outdir, "_tmp"), ignore_errors=True)
    if not args.keep_raw:
        shutil.rmtree(rawdir, ignore_errors=True)

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(mu.MODELS["smol"])
    shutil.copy(hf_hub_download(mu.MODELS["smol"], "tokenizer.json",
                                local_files_only=True),
                os.path.join(args.outdir, "tokenizer.json"))

    meta = {"k": K, "hidden_size": cfg["hidden_size"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "n_lower_layers": K + 1,
            "n_upper_layers": cfg["num_hidden_layers"] - K - 1,
            "vocab_size": cfg["vocab_size"],
            "eos_token_id": tk.eos_token_id,
            "n_result_tokens": mu.N_RESULT_TOKENS,
            "n_stream_steps": mu.N_STREAM_STEPS,
            "tokenizer": "tokenizer.json",
            "query_head": {"model": "query_head.onnx", "external": []},
            "encoder": {"model": "encoder.onnx", "external": []},
            "variants": report["variants"]}
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(HERE, "export_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"DONE export in {time.time() - t0:.0f}s -> {args.outdir}")


if __name__ == "__main__":
    main()
