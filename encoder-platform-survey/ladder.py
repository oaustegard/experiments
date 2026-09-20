"""CPU latency ladder for int8 encoder ONNX artifacts. batch=1, median of N runs."""
import time, json, os, sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("ettin-17m int8",        "e17/enc_model_quantized.onnx",          "ettin-tokenizer.json"),
    ("ettin-32m int8",        "e32/enc_model_quantized.onnx",          "ettin-tokenizer.json"),
    ("mmBERT-small int8",     "enc_mmbert-small-int8.onnx",    "mmbert-tokenizer.json"),
    ("ModernBERT-base int8",  "enc_modernbert-base-int8.onnx", "modernbert-tokenizer.json"),
    ("NeoBERT int8",          "neobert-int8.onnx",         "neobert-tokenizer.json"),
    ("ModernBERT-large int8", "enc_modernbert-large-int8.onnx","modernbert-tokenizer.json"),
]
TEXT = ("We propose a non-autoregressive encoder for typed decisions. " * 200)
N = 15
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

rows = []
for name, mf, tf in MODELS:
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(os.path.join(HERE, mf), so, providers=["CPUExecutionProvider"])
    except Exception as e:
        rows.append({"model": name, "error": str(e)[:200]}); print(name, "LOAD FAIL", str(e)[:200]); continue
    tok = Tokenizer.from_file(os.path.join(HERE, tf))
    names = [i.name for i in sess.get_inputs()]
    ids_full = tok.encode(TEXT).ids
    row = {"model": name, "size_mb": round(sum(os.path.getsize(os.path.join(HERE, f)) for f in os.listdir(HERE) if f.startswith(os.path.basename(mf).replace(".onnx","")))/1e6, 1), "inputs": names}
    for L in (128, 512):
        ids = np.array([ids_full[:L]], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in names: feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in names}
        try:
            for _ in range(3): sess.run(None, feeds)
            ts = []
            for _ in range(N):
                t = time.perf_counter(); out = sess.run(None, feeds); ts.append(time.perf_counter() - t)
            row[f"ms_{L}"] = round(1000 * float(np.median(ts)), 1)
            row[f"out_{L}"] = list(out[0].shape)
        except Exception as e:
            row[f"ms_{L}"] = None; row[f"err_{L}"] = str(e)[:160]
    rows.append(row); print(json.dumps(row))
json.dump({"threads": THREADS, "cpu": os.cpu_count(), "n": N, "rows": rows}, open(os.path.join(HERE, f"ladder_t{THREADS}.json"), "w"), indent=1)
