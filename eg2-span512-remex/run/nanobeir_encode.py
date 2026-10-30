"""Encode the 13 English NanoBEIR sets with EG2 and with tomaarsen/embeddinggemma-2-512d (Colab GPU job).

Usage (from this directory's parent): colab new -s nanobeir --gpu L4
    colab exec -s nanobeir -f run/nanobeir_encode.py
    colab download -s nanobeir /content/nanobeir_emb.tar out/   ; colab stop -s nanobeir

Same setup as the model card of tomaarsen/embeddinggemma-2-512d: sentence-transformers' NanoBEIR-en
(the 13 English sets), queries through encode_query and documents through encode_document (the
models' "query" / "document" prompts), bf16 on a GPU, unit-norm outputs. Writes per set
<set>_<model>.npz with q (queries x d) and d (corpus x d) as float32, rows in the parquet order,
then tars them to /content/nanobeir_emb.tar. Scoring and remex quantization run on CPU in
run/nanobeir_eval.py.
"""
import gc
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=6.1", "transformers"], check=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SETS = ["NanoArguAna", "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018", "NanoHotpotQA",
        "NanoMSMARCO", "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoSciFact", "NanoTouche2020"]
MODELS = {"eg2": "google/embeddinggemma-2", "svd512": "tomaarsen/embeddinggemma-2-512d"}
OUT = Path("/content/nanobeir_emb")
OUT.mkdir(exist_ok=True)


def part(kind, name):
    return pd.read_parquet(hf_hub_download("sentence-transformers/NanoBEIR-en", f"{kind}/{name}-00000-of-00001.parquet",
                                           repo_type="dataset"))


print(torch.cuda.get_device_name(0), flush=True)
for key, repo in MODELS.items():
    model = SentenceTransformer(repo, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    for name in SETS:
        path = OUT / f"{name}_{key}.npz"
        if path.exists():
            continue
        t0 = time.time()
        corpus, queries = part("corpus", name), part("queries", name)
        q = model.encode_query(queries.text.tolist(), batch_size=64, convert_to_numpy=True)
        d = None
        for bs in ((4, 1) if name == "NanoTouche2020" else (32, 8, 2)):   # NanoTouche2020's long arguments need a small batch (padding is masked, so the vectors match)
            try:
                d = model.encode_document(corpus.text.tolist(), batch_size=bs, convert_to_numpy=True)
                break
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"{key} {name}: out of memory at batch {bs}", flush=True)
        if d is None or len(d) != len(corpus):
            raise RuntimeError(f"{key} {name}: no corpus embeddings")
        np.savez(path, q=q.astype(np.float32), d=d.astype(np.float32))
        print(f"{key} {name}: {len(queries)} queries, {len(corpus)} docs, dim {d.shape[1]}, {time.time() - t0:.0f} s", flush=True)
    del model
    torch.cuda.empty_cache()

with tarfile.open("/content/nanobeir_emb.tar", "w") as tf:
    tf.add(OUT, arcname="nanobeir_emb")
print("done", flush=True)
