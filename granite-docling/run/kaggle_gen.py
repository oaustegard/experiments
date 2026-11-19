"""granite-docling-258M on OmniParseBench pages, as a Kaggle GPU job (2x T4).

Pushed by run/launch.py, which fills in the CONFIG block below. The job:

  1. installs its runtime (vLLM by default) and the page renderer, with internet on;
  2. downloads the model and each page it is given from the Hugging Face Hub;
  3. renders each page to grayscale exactly as indra-browser/run/prerender.py does
     (pdfium at 200 DPI, lowered past 100 MP; an image page read as IMREAD_GRAYSCALE),
     so docling reads the pixels Indra read; it records a hash of each page's pixels so
     the local copy can be checked against it;
  4. generates DocTags for every page with the prompt "Convert this page to docling."
     (greedy, up to MAX_NEW tokens), one engine process per GPU, half the pages each;
  5. writes one JSON per page to /kaggle/working/pages/<page_id>.json.

A deadline (DEADLINE_S) stops each worker cleanly well inside Kaggle's 12 h session limit;
the pages it reached are in the output and the launcher sends the rest in a new job.
"""
# ---- CONFIG (launch.py rewrites these lines) ----
PAGES = []            # page ids to run
ENGINES = ["vllm"]    # "vllm" and/or "hf"
DTYPE = "float32"     # T4 has no bf16: float32 (exact upcast of the bf16 weights) or float16
MAX_NEW = 8192
DEADLINE_S = 10.5 * 3600
BATCH = 16            # hf engine only
MODEL = "ibm-granite/granite-docling-258M"
REVISION = "untied"   # vLLM's Idefics3 wants the untied revision (the model card's vLLM example); same weights
PROMPT = "Convert this page to docling."
MAX_LEN = 8192        # vLLM max_model_len (prompt + generation)
CHUNK = 32            # pages per llm.generate call. vLLM preprocesses a whole call before generating:
                      # at 128 docling pages (up to 17 tiles of 512 px each, fp32) a worker's engine was
                      # killed for host memory on 2026-11-19. The KV cache holds ~28 pages at once anyway.
TAG = "docling"
PACK_ARMS = []        # e.g. ["pack", "pack-head8"]: run fake-quantized copies (run/pack_quant.py) instead
DS_REV = "d70182b9f901d1660a27e60bf6be6d6e6ea83013"
# ---- end CONFIG ----

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
W = Path("/kaggle/working")
CACHE = Path("/tmp/gd")
SELF = CACHE / "gen.py"


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode


def setup():
    pkgs = "pypdfium2==5.13.0 opencv-python-headless==5.1.0.95"
    if "vllm" in ENGINES:
        pkgs += " vllm"
    rc = sh(f"{sys.executable} -m pip install -q {pkgs}")
    info = {"pip_rc": rc}
    for m in ("torch", "transformers", "vllm", "pypdfium2", "cv2"):
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[m] = f"missing: {e}"
    info["nvidia"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                                    shell=True, capture_output=True, text=True).stdout.strip()
    return info


def fetch_pages():
    from huggingface_hub import hf_hub_download, snapshot_download
    import pyarrow.parquet as pq

    ds = Path(snapshot_download("datalab-to/omni_parse_bench", repo_type="dataset", revision=DS_REV,
                                allow_patterns=["manifest.parquet"], cache_dir=CACHE))
    rows = {r["page_id"]: r for r in pq.read_table(ds / "manifest.parquet").to_pylist()}
    gray_dir = CACHE / "gray"
    gray_dir.mkdir(parents=True, exist_ok=True)
    root = Path(snapshot_download("datalab-to/omni_parse_bench", repo_type="dataset", revision=DS_REV,
                                  allow_patterns=[rows[p]["doc_path"] for p in PAGES], cache_dir=CACHE,
                                  max_workers=16))
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(root / rows[p]["doc_path"], gray_dir / f"{p}.png") for p in PAGES]
    import multiprocessing as mp

    with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex:
        done = list(ex.map(_render, jobs, chunksize=8))
    return dict(zip(PAGES, done))


def _render(job):
    return render(*job)


def render(src, dst):
    """indra-browser/run/prerender.py's render, returning the pixel hash."""
    import cv2
    import numpy as np

    with open(src, "rb") as f:
        head = f.read(1024)
    if b"%PDF-" in head:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(src))
        try:
            page = doc[0]
            w, h = page.get_size()
            k = min(200.0 / 72.0, (100e6 / max(w * h, 1.0)) ** 0.5)
            arr = page.render(scale=k, grayscale=True).to_numpy()
            gray = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
        finally:
            doc.close()
    else:
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(dst), gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return {"png": str(dst), "w": int(gray.shape[1]), "h": int(gray.shape[0]),
            "md5": hashlib.md5(gray.tobytes()).hexdigest()}


def pack_copy(model_dir, arm, info):
    """A copy of the model folder whose weights are fake-quantized by run/pack_quant.py (PACK_SRC)."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    ns = {}
    exec(base64.b64decode(PACK_SRC).decode(), ns)
    dst = CACHE / f"model-{arm}"
    if dst.exists():
        return str(dst)
    shutil.copytree(model_dir, dst, ignore=shutil.ignore_patterns("*.safetensors"))
    files = sorted(Path(model_dir).glob("*.safetensors"))
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    out = ns["fake"](state, arm)
    info[f"{arm}_bytes"] = ns["sizes"](state)
    diffs = {n: float((out[n] - state[n].float()).abs().max()) for n in state if out[n].ndim == 2}
    info[f"{arm}_max_abs_change"] = max(diffs.values())
    save_file({n: t.contiguous().to(torch.float32) for n, t in out.items()}, str(dst / "model.safetensors"),
              metadata={"format": "pt"})
    idx = dst / "model.safetensors.index.json"
    if idx.exists():
        idx.unlink()
    print(f"{arm}: packed copy written, {len(state)} tensors", flush=True)
    return str(dst)


def worker(engine, gpu, pids, pages, outdir, model_dir):
    """Runs in a child process with CUDA_VISIBLE_DEVICES=gpu."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_dir)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    todo = [p for p in pids if not (outdir / f"{p}.json").exists()]

    def image(pid):
        return Image.open(pages[pid]["png"]).convert("RGB")

    def save(pid, text, ntok, nprompt, finish, sec):
        rec = {"page_id": pid, "doctags": text, "gen_tokens": ntok, "prompt_tokens": nprompt,
               "finish": finish, "batch_seconds": sec, "engine": engine, "dtype": DTYPE, "gpu": gpu,
               "w": pages[pid]["w"], "h": pages[pid]["h"], "md5": pages[pid]["md5"]}
        tmp = outdir / f"{pid}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.rename(outdir / f"{pid}.json")

    if engine == "vllm":
        from vllm import LLM, SamplingParams

        kw = dict(model=model_dir, dtype=DTYPE, limit_mm_per_prompt={"image": 1}, max_model_len=MAX_LEN,
                  gpu_memory_utilization=0.88, seed=0)
        try:
            llm = LLM(**kw, mm_processor_cache_gb=0)
        except TypeError:  # an engine version without that argument
            llm = LLM(**kw)
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, skip_special_tokens=False)
        print(f"gpu{gpu} engine ready {time.time() - T0:.0f}s after start", flush=True)
        # Chunks bound the work lost to the deadline; vLLM batches continuously inside a chunk.
        chunk = CHUNK
        for i in range(0, len(todo), chunk):
            if time.time() - T0 > DEADLINE_S:
                print("deadline", flush=True)
                break
            part = todo[i:i + chunk]
            t = time.time()
            outs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image(p)}} for p in part], sp)
            sec = time.time() - t
            for p, o in zip(part, outs):
                c = o.outputs[0]
                save(p, c.text, len(c.token_ids), len(o.prompt_token_ids), c.finish_reason, sec / len(part))
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            longs = sum(o.outputs[0].finish_reason == "length" for o in outs)
            print(f"gpu{gpu} {i + len(part)}/{len(todo)} chunk {sec:.0f}s {ntok} tokens "
                  f"{ntok / max(sec, 1e-9):.0f} tok/s, {longs} hit the length cap", flush=True)
    else:
        import torch
        from transformers import AutoModelForImageTextToText

        dt = {"float32": torch.float32, "float16": torch.float16}[DTYPE]
        model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=dt).to("cuda").eval()
        proc.tokenizer.padding_side = "left"
        for i in range(0, len(todo), BATCH):
            if time.time() - T0 > DEADLINE_S:
                break
            part = todo[i:i + BATCH]
            t = time.time()
            inp = proc(text=[prompt] * len(part), images=[[image(p)] for p in part], return_tensors="pt",
                       padding=True).to("cuda")
            with torch.inference_mode():
                # use_cache must be explicit: the model's generation_config.json sets it false, which makes
                # every step recompute the whole sequence (the 2026-11-19 probe ran for an hour on 24 pages).
                ids = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
            sec = time.time() - t
            n = inp["input_ids"].shape[1]
            for j, p in enumerate(part):
                gen = ids[j, n:]
                eos = (gen == proc.tokenizer.eos_token_id).nonzero()
                k = int(eos[0]) + 1 if len(eos) else len(gen)
                text = proc.batch_decode(gen[None, :k], skip_special_tokens=False)[0]
                nprompt = int(inp["attention_mask"][j].sum())
                save(p, text, k, nprompt, "stop" if len(eos) else "length", sec / len(part))
            print(f"gpu{gpu} hf {i + len(part)}/{len(todo)} batch {sec:.0f}s", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        args = json.loads(sys.argv[2])
        pages = json.loads(Path(args["pages"]).read_text())
        worker(args["engine"], args["gpu"], args["pids"], pages, Path(args["out"]), args["model"])
        return
    info = setup()
    print(info, flush=True)
    # Kaggle may run this through IPython, where __file__ is unset: the launcher appends this file's own
    # source as SELF_SRC, and the workers run from that copy.
    CACHE.mkdir(parents=True, exist_ok=True)
    SELF.write_text(base64.b64decode(SELF_SRC).decode())
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE)
    runs = [(TAG, model_dir)]
    if PACK_ARMS:
        runs = [(arm, pack_copy(model_dir, arm, info)) for arm in PACK_ARMS]
    t = time.time()
    pages = fetch_pages()
    info["fetch_s"] = round(time.time() - t, 1)
    (CACHE / "pages.json").write_text(json.dumps(pages))
    W.mkdir(exist_ok=True)
    (W / "render.json").write_text(json.dumps({p: v["md5"] for p, v in pages.items()}))
    for (tag, model_dir), engine in ((r, e) for r in runs for e in ENGINES):
        outdir = W / f"pages-{tag}-{engine}-{DTYPE}"
        outdir.mkdir(exist_ok=True)
        ids = list(pages)
        t = time.time()
        procs = []
        for gpu in (0, 1):
            a = {"engine": engine, "gpu": gpu, "pids": ids[gpu::2], "pages": str(CACHE / "pages.json"),
                 "out": str(outdir), "model": model_dir}
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            # Workers print to this job's stdout, so the live log stream shows their progress.
            procs.append(subprocess.Popen([sys.executable, str(SELF), "--worker", json.dumps(a)], env=env))
        # Hard stop: a worker still running half an hour past the deadline is killed, so a hang cannot
        # hold the GPUs (and the weekly quota) until Kaggle's 12 h limit. Saved pages stay saved.
        # A worker that dies (an engine killed for memory) is restarted once on its remaining pages.
        retries = {0: 1, 1: 1}
        rcs = [None, None]
        while any(r is None for r in rcs):
            time.sleep(20)
            over = time.time() - T0 > DEADLINE_S + 1800
            for gpu, p in enumerate(procs):
                if rcs[gpu] is not None:
                    continue
                rc = p.poll()
                if rc is None:
                    if over:
                        p.kill()
                        rcs[gpu] = "killed"
                    continue
                if rc != 0 and retries[gpu] and time.time() - T0 < DEADLINE_S:
                    retries[gpu] -= 1
                    print(f"gpu{gpu} worker exited {rc}; restarting it on its remaining pages", flush=True)
                    a = {"engine": engine, "gpu": gpu, "pids": ids[gpu::2], "pages": str(CACHE / "pages.json"),
                         "out": str(outdir), "model": model_dir}
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    procs[gpu] = subprocess.Popen([sys.executable, str(SELF), "--worker", json.dumps(a)], env=env)
                    continue
                rcs[gpu] = rc
        info[f"{tag}_{engine}_rc"] = rcs
        info[f"{tag}_{engine}_wall_s"] = round(time.time() - t, 1)
        info[f"{tag}_{engine}_done"] = len(list(outdir.glob("*.json")))
        print(info, flush=True)
    info["total_s"] = round(time.time() - T0, 1)
    (W / "info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
