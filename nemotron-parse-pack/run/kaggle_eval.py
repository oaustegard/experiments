"""Nemotron Parse 2.0 on Omni Parse Bench pages, fp32 and with weights fake-quantized exactly as the
browser pack stores them. Runs as a Kaggle GPU script (T4 x2, internet on): one worker process per GPU,
pages split between them, one JSON per page and arm under /kaggle/working/<arm>/.

Pages are rendered as the benchmark renders them for every image-only model
(omni_parse_bench.harness.providers.llm.image: long edge 2048 px), so the container's CPU engines can use
the same images.

The job is configured by the constants below; make_kaggle() in this file is not used on Kaggle.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PAGES = "__PAGES__"          # newline-separated page ids, filled in at push time
ARMS = "__ARMS__"            # JSON list of [name, scheme-or-null]
MAX_TOKENS = int("__MAX_TOKENS__")
BATCH = int("__BATCH__")
TEACHER = "__TEACHER__"      # arm whose tokens the others are scored against teacher-forced ("" = off)
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
REPO = "nvidia/NVIDIA-Nemotron-Parse-2.0"
DS = "datalab-to/omni_parse_bench"
OUT = Path("/kaggle/working")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup():
    sh("pip install -q transformers==5.6.1 timm==1.0.22 open_clip_torch==3.2.0 einops==0.8.1 "
       "accelerate==1.12.0 beautifulsoup4 'omni-parse-bench[harness]==1.8.0' rapidfuzz")
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(REPO, local_dir="/kaggle/tmp/model")
    pages = PAGES.split()
    import pyarrow.parquet as pq
    man = {r["page_id"]: r for r in pq.read_table(hf_hub_download(DS, "manifest.parquet", repo_type="dataset")).to_pylist()}
    from omni_parse_bench.harness.providers import llm
    img_dir = Path("/kaggle/tmp/png"); img_dir.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(pages):
        p = Path(hf_hub_download(DS, man[pid]["doc_path"], repo_type="dataset"))
        (img_dir / f"{pid}.png").write_bytes(llm.image(p))
    print(f"rendered {len(pages)} pages", flush=True)


# ---------- quantization: the pack's exact arithmetic ----------

def fq(w, kind, group=32):
    import torch
    shape = w.shape
    rows = w.squeeze(0).shape[0] if w.ndim > 2 and shape[0] == 1 else shape[0]   # pos_embed [1,N,C] -> N rows
    w2 = w.reshape(rows, -1).float()
    if kind == "f16":
        return w2.half().float().reshape(shape)
    if kind == "int8":
        s = (w2.abs().amax(1, keepdim=True) / 127).half().float()
        s[s == 0] = 1
        return (torch.clamp(torch.round(w2 / s), -127, 127) * s).reshape(shape)
    if kind == "int4g":
        o, i = w2.shape
        pad = (-i) % group
        wp = torch.nn.functional.pad(w2, (0, pad)).reshape(o, -1, group)
        s = (wp.abs().amax(2, keepdim=True) / 7).half().float()
        s[s == 0] = 1
        q = torch.clamp(torch.round(wp / s), -7, 7) * s
        return q.reshape(o, -1)[:, :i].reshape(shape)
    raise ValueError(kind)


def apply_scheme(model, master, scheme):
    """scheme: list of [regex, kind, group?]; first match wins; unmatched 2-D+ weights stay fp32.
    Norms and biases (1-D) are never touched by the regexes used here."""
    import torch
    rules = [(re.compile(r[0]), r[1], (r[2] if len(r) > 2 else 32)) for r in (scheme or [])]
    sd = model.state_dict()
    n_q = 0
    with torch.no_grad():
        for k, v in master.items():
            if k not in sd:        # RADIO's input conditioner leaves the state dict after its first forward
                continue
            t = v
            if v.ndim >= 2:
                for rx, kind, g in rules:
                    if rx.search(k):
                        t = fq(v, kind, g); n_q += 1
                        break
            sd[k].copy_(t.to(sd[k].dtype))
    return n_q


# ---------- one GPU worker ----------

def worker(gpu: int, ngpu: int):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, GenerationConfig
    from transformers.modeling_outputs import BaseModelOutput
    sys.path.insert(0, "/kaggle/tmp/model")
    dev = "cuda:0"
    tag = f"[gpu{gpu}]"
    model = AutoModel.from_pretrained("/kaggle/tmp/model", trust_remote_code=True, torch_dtype=torch.float32)
    # __init__ casts to the config's bfloat16, so from_pretrained leaves bf16-rounded values even in float32;
    # reload the float32 file over it so the reference and the quantized arms start from the shipped weights.
    from safetensors.torch import load_file
    missing, unexpected = model.load_state_dict(load_file("/kaggle/tmp/model/model.safetensors"), strict=False)
    print(f"{tag} fp32 reload: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    model = model.float().to(dev).eval()
    for m in model.modules():          # timm attention: use SDPA so 13k tokens never build a full score matrix
        if hasattr(m, "fused_attn"):
            m.fused_attn = True
    # On a T4 in fp32 SDPA falls back to the math kernel, which builds the 13k x 13k scores (10.6 GB) at once.
    # Query chunks of 2048 keep it near 1.7 GB; the result is the same attention.
    import torch.nn.functional as F
    _sdpa = F.scaled_dot_product_attention
    def _chunked(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if q.shape[-2] <= 4096 or attn_mask is not None or is_causal:
            return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
        return torch.cat([_sdpa(q[..., i:i + 2048, :], k, v, dropout_p=dropout_p, **kw)
                          for i in range(0, q.shape[-2], 2048)], dim=-2)
    F.scaled_dot_product_attention = _chunked
    proc = AutoProcessor.from_pretrained("/kaggle/tmp/model", trust_remote_code=True)
    arms = json.loads(ARMS)
    master = ({k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
              if any(sc for _, sc in arms) else None)   # a CPU copy only when some arm quantizes
    gen = GenerationConfig.from_pretrained("/kaggle/tmp/model")
    gen.max_new_tokens = MAX_TOKENS
    pages = PAGES.split()[gpu::ngpu]
    prompt_ids = proc.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    teacher_dir = OUT / TEACHER if TEACHER else None
    for name, scheme in arms:
        nq = apply_scheme(model, master, scheme) if scheme else 0
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        print(f"{tag} arm {name}: {nq} tensors quantized", flush=True)
        for b0 in range(0, len(pages), BATCH):
            batch = [p for p in pages[b0:b0 + BATCH] if not (d / f"{p}.json").exists()]
            if not batch:
                continue
            t0 = time.time()
            encs, sizes = [], []
            with torch.no_grad():
                for pid in batch:
                    im = Image.open(f"/kaggle/tmp/png/{pid}.png").convert("RGB")
                    sizes.append(im.size)
                    pv = proc(images=[im], return_tensors="pt")["pixel_values"].to(dev)
                    encs.append(model.encoder(pv).last_hidden_state)
                torch.cuda.synchronize(); t_enc = time.time() - t0
                enc = BaseModelOutput(last_hidden_state=torch.cat(encs))
                ids = prompt_ids.repeat(len(batch), 1)
                out = model.generate(encoder_outputs=enc, decoder_input_ids=ids, generation_config=gen)
            t_all = time.time() - t0
            for j, pid in enumerate(batch):
                toks = out[j, ids.shape[1]:].tolist()
                if 2 in toks:
                    toks = toks[:toks.index(2)]
                rec = {"page_id": pid, "arm": name, "tokens": toks,
                       "raw": proc.tokenizer.decode(toks, skip_special_tokens=False),
                       "width": sizes[j][0], "height": sizes[j][1], "hit_limit": len(toks) >= MAX_TOKENS,
                       "batch_s": t_all, "enc_s": t_enc / len(batch), "batch_n": len(batch)}
                # teacher-forced agreement with the teacher arm's tokens (next-token top-1)
                tf = teacher_dir / f"{pid}.json" if teacher_dir else None
                if tf and name != TEACHER and tf.exists():
                    tt = json.loads(tf.read_text())["tokens"]
                    if tt:
                        with torch.no_grad():
                            seq = torch.tensor([prompt_ids[0].tolist() + tt + [2]], device=dev)
                            lo = model(encoder_outputs=BaseModelOutput(last_hidden_state=encs[j]),
                                       decoder_input_ids=seq[:, :-1]).logits[0, prompt_ids.shape[1] - 1:]
                            tgt = seq[0, prompt_ids.shape[1]:]
                            rec["tf_top1"] = float((lo.argmax(-1) == tgt).float().mean())
                            rec["tf_n"] = int(tgt.numel())
                (d / f"{pid}.json").write_text(json.dumps(rec))
            print(f"{tag} {name} {b0 + len(batch)}/{len(pages)} pages, batch {t_all:.0f}s "
                  f"(enc {t_enc:.1f}s), max tokens {max(len(json.loads((d / f'{p}.json').read_text())['tokens']) for p in batch)}",
                  flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        return worker(int(sys.argv[2]), int(sys.argv[3]))
    setup()
    import torch
    n = max(1, torch.cuda.device_count())
    print("gpus", n, [torch.cuda.get_device_name(i) for i in range(n)], flush=True)
    # The teacher arm runs first everywhere, so each worker's teacher files exist before its other arms.
    procs = []
    for g in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", str(g), str(n)], env=env))
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)
    summary = {}
    for name, _ in json.loads(ARMS):
        recs = [json.loads(f.read_text()) for f in (OUT / name).glob("*.json")]
        tf = [r["tf_top1"] for r in recs if "tf_top1" in r]
        summary[name] = {"pages": len(recs), "tokens": sum(len(r["tokens"]) for r in recs),
                         "hit_limit": sum(r["hit_limit"] for r in recs),
                         "tf_top1_mean": sum(tf) / len(tf) if tf else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
