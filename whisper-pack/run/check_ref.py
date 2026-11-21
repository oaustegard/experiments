"""Check ref.py (NumPy, from the pack) against the HF model rewritten by quant.apply_scheme.

    python3 run/check_ref.py --model openai/whisper-base --scheme "all=r15,conv=f16" --n 20

Packs into ../out/packs/<slug> (unless --pack-dir), then per utterance compares generated token ids,
encoder states, and logits (first step, and every step teacher-forced on the generated ids).
Exits 1 on any token mismatch.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack as packmod  # noqa: E402
import quant  # noqa: E402
import ref  # noqa: E402


def run(model_id, scheme, n, data, pack_dir=None, quiet=False):
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    slug = re.sub(r"[^A-Za-z0-9.=,_-]+", "_", model_id.split("/")[-1] + "-" + scheme)
    pack_dir = Path(pack_dir) if pack_dir else HERE.parent / "out" / "packs" / slug
    sizes = packmod.build(model_id, scheme, pack_dir)
    p = ref.load(pack_dir)
    g = p.gen

    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
    quant.apply_scheme(model, scheme)
    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    z = np.load(data, allow_pickle=True)
    audio = z["audio"][:n]

    if g["multilingual"]:
        prompt = [g["sot"], g["lang_to_id"]["<|en|>"], g["task_to_id"]["transcribe"], g["no_timestamps"]]
        kw = dict(language="en", task="transcribe")
    else:
        prompt = [g["sot"], g["no_timestamps"]]
        kw = {}
    ok, enc_d, first_d, all_d = 0, 0.0, 0.0, 0.0
    for i, a in enumerate(audio):
        feats = fe(a, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            hf_states = model.model.encoder(feats).last_hidden_state[0].numpy()
            seq = model.generate(feats, num_beams=1, do_sample=False, return_timestamps=False,
                                 max_new_tokens=224, **kw)[0].tolist()
        if seq[: len(prompt)] == prompt:
            seq = seq[len(prompt):]
        # generate() strips the forced prompt; compare the generated text tokens, end-of-text excluded
        hf_ids = seq[: seq.index(g["eot"])] if g["eot"] in seq else seq
        states = p.encode(feats[0].numpy())
        ids, logits = p.decode(states, prompt, 224)
        stopped = bool(ids) and ids[-1] == g["eot"]
        same = (ids[:-1] if stopped else ids) == hf_ids  # HF drops the end-of-text token from its output
        ok += same
        enc_d = max(enc_d, float(np.abs(states - hf_states).max()))
        with torch.no_grad():
            dec_in = torch.tensor([prompt + ids[:-1]])
            hl = model(input_features=feats, decoder_input_ids=dec_in).logits[0, len(prompt) - 1:].numpy()
        rl = np.stack(logits)
        first_d = max(first_d, float(np.abs(rl[0] - hl[0]).max()))
        all_d = max(all_d, float(np.abs(rl - hl[: len(rl)]).max()))
        if not quiet:
            print(f"[{i}] {'ok ' if same else 'DIFF'} {len(ids)} tokens", flush=True)
        if not same:
            k = next((j for j, (x, y) in enumerate(zip(ids, hf_ids)) if x != y), min(len(ids), len(hf_ids)))
            print(f"   first difference at step {k}: ref {ids[k:k + 3]} hf {hf_ids[k:k + 3]} (ref stopped: {stopped})")
    res = dict(model=model_id, scheme=scheme, n=len(audio), ok=int(ok), enc_diff=enc_d, first_logit_diff=first_d,
               all_logit_diff=all_d, **sizes)
    print(f"{model_id} {scheme}: tokens identical {ok}/{len(audio)}; max encoder diff {enc_d:.3g}; "
          f"max first-step logit diff {first_d:.3g}; max logit diff all steps {all_d:.3g}; "
          f"weights.bin {sizes['weights_bytes']} B, gzip-6 {sizes['gzip_bytes']} B "
          f"(whole pack {sizes['total_bytes']} / {sizes['total_gzip_bytes']})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--data", default=str(HERE.parent / "cache" / "ls-clean.npz"))
    ap.add_argument("--pack-dir")
    a = ap.parse_args()
    r = run(a.model, a.scheme, a.n, a.data, a.pack_dir)
    sys.exit(0 if r["ok"] == r["n"] else 1)


if __name__ == "__main__":
    main()
