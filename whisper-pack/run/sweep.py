"""Sweep quantization schemes over eval sets; greedy Whisper decoding, jiwer scoring.

Resumable: (model, scheme, set) pairs already in --out are skipped.
Japanese is scored as character error rate (chars space-separated after normalizing).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import jiwer
import transformers
transformers.logging.set_verbosity_error()
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quant import apply_scheme  # noqa: E402


def norm_for(tok, set_name, lang):
    if set_name.startswith("ls-"):
        f = tok.normalize
    else:
        f = tok.basic_normalize
    if lang == "ja":
        return lambda s: " ".join(f(s).replace(" ", ""))
    return lambda s: " ".join(f(s).split())


def score(refs, hyps):
    errs, nref = [], []
    for r, h in zip(refs, hyps):
        rw = r.split()
        if not rw:
            errs.append(len(h.split())); nref.append(0); continue
        if not h.split():
            errs.append(len(rw)); nref.append(len(rw)); continue
        o = jiwer.process_words(r, h)
        errs.append(o.substitutions + o.deletions + o.insertions); nref.append(len(rw))
    return errs, nref


def transcribe(model, proc, audio, lang, batch, device):
    hyps = []
    for i in range(0, len(audio), batch):
        chunk = [np.asarray(a, dtype=np.float32) for a in audio[i:i + batch]]
        feats = proc.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            # English-only checkpoints (*.en) take no language or task token
            lt = {} if not model.config.to_dict().get("is_multilingual", model.config.vocab_size >= 51865) else {"language": lang, "task": "transcribe"}
            out = model.generate(feats, num_beams=1, do_sample=False, return_timestamps=False, max_new_tokens=224, **lt)
        hyps += proc.tokenizer.batch_decode(out, skip_special_tokens=True)
    return [h.strip() for h in hyps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--schemes", default="f32")
    ap.add_argument("--sets", default="ls-clean,ls-other")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=str(HERE.parent / "cache"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    cache = Path(a.cache_dir)
    out = Path(a.out) if a.out else HERE.parent / "results" / f"sweep-{short}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["scheme"], r["set"]))
    sets = a.sets.split(",")
    proc = WhisperProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    for scheme in a.schemes.split(";"):
        todo = [s for s in sets if (a.model, scheme, s) not in done]
        if not todo:
            continue
        model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32)
        info = apply_scheme(model, scheme)
        model.to(a.device).eval()
        for s in todo:
            d = np.load(cache / f"{s}.npz", allow_pickle=True)
            audio, text, ids, lang = list(d["audio"][:a.n]), list(d["text"][:a.n]), list(d["id"][:a.n]), str(d["language"])
            t0 = time.time()
            hyps = transcribe(model, proc, audio, lang, a.batch, a.device)
            secs = time.time() - t0
            nf = norm_for(tok, s, lang)
            refs_n, hyps_n = [nf(t) for t in text], [nf(h) for h in hyps]
            errs, nref = score(refs_n, hyps_n)
            wer = sum(errs) / max(1, sum(nref))
            rec = {"model": a.model, "scheme": scheme, "set": s, "n": len(ids), "language": lang,
                   "metric": "cer" if lang == "ja" else "wer",
                   "wer": wer, "errors": sum(errs), "ref_words": sum(nref), "seconds": secs,
                   "device": a.device, "quant": info, "ids": ids, "utt_errors": errs, "utt_ref_words": nref}
            (cache / f"hyp-{short}-{scheme}-{s}.json").write_text(json.dumps(
                [{"id": i, "ref": t, "hyp": h} for i, t, h in zip(ids, text, hyps)], ensure_ascii=False, indent=1))
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{short} {scheme} {s} n={len(ids)} {rec['metric']}={wer*100:.2f}% "
                  f"{secs:.0f}s ({secs/len(ids):.2f}s/utt)", flush=True)
        del model


if __name__ == "__main__":
    main()
