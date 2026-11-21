"""Fixtures for the timestamp-mode check: HF generate with timestamps and a 3-gram ban on the
quantized tiny model, for three short clips and one 50 s clip.

Writes test/fixtures/ts/{index.json, clip<i>.f32}.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "run"))
import quant

MODEL, SCHEME, PACK = "openai/whisper-tiny", "all=r63,conv=f16", "out/packs/whisper-tiny-0/"
out = HERE / "fixtures" / "ts"
out.mkdir(parents=True, exist_ok=True)
z = np.load(HERE.parent / "cache" / "ls-clean.npz", allow_pickle=True)
audio, text = z["audio"], z["text"]
clips = [audio[0], audio[1], audio[2], np.concatenate([audio[i] for i in range(3, 9)])[: 50 * 16000]]
refs = [text[0], text[1], text[2], " ".join(text[3:9])]
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
quant.apply_scheme(model, SCHEME)
proc = WhisperProcessor.from_pretrained(MODEL)
cases = []
for i, (pcm, ref) in enumerate(zip(clips, refs)):
    pcm = pcm.astype(np.float32)
    pcm.tofile(out / f"clip{i}.f32")
    case = {"i": i, "seconds": len(pcm) / 16000, "ref": proc.tokenizer.normalize(str(ref))}
    if len(pcm) <= 30 * 16000:
        feats = proc.feature_extractor(pcm, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe", num_beams=1, do_sample=False,
                                 return_timestamps=True, no_repeat_ngram_size=3, max_new_tokens=224)[0].tolist()
        g = model.generation_config
        ids = [t for t in ids if t not in (g.decoder_start_token_id, g.lang_to_id["<|en|>"], g.task_to_id["transcribe"], g.eos_token_id)]
        case["ids"] = ids
        case["text"] = proc.tokenizer.decode(ids, skip_special_tokens=True)
        case["with_ts"] = proc.tokenizer.decode(ids, decode_with_timestamps=True)
    cases.append(case)
    print(i, case.get("with_ts", "(long)")[:160], flush=True)
(out / "index.json").write_text(json.dumps({"pack": PACK, "cases": cases}))
