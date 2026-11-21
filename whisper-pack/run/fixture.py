"""Fixtures for the WGSL engine check: packs plus the NumPy reference's outputs on a few utterances.

usage: fixture.py [--model openai/whisper-tiny] [--n 2]
Writes test/fixtures/engine/index.json and, per pack, mel / encoder states / first-step logits / ids.
"""
import argparse, json
from pathlib import Path
import numpy as np
from transformers import WhisperFeatureExtractor

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import pack as packmod, ref

SCHEMES = ["all=r63,conv=f16", "tab=g32r7,enc=r31,dec=r31,conv=f16"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/whisper-tiny")
ap.add_argument("--n", type=int, default=2)
a = ap.parse_args()
out = HERE.parent / "test" / "fixtures" / "engine"
out.mkdir(parents=True, exist_ok=True)
fe = WhisperFeatureExtractor.from_pretrained(a.model)
audio = np.load(HERE.parent / "cache" / "ls-clean.npz", allow_pickle=True)["audio"][: a.n]
index = []
for k, scheme in enumerate(SCHEMES):
    slug = f"{a.model.split('/')[-1]}-{k}"
    pdir = HERE.parent / "out" / "packs" / slug
    packmod.build(a.model, scheme, pdir)
    p = ref.load(pdir)
    g = p.gen
    prompt = ([g["sot"], g["lang_to_id"]["<|en|>"], g["task_to_id"]["transcribe"], g["no_timestamps"]]
              if g["multilingual"] else [g["sot"], g["no_timestamps"]])
    cases = []
    for i, pcm in enumerate(audio):
        mel = fe(pcm, sampling_rate=16000, return_tensors="np").input_features[0].astype(np.float32)
        states = p.encode(mel)
        ids, logits = p.decode(states, prompt, 224)
        if ids and ids[-1] == g["eot"]:
            ids = ids[:-1]
        mel.tofile(out / f"{slug}-mel{i}.f32")
        states.astype(np.float32).tofile(out / f"{slug}-enc{i}.f32")
        np.asarray(logits[0], np.float32).tofile(out / f"{slug}-logits{i}.f32")
        cases.append({"i": i, "ids": [int(t) for t in ids]})
        print(slug, i, len(ids), "tokens", flush=True)
    index.append({"slug": slug, "scheme": scheme, "pack": f"out/packs/{slug}/", "prompt": prompt, "cases": cases})
ip = out / "index.json"   # keep other models' entries
old = [e for e in (json.loads(ip.read_text()) if ip.exists() else []) if e["slug"] not in {x["slug"] for x in index}]
ip.write_text(json.dumps(old + index))
