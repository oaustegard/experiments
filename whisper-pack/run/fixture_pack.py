"""Reference outputs for an existing pack directory (no repacking): mel, encoder states, first-step
logits and greedy ids for the first N ls-clean utterances, appended to test/fixtures/engine/index.json.

usage: fixture_pack.py PACK_DIR [--n 1] [--url BASE_URL]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from transformers import WhisperFeatureExtractor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--n", type=int, default=1); ap.add_argument("--url")
a = ap.parse_args()
pdir = Path(a.pack)
slug = pdir.name
out = HERE.parent / "test" / "fixtures" / "engine"
out.mkdir(parents=True, exist_ok=True)
p = ref.load(pdir)
g = p.gen
fe = WhisperFeatureExtractor.from_pretrained(p.m["model"])
prompt = ([g["sot"], g["lang_to_id"]["<|en|>"], g["task_to_id"]["transcribe"], g["no_timestamps"]]
          if g["multilingual"] else [g["sot"], g["no_timestamps"]])
cases = []
for i, pcm in enumerate(np.load(HERE.parent / "cache" / "ls-clean.npz", allow_pickle=True)["audio"][: a.n]):
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
entry = {"slug": slug, "scheme": p.m["scheme"], "pack": f"out/packs/{slug}/", "url": a.url, "prompt": prompt, "cases": cases}
ip = out / "index.json"
old = [e for e in (json.loads(ip.read_text()) if ip.exists() else []) if e["slug"] != slug]
ip.write_text(json.dumps(old + [entry]))
