"""Synthesize lines with OmniVoice (k2-fsa/OmniVoice 0.2.1). Runs inside claude-workspace scripts/jail.sh;
see README "Voiceover". Usage: run.py lines.json outdir
lines.json: [{"id":..,"text":..,"instruct":..,"speed":..,"duration":..,"seed":..}]"""
import json, sys, time, torch, soundfile as sf
from omnivoice import OmniVoice
torch.set_num_threads(4)
lines = json.load(open(sys.argv[1])); out = sys.argv[2]
t = time.time()
m = OmniVoice.from_pretrained("models/OmniVoice", dtype=torch.float32)
print(f"load {time.time()-t:.1f}s", flush=True)
for L in lines:
    torch.manual_seed(L.get("seed", 0))
    t = time.time()
    kw = {k: L[k] for k in ("instruct", "speed", "duration", "ref_audio", "ref_text") if L.get(k) is not None}
    wav = m.generate(text=L["text"], language="en", **kw)[0]
    sf.write(f"{out}/{L['id']}.wav", wav, m.sampling_rate)
    dur = len(wav) / m.sampling_rate
    print(json.dumps({"id": L["id"], "secs": round(dur, 2), "rtf": round((time.time()-t)/dur, 2),
                      "wpm": round(len(L["text"].split()) / dur * 60)}), flush=True)
