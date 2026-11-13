"""Would ettin-native's packing (weight-only int8 per output row with codes held to ±63, or int4 in blocks of 32)
shrink Kokoro without moving its word times or its sound? Fake-quantizes weights and voice packs in PyTorch, then
speaks the first four narration lines with the vocoder noise seeded, so every difference is caused by the packing.

    HF_HOME=<Kokoro-82M cache> /opt/kokoro-env/bin/python pack_probe.py   -> results/pack_probe.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel, KPipeline

HERE = Path(__file__).resolve().parent
REQS = json.loads((HERE / "requests.json").read_text())[:4]
F = np.fft.rfftfreq(1024, 1 / 24000)
BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12000)]


def q_rows(w, levels):
    """Symmetric per-row quantization: codes in [-levels, levels], one scale per row (dim 0)."""
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / levels
    return (torch.round(flat / scale).clamp(-levels, levels) * scale).reshape(w.shape)


def q_blocks(w, levels=7, block=32):
    """Symmetric blockwise quantization along each row: one scale per 32 values, codes in [-7, 7]."""
    flat = w.reshape(w.shape[0], -1)
    pad = (-flat.shape[1]) % block
    x = torch.nn.functional.pad(flat, (0, pad)).reshape(flat.shape[0], -1, block)
    scale = x.abs().amax(2, keepdim=True).clamp_min(1e-12) / levels
    x = torch.round(x / scale).clamp(-levels, levels) * scale
    return x.reshape(flat.shape[0], -1)[:, :flat.shape[1]].reshape(w.shape)


def pack(model, fn, skip=()):
    """Quantize model's weights in place (deepcopy fails on weight_norm layers); restore() puts them back."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if packable(name, p) and not any(name.startswith(s) for s in skip):
                p.copy_(fn(p))
    return model


def packable(name, p):
    """Weight matrices and kernels only. weight_g (a weight_norm gain per channel) and the Snake activations'
    alpha (1, C, 1) are vectors stored as 3-D tensors; rounding a small alpha to 0 divides by zero (NaN audio)."""
    return p.dim() >= 2 and p.shape[0] > 1 and not name.endswith("weight_g") and "alpha" not in name


def restore(model):
    model.load_state_dict(ORIGINAL)


def speak(model, voice_fn=None):
    pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=model)
    if voice_fn:
        for v in {"bm_fable", "bm_lewis"}:
            pipe.voices[v] = voice_fn(pipe.load_single_voice(v).clone())
    audio, times = [], []
    for r in REQS:
        torch.manual_seed(1)
        res = list(pipe(r["text"], voice=r["speaker"], speed=r["rate"]))
        audio.append(np.concatenate([x.audio.numpy() for x in res]))
        times += [(t.start_ts, t.end_ts) for x in res for t in x.tokens if t.start_ts is not None]
    return audio, times


def power(x, n=1024, hop=256):
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), n)[::hop] * np.hanning(n)
    return (np.abs(np.fft.rfft(fr, axis=1)) ** 2).sum(0)


def score(base, arm):
    (ba, bt), (aa, at) = base, arm
    corr = [float(np.corrcoef(x[:min(len(x), len(y))], y[:min(len(x), len(y))])[0, 1]) for x, y in zip(aa, ba)]
    pa, pb = sum(power(x) for x in aa), sum(power(x) for x in ba)
    bands = [round(float(10 * np.log10(pa[(F >= lo) & (F < hi)].sum() / pb[(F >= lo) & (F < hi)].sum())), 2)
             for lo, hi in BANDS]
    same = sum(1 for x, y in zip(at, bt) if x == y) if len(at) == len(bt) else 0
    moved = [abs(x[0] - y[0]) for x, y in zip(at, bt)] if len(at) == len(bt) else [np.nan]
    return {"tokens_exact": same, "tokens": len(bt), "start_max_ms": round(1000 * float(np.max(moved)), 1),
            "corr": round(float(np.mean(corr)), 4), "bands_db": bands,
            "len_diff_s": round(float(sum(len(x) for x in aa) - sum(len(x) for x in ba)) / 24000, 3)}


model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
COUNT = {"packed": sum(p.numel() for n, p in model.named_parameters() if packable(n, p)),
         "total": sum(p.numel() for p in model.parameters()),
         "duration_path": sum(p.numel() for n, p in model.named_parameters()
                              if packable(n, p) and n.startswith(("predictor.", "bert", "text_encoder.")))}
print(COUNT)
ORIGINAL = {k: v.clone() for k, v in model.state_dict().items()}
base = speak(model)
predictor = ("predictor.", "bert", "text_encoder.")  # everything that decides durations, not the sound
arms = {
    "fp32 again (determinism check)": lambda: speak(model),
    "int8 rows ±63, all weights": lambda: speak(pack(model, lambda w: q_rows(w, 63))),
    "int8 rows ±127, all weights": lambda: speak(pack(model, lambda w: q_rows(w, 127))),
    "int8 ±63, duration path kept fp32": lambda: speak(pack(model, lambda w: q_rows(w, 63), skip=predictor)),
    "int4 blocks of 32, all weights": lambda: speak(pack(model, q_blocks)),
    "int4 blocks of 32, duration path kept fp32": lambda: speak(pack(model, q_blocks, skip=predictor)),
    "duration path fp16, decoder int8 ±63": lambda: speak(pack(pack(model, lambda w: w.half().float(),
                                                                skip=("decoder.",)), lambda w: q_rows(w, 63), skip=predictor)),
    "voice packs fp16": lambda: speak(model, lambda v: v.half().float()),
    "voice packs int8 rows ±127": lambda: speak(model, lambda v: q_rows(v.reshape(v.shape[0], -1), 127).reshape(v.shape)),
    "voice packs int4 blocks": lambda: speak(model, lambda v: q_blocks(v.reshape(v.shape[0], -1)).reshape(v.shape)),
}
out = {}
for name, run in arms.items():
    out[name] = score(base, run())
    restore(model)
    print(f"{name:45s} {out[name]}", flush=True)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "pack_probe.json").write_text(json.dumps({"params": COUNT, "arms": out}, indent=1, ensure_ascii=False))
