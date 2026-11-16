"""Derive new tiny-voca voices from F2 and M2 without retraining.

Four controls, each acting on a different part of the model:

  mix      Blend of the learned voice embeddings (encoder + frame level).
           {"F2": 1.2, "M2": -0.2} extrapolates away from M2. A blended voice
           is a real new speaker row, so it survives export.
  warp     Formant scale. The waveform head predicts a cepstral filter per
           frame; its spectrum is resampled along frequency, so 0.94 moves
           every formant down 6% (a longer vocal tract, an older/larger
           speaker) without touching pitch.
  tilt     Spectral tilt in dB/octave around 1 kHz on the harmonic branch:
           negative is darker and softer, positive brighter.
  breath   Gain of the noise branch in dB: positive is breathier.

plus the model's own pitch_shift (here in semitones), pitch_range, speed and
pitch_jitter.

    python voice_lab.py PRESETS.json OUT_DIR [--text-file T] [--checkpoint C]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

import tiny_voca.model as tm
from tiny_voca.config import Config
from tiny_voca.infer import finished, limit_peak, load_model, sentences
from tiny_voca.text import phonemize, set_frontend

SHAPE = {"warp": 1.0, "tilt": 0.0, "breath": 0.0}
SEMITONE = math.log(2) / 12


def shaped_response(causal, anticausal, fft):
    """tiny_voca.model.cepstrum_response with formant warp, tilt and noise gain.

    rfft of the cepstrum gives log|H| as its real part and the (already
    continuous) phase as its imaginary part, so both can be resampled along
    frequency by plain linear interpolation before exponentiating."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., :causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1]:] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    bins = spectrum.shape[-1]
    warp = SHAPE["warp"]
    if warp != 1.0:
        source = (torch.arange(bins, dtype=torch.float32, device=buffer.device) / warp).clamp(max=bins - 1)
        low = source.floor().long()
        high = (low + 1).clamp(max=bins - 1)
        weight = source - low
        spectrum = spectrum[..., low] * (1 - weight) + spectrum[..., high] * weight
    log_magnitude = spectrum.real
    harmonic = anticausal is not None          # the noise branch has no anticausal part
    if harmonic and SHAPE["tilt"]:
        hz = torch.linspace(0, 12000, bins, device=buffer.device).clamp_min(125)
        log_magnitude = log_magnitude + SHAPE["tilt"] * math.log(10) / 20 * torch.log2(hz / 1000)
    if not harmonic and SHAPE["breath"]:
        log_magnitude = log_magnitude + SHAPE["breath"] * math.log(10) / 20
    return torch.polar(torch.exp(log_magnitude.clamp(-14, 6)), spectrum.imag)


tm.cepstrum_response = shaped_response


def with_voices(model, presets):
    """A copy of ``model`` with one extra speaker row per preset whose mix is
    not exactly a stock voice. Returns (model, {preset: speaker name})."""
    stock = model.config.speakers
    extra, speaker = [], {}
    for name, p in presets.items():
        mix = p.get("mix", {p.get("base", "F2"): 1.0})
        hit = [v for v, w in mix.items() if w == 1.0 and len(mix) == 1]
        if hit:
            speaker[name] = hit[0]
        else:
            extra.append((name, mix))
            speaker[name] = name
    if not extra:
        return model, speaker
    config = Config.from_dict({**model.config.to_dict(), "speakers": list(stock) + [n for n, _ in extra]})
    grown = tm.TinyVoca(config)
    state = model.state_dict()
    for key in ("voices.weight", "frame_voices.weight"):
        table = state[key]
        rows = [sum(w * table[stock.index(v)] for v, w in mix.items()) for _, mix in extra]
        state[key] = torch.cat((table, torch.stack(rows)))
    grown.load_state_dict(state, strict=True)
    return grown.eval(), speaker


def render(model, text, speaker, p, seed=0):
    sid = torch.tensor([model.config.speakers.index(speaker)])
    SHAPE.update(warp=p.get("warp", 1.0), tilt=p.get("tilt", 0.0), breath=p.get("breath", 0.0))
    waves, f0s = [], []
    for i, sentence in enumerate(sentences(finished(text))):
        phones = phonemize(sentence)
        tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
        wave, lengths, details = model.synthesize(
            tokens, stress, torch.tensor([len(phones)]), sid, speed=p.get("speed", 1.0),
            pitch_shift=p.get("semitones", 0.0) * SEMITONE, seed=seed + i, pitch_range=p.get("range", 1.0),
            pitch_jitter=p.get("jitter", 0.0))
        waves.append(wave[0, :int(lengths[0])])
        prosody, mask = details["prosody"][0], details["frame_mask"][0]
        voiced = mask & (prosody[:, 1] > 0)
        f0s.append(model.frequency(prosody[voiced, 0] + p.get("semitones", 0.0) * SEMITONE))
    SHAPE.update(warp=1.0, tilt=0.0, breath=0.0)
    wave = torch.cat(waves)
    if not torch.isfinite(wave).all():
        raise RuntimeError(f"non-finite audio for {speaker}")
    return limit_peak(wave), torch.cat(f0s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("w/tiny-voca.pt"))
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--save-checkpoint", type=Path, help="Write a checkpoint with the blended voices added")
    args = parser.parse_args()
    set_frontend("own")
    torch.set_num_threads(4)
    presets = json.loads(args.presets.read_text())
    texts = (args.text_file.read_text().strip().split("\n\n") if args.text_file else
             ["She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word."])
    model, speaker = with_voices(load_model(args.checkpoint, check_frontend=False), presets)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, p in presets.items():
        f0_all = []
        for t, text in enumerate(texts):
            wave, f0 = render(model, text, speaker[name], p, seed=100 * t)
            wavfile.write(args.out / f"{name}-{t + 1}.wav", model.config.sample_rate,
                          (wave.numpy() * 32767).astype(np.int16))
            f0_all.append(f0)
        f0 = torch.cat(f0_all)
        report[name] = {"f0_median_hz": round(float(f0.median()), 1),
                        "f0_p10_p90_hz": [round(float(q), 1) for q in torch.quantile(f0, torch.tensor([0.1, 0.9]))]}
        print(name, report[name], flush=True)
    (args.out / "pitch.json").write_text(json.dumps(report, indent=1))
    if args.save_checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state["config"] = model.config.to_dict()
        state["model"] = model.state_dict()
        torch.save(state, args.save_checkpoint)


if __name__ == "__main__":
    main()
