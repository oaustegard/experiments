"""Is the ONNX export's tonal difference from torch caused by disable_complex=True (CustomSTFT), which the export
uses? Speaks the ten lines with torch KPipeline and KModel(disable_complex=True) into out/torch312-customstft/.
With --fix it also rescales the inverse DFT rows (stft_correction) and writes out/torch312-customstft-fixed/.
Run with the Python 3.12 kokoro env and HF_HOME holding hexgrad/Kokoro-82M."""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel, KPipeline

HERE = Path(__file__).resolve().parent


def stft_correction(bins, n_fft, hop):
    """Per-bin factor that turns CustomSTFT.inverse into torch.istft: a one-sided inverse DFT counts bins 1 to
    n_fft/2 - 1 twice, and istft divides by the overlap-added squared Hann window (1.5 at n_fft 20, hop 5)."""
    w = np.hanning(n_fft + 1)[:-1]  # periodic Hann, as torch.hann_window
    overlap = float((w ** 2).sum() / hop)
    k = np.arange(bins)
    double = np.where((k == 0) | (k == n_fft // 2), 1.0, 2.0)
    return double / overlap
FIX = "--fix" in sys.argv  # rescale the inverse DFT rows as torch.istft weighs them
out = HERE / "out" / ("torch312-customstft-fixed" if FIX else "torch312-customstft")
out.mkdir(parents=True, exist_ok=True)
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
if FIX:
    stft = model.decoder.generator.stft
    scale = stft_correction(stft.freq_bins, stft.n_fft, stft.hop_length)
    stft.weight_backward_real *= torch.tensor(scale, dtype=torch.float32)[:, None, None]
    stft.weight_backward_imag *= torch.tensor(scale, dtype=torch.float32)[:, None, None]
pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=model)
for r in json.loads((HERE / "requests.json").read_text()):
    audio = np.concatenate([x.audio.numpy() for x in pipe(r["text"], voice=r["speaker"], speed=r["rate"])])
    with wave.open(str(out / f"{r['id']}.wav"), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(24000)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
print("wrote", out)
