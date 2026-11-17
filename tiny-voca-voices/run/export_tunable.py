"""Export tiny-voca to ONNX with the voice controls as graph inputs.

The stock export takes a voice index. This one takes the two voice embeddings
themselves, so a page can build any voice from the stock rows and the
directions written to the metadata (``voice_space``):

    voice        float32 [1, 192]  phoneme-level voice embedding
    frame_voice  float32 [1, 192]  frame-level voice embedding
    formant    float32 [1]     frequency scale of both synthesis filters (1 = stock)
    tilt       float32 [1]     dB per octave around 1 kHz on the harmonic filter
    breath     float32 [1]     dB on the noise filter

next to the stock speed, pitch_shift and pitch_range. The formant scale is a
linear-interpolation matrix built in the graph from the input, so warping costs
one small matmul per filter per call; the cepstrum-to-spectrum matrices are
multiplied by it before they meet the per-frame cepstra.

    python export_tunable.py w/tiny-voca.pt out/tunable.onnx [--external-noise]
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_voca.export import Exportable
from tiny_voca.infer import load_model
from tiny_voca.text import SYMBOLS, set_frontend

INPUTS = ("tokens", "stress", "voice", "frame_voice", "speed", "pitch_shift", "pitch_range", "formant", "tilt", "breath")
NEPER = math.log(10) / 20


def warp_matrix(bins, formant):
    """W [bins, bins] with (spectrum @ W.T)[b] = spectrum at bin b / formant,
    linearly interpolated and held at the top bin."""
    b = torch.arange(bins, dtype=torch.float32)
    source = torch.clamp(b / formant, max=bins - 1)
    return torch.relu(1 - (b[None, :] - source[:, None]).abs())


class Tunable(Exportable):
    def __init__(self, model, external_noise=False):
        super().__init__(model, external_noise)
        c = model.config
        hz = torch.linspace(0, c.sample_rate / 2, c.synth_fft // 2 + 1).clamp_min(125)
        self.register_buffer("tilt_curve", torch.log2(hz / 1000) * NEPER, persistent=False)

    @staticmethod
    def _shaped(cepstrum, real, imaginary, offset):
        magnitude = torch.exp((cepstrum @ real + offset).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _tuned_vocoder(self, hidden, f0, total, noise, formant, tilt, breath):
        c = self.model.config
        v = self.model.vocoder
        last = total - 1
        wh = warp_matrix(c.synth_fft // 2 + 1, formant).T
        wn = warp_matrix(c.noise_fft // 2 + 1, formant).T
        h = v.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        steps = v.harmonic_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        here = torch.div(sub, steps, rounding_mode="floor")
        following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
        envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
        envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
        gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
        envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
        harmonic = self._shaped(envelope, self.harmonic_real @ wh, self.harmonic_imaginary @ wh, tilt * self.tilt_curve)
        n = v.noise(hidden)
        steps = v.noise_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        shape = n[..., :c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor")) * v.noise_lifter
        gain = n[..., c.noise_cep:].flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noisy = self._shaped(shape, self.noise_real @ wn, self.noise_imaginary @ wn, breath * NEPER)
        pulses = self._pulses(f0)
        if noise is None:
            noise = torch.randn_like(pulses)
        else:
            noise = noise.index_select(1, torch.arange(c.hop * total, device=hidden.device))
        return (self._filter(pulses, *harmonic, "synth", c.synth_fft, c.synth_hop)
                + self._filter(noise, *noisy, "noise", c.noise_fft, c.noise_hop))

    def forward(self, tokens, stress, voice, frame_voice, speed, pitch_shift, pitch_range, formant, tilt, breath, noise=None):
        m = self.model
        c = m.config
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        hidden = m.encoder(m.tokens(tokens) + m.stress(stress) + voice[:, None], token_mask)
        output = m.duration(hidden, token_mask)
        frames = torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (output[..., 1] > 0)
        durations = torch.where(present, frames, torch.zeros_like(frames))
        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(total, device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(features) + frame_voice[:, None]
        frame_mask = torch.ones_like(frame, dtype=torch.bool)
        prosody_hidden = m.prosody_stack(expanded, frame_mask)
        prosody = m.prosody_head(prosody_hidden)
        vowel = m.vowel_tokens[tokens].gather(1, index)
        voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
        voiced = (voicing > 0).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1)
        acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask)
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
        return self._tuned_vocoder(acoustic, f0, total, noise, formant, tilt, breath), durations


def example_inputs(length=24):
    tokens = torch.randint(5, len(SYMBOLS), (1, length))
    one = lambda value: torch.tensor([value], dtype=torch.float32)
    return (tokens, torch.zeros_like(tokens), torch.zeros(1, 192), torch.zeros(1, 192), one(1.0), one(0.0), one(1.0),
            one(1.0), one(0.0), one(0.0))


def voice_space(model, count=4, seed=7):
    """Stock rows of both embedding tables, plus ``count`` unit directions per
    table orthogonal to F2-M2 and to each other (the ones explore_dirs.py tested)."""
    g = torch.Generator().manual_seed(seed)
    space = {}
    for key, name in (("voices.weight", "voice"), ("frame_voices.weight", "frame_voice")):
        t = model.state_dict()[key]
        axis = (t[0] - t[1]) / (t[0] - t[1]).norm()
        ds = []
        for _ in range(count):
            d = torch.randn(t.shape[1], generator=g)
            d = d - (d @ axis) * axis
            for prev in ds:
                d = d - (d @ prev) * prev
            ds.append(d / d.norm())
        space[name] = {"rows": [[round(float(x), 6) for x in row] for row in t],
                       "distance": round(float((t[0] - t[1]).norm()), 6),
                       "directions": [[round(float(x), 6) for x in d] for d in ds]}
    return space


def export(model, path, external_noise=False, opset=17):
    wrapper = Tunable(model, external_noise).eval()
    inputs = example_inputs()
    names = list(INPUTS)
    axes = {"tokens": {1: "tokens"}, "stress": {1: "tokens"}, "audio": {1: "samples"}, "durations": {1: "tokens"}}
    if external_noise:
        inputs = (*inputs, torch.randn(1, model.config.hop * model.config.max_frames))
        names.append("noise")
        axes["noise"] = {1: "noise_samples"}
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=["audio", "durations"],
                          dynamic_axes=axes, opset_version=opset, do_constant_folding=True, dynamo=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--external-noise", action="store_true")
    args = parser.parse_args()
    set_frontend("own")
    model = load_model(args.checkpoint, check_frontend=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.output, args.external_noise)
    c = model.config
    meta = {"sample_rate": c.sample_rate, "hop": c.hop, "voices": list(c.speakers), "symbols": list(SYMBOLS),
            "max_tokens": c.max_tokens, "max_frames": c.max_frames, "inputs": list(INPUTS), "outputs": ["audio", "durations"],
            "voice_space": voice_space(model)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
