"""PyTorch reference for the WebGPU engine: the HF model with the weights the engine will see.

    from ref import load_quantized
    model = load_quantized(scheme)            # scheme: dict as in export.py, or a path to scheme.json

`load_quantized` returns the float32 HF model whose every packed tensor is replaced by exactly
export.quantize's dequantized values (q * f32(f16 scale), or the f16-rounded value), so accuracy
measured on it is the accuracy of the shipped pack. `load_model` is the same model untouched (fp32).

As a script it writes the parity fixtures test/check.mjs compares the engine against:

    python3 run/ref.py [--scheme scheme.json] [--size 512x416] [--tokens 40] [--out test/fixtures]

A synthetic text page is run at a reduced canvas (any multiple of 16 works: RADIO's CPE interpolates
the position grid) through the fp32 model and the quantized one; encoder output, first-step logits
and the first generated tokens are dumped for both.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export  # noqa: E402

MODEL_DIR = os.environ.get("NP_MODEL_DIR", "/home/user/np-work/hf")
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def load_model(model_dir=None):
    """The HF model in float32 with the checkpoint's fp32 weights.

    from_pretrained builds the model in bfloat16 (the config's torch_dtype, applied in the model's
    __init__), so the weights it holds are bf16-rounded even after .float(). The safetensors file is
    fp32; it is loaded again over the float32 model so the reference starts from the same numbers
    export.py reads.
    """
    from transformers import AutoModel
    from safetensors.torch import load_file
    src = model_dir or MODEL_DIR
    m = AutoModel.from_pretrained(src, trust_remote_code=True).float().eval()
    if os.path.isdir(src):
        path = os.path.join(src, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(src, "model.safetensors")
    sd = load_file(path)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("lm_head.")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    if m.lm_head.weight is not m.decoder.embed_tokens.weight:
        m.lm_head.weight = m.decoder.embed_tokens.weight
    assert next(m.parameters()).dtype == torch.float32
    return m


def apply_scheme(model, scheme):
    """Replace the model's weights in place by their dequantized values under `scheme`."""
    if isinstance(scheme, (str, Path)):
        scheme = json.loads(Path(scheme).read_text())
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if export.SKIP.search(name) or not p.is_floating_point():
                continue
            p.copy_(export.dequantized(name, p.float(), scheme).to(p.dtype))
    return model


def load_quantized(scheme=None, model_dir=None):
    return apply_scheme(load_model(model_dir), export.DEFAULT_SCHEME if scheme is None else scheme)


def load_processor(model_dir=None, size=None):
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(model_dir or MODEL_DIR, trust_remote_code=True)
    if size:
        ip = p.image_processor
        ip.final_size = tuple(size)
        ip.target_height, ip.target_width = size
    return p


def synthetic_page(w, h):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    def font(sz):
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
            if os.path.exists(f):
                return ImageFont.truetype(f, sz)
        return ImageFont.load_default(sz)
    d.text((24, 20), "Quarterly Report", fill="black", font=font(30))
    y = 76
    for line in ["The model reads this page and", "returns text with boxes.", "Revenue grew 12% in 2026,",
                 "while costs fell by 3%.", "", "Next steps: ship the engine", "and measure it on a real GPU."]:
        d.text((24, y), line, fill="black", font=font(21))
        y += 34
    return img


@torch.no_grad()
def run(model, proc, image, n_tokens):
    pv = proc.image_processor(image, return_tensors="pt")["pixel_values"].float()
    ids = proc.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    enc = model.encoder(pv)
    out = model(encoder_outputs=enc, decoder_input_ids=ids)
    logits = out.logits[0, -1].float()
    gen = model.generate(pixel_values=pv, decoder_input_ids=ids, max_new_tokens=n_tokens, do_sample=False,
                         num_beams=1, repetition_penalty=1.1, eos_token_id=2, pad_token_id=1,
                         forced_eos_token_id=None)   # the config forces EOS as the last allowed token; the engine does not
    new = gen[0, ids.shape[1]:].tolist()
    return {"pixels": pv[0].numpy(), "prompt": ids[0].tolist(), "enc": enc.last_hidden_state[0].float().numpy(),
            "logits": logits.numpy(), "tokens": new, "raw": proc.tokenizer.decode(new, skip_special_tokens=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme")
    ap.add_argument("--size", default="512x416", help="canvas HxW, multiples of 16")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default=str(HERE.parent / "test" / "fixtures"))
    a = ap.parse_args()
    H, W = (int(x) for x in a.size.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scheme = json.loads(Path(a.scheme).read_text()) if a.scheme else export.DEFAULT_SCHEME
    proc = load_processor(size=(H, W))
    image = synthetic_page(W - 32, H - 48)   # smaller than the canvas, so the white centre padding is exercised
    image.save(out / "page.png")
    model = load_model()
    blk = model.encoder.model_encoder.radio_model.model.blocks[0]
    print("encoder norm eps", blk.norm1.eps, "act", type(blk.mlp.act).__name__, "attn", type(blk.attn).__name__)
    res = {"size": [H, W], "image": [image.width, image.height], "prompt": None, "cases": {}}
    for tag in ("fp32", "quant"):
        if tag == "quant":
            apply_scheme(model, scheme)
        r = run(model, proc, image, a.tokens)
        res["prompt"] = r["prompt"]
        if tag == "fp32":
            r["pixels"].astype("<f4").tofile(out / "pixels.f32")
        r["enc"].astype("<f4").tofile(out / f"enc_{tag}.f32")
        r["logits"].astype("<f4").tofile(out / f"logits_{tag}.f32")
        res["cases"][tag] = {"tokens": r["tokens"], "raw": r["raw"], "argmax": int(r["logits"].argmax()),
                             "enc_shape": list(r["enc"].shape)}
        print(tag, r["tokens"], repr(r["raw"]), flush=True)
    f, q = (np.fromfile(out / f"enc_{t}.f32", "<f4") for t in ("fp32", "quant"))
    res["quant_vs_fp32"] = {"enc_max_rel": float(np.abs(f - q).max() / np.abs(f).max()),
                            "enc_cos": float(f @ q / np.sqrt((f @ f) * (q @ q)))}
    res["scheme"] = scheme
    (out / "ref.json").write_text(json.dumps(res))
    print(res["quant_vs_fp32"])


if __name__ == "__main__":
    main()
