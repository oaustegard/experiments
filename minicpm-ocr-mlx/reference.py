"""Reference run of MiniCPM-OCR 1.0 in transformers (fp32, CPU).

Saves the processor outputs, first-step logits and the greedy transcript for
each test image, so the MLX port can be checked against them.

    python reference.py HF_MODEL_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

PARAGRAPH = (
    "The Norwegian ship Fram carried Amundsen south in 1910.\n"
    "His party reached the Pole on 14 December 1911, five weeks\n"
    "ahead of Scott, and returned to Framheim with all five men\n"
    "and eleven of the fifty-two dogs they had started with."
)


TABLE = [
    ["Expedition", "Leader", "Reached Pole", "Dogs"],
    ["Norwegian", "Amundsen", "14 Dec 1911", "52 / 11"],
    ["Terra Nova", "Scott", "17 Jan 1912", "33 / 0"],
    ["Nimrod", "Shackleton", "turned back", "9 / 0"],
]


def make_images(out: Path) -> dict[str, tuple[Path, str]]:
    """Crops with the prompt each needs: a paragraph (single slice), a wide page
    (sliced), dense 11 px text, and a ruled table."""
    paths = {}
    font = ImageFont.truetype(FONT, 20)
    img = Image.new("RGB", (640, 140), "white")
    ImageDraw.Draw(img).multiline_text((12, 10), PARAGRAPH, font=font, fill="black", spacing=8)
    paths["paragraph"] = out / "paragraph.png"
    img.save(paths["paragraph"])

    font = ImageFont.truetype(FONT, 26)
    lines = [f"{i + 1:>2}. " + line for i, line in enumerate((PARAGRAPH + "\n" + PARAGRAPH).split("\n"))]
    img = Image.new("RGB", (1400, 560), "white")
    ImageDraw.Draw(img).multiline_text((30, 20), "\n".join(lines), font=font, fill="black", spacing=14)
    paths["page"] = out / "page.png"
    img.save(paths["page"])

    font = ImageFont.truetype(FONT, 11)
    img = Image.new("RGB", (420, 70), "white")
    ImageDraw.Draw(img).multiline_text((6, 4), PARAGRAPH, font=font, fill="black", spacing=3)
    paths["small"] = out / "small.png"
    img.save(paths["small"])

    font = ImageFont.truetype(FONT, 18)
    widths, row_h = [170, 160, 160, 110], 40
    img = Image.new("RGB", (sum(widths) + 21, row_h * len(TABLE) + 21), "white")
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(TABLE):
        x = 10
        for c, cell in enumerate(row):
            draw.rectangle((x, 10 + r * row_h, x + widths[c], 10 + (r + 1) * row_h), outline="black")
            draw.text((x + 8, 20 + r * row_h), cell, font=font, fill="black")
            x += widths[c]
    paths["table"] = out / "table.png"
    img.save(paths["table"])
    prompts = {"table": "Table Recognition:"}
    return {name: (path, prompts.get(name, "Text Recognition:")) for name, path in paths.items()}


def main(model_dir: str, out_dir: str, max_new_tokens: int = 256) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).eval()
    summary = {}
    for name, (path, task) in make_images(out).items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=[Image.open(path).convert("RGB")], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float().numpy()
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(new, skip_special_tokens=True)
        arrays = {k: v.numpy() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        np.savez(out / f"{name}_inputs.npz", **arrays)
        np.save(out / f"{name}_logits.npy", logits)
        np.save(out / f"{name}_tokens.npy", new.numpy())
        summary[name] = {
            "prompt": prompt,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "n_tokens": int(new.shape[0]),
            "text": text,
        }
        print(name, summary[name]["shapes"], repr(text))
    (out / "reference.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
