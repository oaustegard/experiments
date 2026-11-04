"""Re-apply A8p's loop edit to a fine-tuned checkpoint: W_o <- (I - u u^T) W_o in the attention output
projection of layers 27-34, with u from models/A8p/edit.json (recover_a8p.py). Fine-tuning rotates
the weights away from the edited state and loops come back; the projection is idempotent on weights
that still lack the direction, so it only removes what training put back.

    python3 reedit.py ckpt/para5e-4-e8 ckpt/para5e-4-e8-re [--edit models/A8p/edit.json]
"""
import argparse, json, shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--edit", default=str(HERE / "models/A8p/edit.json"))
    a = ap.parse_args()
    e = json.loads(Path(a.edit).read_text())
    u = torch.tensor(e["u"], dtype=torch.float32)
    u = u / u.norm()
    P = torch.eye(u.numel()) - torch.outer(u, u)
    m = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16)
    before = []
    with torch.no_grad():
        for l in e["oproj"]:
            w = m.model.layers[l].self_attn.o_proj.weight
            before.append(float((u @ w.float()).norm() / w.float().norm()))
            w.copy_((P @ w.float()).to(w.dtype))
    Path(a.dst).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(a.dst)
    AutoTokenizer.from_pretrained(a.src).save_pretrained(a.dst)
    for f in ("chat_template.json", "generation_config.json"):
        if (Path(a.src) / f).exists():
            shutil.copy(Path(a.src) / f, Path(a.dst) / f)
    print(json.dumps({"dst": a.dst, "u_share_before": [round(x, 4) for x in before]}))


if __name__ == "__main__":
    main()
