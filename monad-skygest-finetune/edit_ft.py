"""Remove the fine-tuned model's own loop direction (rediag.py's u_perp) from chosen writers.

    python3 edit_ft.py --name E1 --down 30
    python3 edit_ft.py --name E3 --oproj 27,28,29,30,31,32,33,34

W <- (I - u u^T) W for the attention output projection (o_proj) of the --oproj layers and the MLP output
projection (down_proj) of the --down layers, computed in fp32 and written back in bf16 like the original
weights. Writes models/ft_<name>/ (HF dir) with edit.json.
"""
import argparse, json, shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", default=str(HERE / "models/skygest_e8"))
    ap.add_argument("--oproj", default="")
    ap.add_argument("--down", default="")
    a = ap.parse_args()
    u = torch.load(HERE / "models/loopdir_ft.pt")["u_perp"].float()
    u = u / u.norm()
    P = torch.eye(u.numel()) - torch.outer(u, u)
    m = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16)
    op = [int(x) for x in a.oproj.split(",") if x]
    dn = [int(x) for x in a.down.split(",") if x]
    with torch.no_grad():
        for l in op:
            w = m.model.layers[l].self_attn.o_proj.weight
            w.copy_((P @ w.float()).to(w.dtype))
        for l in dn:
            w = m.model.layers[l].mlp.down_proj.weight
            w.copy_((P @ w.float()).to(w.dtype))
    out = HERE / f"models/ft_{a.name}"
    shutil.rmtree(out, ignore_errors=True)
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(a.src).save_pretrained(out)
    for f in ("chat_template.json", "generation_config.json"):
        if (Path(a.src) / f).exists():
            shutil.copy(Path(a.src) / f, out / f)
    (out / "edit.json").write_text(json.dumps({"src": a.src, "oproj": op, "down": dn, "u": u.tolist()}))
    print("saved", out)


if __name__ == "__main__":
    main()
