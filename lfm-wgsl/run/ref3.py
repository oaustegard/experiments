"""Reference logits for web/lfm3.js: run/lfm3.py on weights decoded from the exported lfm3-gq4.bin
(so the check covers the export too), on a chat prompt, then greedy decode steps.

  python3 lfm-wgsl/run/ref3.py [STEPS]    -> test/fixture3.json, test/logits3.f32
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3 import M3, ROOT, Lfm3, config  # noqa: E402
import quant3 as q  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def load(man_path=ROOT / "web" / "manifest3.json", bin_path=M3 / "web" / "lfm3-gq4.bin"):
    man = json.loads(Path(man_path).read_text())
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r")
    buf = {b["name"]: raw[b["offset"]:b["offset"] + b["size"]] for b in man["buffers"]}
    lv = torch.tensor(man["levels"], dtype=torch.float32)
    R = (q.hadamard(man["rot"]) * torch.tensor(man["signs"], dtype=torch.float64) / man["rot"] ** 0.5).float()

    def gq(spec, rows=None):
        b, N, K = buf[spec["buf"]], spec["N"], spec["K"]
        c = torch.from_numpy(np.array(b[:N * K // 2])).reshape(N, K // 2)
        codes = torch.stack([c & 15, c >> 4], -1).reshape(N, K).long()
        s = torch.from_numpy(np.array(b[N * K // 2:N * K // 2 + N * K // 16]).view(np.float16).astype(np.float32)).reshape(N, K // 32)
        Wr = (lv[codes].reshape(N, K // 32, 32) * s[..., None]).reshape(N, K // q.ROT, q.ROT)
        return (Wr @ R).reshape(N, K)[:rows or N]

    f32 = lambda name: torch.from_numpy(np.array(buf[name]).view(np.float32).copy())  # noqa: E731
    e = man["embed"]; V, D = e["N"], e["K"]
    eb = buf["embed"]
    c = torch.from_numpy(np.array(eb[:V * D // 2])).reshape(V, D // 2)
    codes = torch.stack([c & 15, c >> 4], -1).reshape(V, D).float()
    s = torch.from_numpy(np.array(eb[e["s"] * 4:e["z"] * 4]).view(np.float32).copy()).reshape(V, D // 32)
    zb = torch.from_numpy(np.array(eb[e["z"] * 4:]))[:V * D // 64].reshape(V, D // 64)
    z = torch.stack([zb & 15, zb >> 4], -1).reshape(V, D // 32).float()
    W = {"model.embed_tokens.weight": ((codes.reshape(V, D // 32, 32) - z[..., None]) * s[..., None]).reshape(V, D)}
    for i, L in enumerate(man["layers"]):
        p = f"model.layers.{i}."
        n = f32(L["norms"])
        for j, nm in enumerate(("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm")):
            W[p + nm + ".weight"] = n[j * D:(j + 1) * D]
        if L["kind"] == "conv":
            W[p + "conv.conv.weight"] = n[4 * D:].reshape(D, 1, -1)
            W[p + "conv.in_proj.weight"] = gq(L["in"])
            W[p + "conv.out_proj.weight"] = gq(L["out"])
        else:
            hd = man["head_dim"]
            W[p + "self_attn.q_norm.weight"], W[p + "self_attn.k_norm.weight"] = n[4 * D:4 * D + hd], n[4 * D + hd:]
            qkvg = gq(L["qkvg"], L["qkvg"]["rows"])
            nq, nk = man["heads"] * hd, man["kv_heads"] * hd
            W[p + "self_attn.q_proj.weight"], W[p + "self_attn.k_proj.weight"] = qkvg[:nq], qkvg[nq:nq + nk]
            W[p + "self_attn.v_proj.weight"], W[p + "self_attn.gate_proj.weight"] = qkvg[nq + nk:nq + 2 * nk], qkvg[nq + 2 * nk:]
            W[p + "self_attn.o_proj.weight"] = gq(L["out"])
        w13 = gq(L["w13"])
        W[p + "feed_forward.w1.weight"], W[p + "feed_forward.w3.weight"] = w13[:man["ffn"]], w13[man["ffn"]:]
        W[p + "feed_forward.w2.weight"] = gq(L["w2"])
    W["model.embedding_norm.weight"] = f32("final")
    return W, man


if __name__ == "__main__":
    torch.set_num_threads(4)
    W, man = load()
    chats = json.loads((M3 / "quant" / "chat.json").read_text())
    ids = chats[0]["ids"][:chats[0]["reply_start"]]
    m = Lfm3(W)
    logits, greedy = [], []
    seq = list(ids)
    for s in range(STEPS + 1):
        lg = m.logits(seq, last=1)[0]
        logits.append(lg.numpy().astype(np.float32))
        greedy.append(int(lg.argmax()))
        seq.append(greedy[-1])
    (ROOT / "test").mkdir(exist_ok=True)
    np.stack(logits).tofile(ROOT / "test" / "logits3.f32")
    (ROOT / "test" / "fixture3.json").write_text(json.dumps({"prompt": chats[0]["prompt"], "ids": ids, "greedy": greedy}))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(M3 / "tokenizer.json"))
    print(len(ids), "prompt tokens; greedy:", repr(tok.decode(greedy)))
