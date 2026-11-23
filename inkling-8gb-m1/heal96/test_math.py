import sys, torch, types
sys.path.insert(0, ".")
import heal
from transformers import core_model_loading as CML
import transformers.models.inkling2_loop.modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
torch.manual_seed(0)
E, H, I = 6, 64, 32
cfg = Inkling2LoopTextConfig(hidden_size=H, n_routed_experts=E, moe_intermediate_size=I, num_experts_per_tok=2)
OrigExperts = M.Inkling2LoopExperts
def rand_nvfp4(rows, cols):
    return {"weight": torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8),
            "scale": (torch.rand(E, rows, cols // 16) * 2 + 0.1).to(torch.float8_e4m3fn),
            "amax": torch.rand(E) + 0.5, "den": torch.tensor([3.0])}
def rand_mxfp8(rows, cols):
    return {"weight": (torch.randn(E, rows, cols)).to(torch.float8_e4m3fn),
            "scale": torch.randint(120, 130, (E, rows, cols // 32), dtype=torch.uint8)}
op = CML.Inkling2LoopDequantizeExperts()
for kind, mk in (("nvfp4", rand_nvfp4), ("mxfp8", rand_mxfp8)):
    p13, p2 = mk(2 * I, H), mk(H, I)
    def tf(p, interleave):
        srcs = ["w", "s", "a", "d"]
        inp = {"w": [p["weight"]], "s": [p["scale"]]}
        if "amax" in p: inp["a"] = [p["amax"]]; inp["d"] = [p["den"]]
        else: srcs = ["w", "s", "a_missing", "d_missing"]
        out = op.convert(inp, srcs, ["t"], model=None, full_layer_name=None)["t"]
        if interleave:
            out = CML.Interleave(dim=1).convert({"t": [out]}, ["t"], ["t"])["t"]
        return out
    gate_up = tf(p13, True).float(); down = tf(p2, False).float()
    mine13 = heal.dequant(p13, 0, E, torch.float32); mine2 = heal.dequant(p2, 0, E, torch.float32)
    # bf16 rounding in tf (dtype default bf16) -> compare loosely
    print(kind, "gate rows", torch.allclose(mine13[:, 0::2], gate_up[:, :I], rtol=1e-2, atol=1e-2),
          "up rows", torch.allclose(mine13[:, 1::2], gate_up[:, I:], rtol=1e-2, atol=1e-2),
          "down", torch.allclose(mine2, down, rtol=1e-2, atol=1e-2))
    # module equivalence
    ref = OrigExperts(cfg)
    ref.gate_up_proj.data = gate_up.to(torch.bfloat16); ref.down_proj.data = down.to(torch.bfloat16)
    pk = heal.PackedExperts(cfg)
    pk.set_packed({"w13": p13, "w2": p2}, list(range(E)))
    N = 40
    x = torch.randn(N, H, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.stack([torch.randperm(E)[:2] for _ in range(N)])
    w = torch.rand(N, 2)
    y_ref = ref(x, idx, w); g_ref, = torch.autograd.grad(y_ref.float().pow(2).sum(), x)
    heal.STATE.student = False
    y = pk(x, idx, w); g, = torch.autograd.grad(y.float().pow(2).sum(), x)
    rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    relg = (g.float() - g_ref.float()).norm() / g_ref.float().norm()
    print(kind, "fwd rel err", f"{rel:.4f}", "grad rel err", f"{relg:.4f}")
    # subset + lora: zero B -> identical; grads reach B
    keep = [0, 2, 3, 5]
    sub = {k: {kk: (vv[keep] if vv.shape[0] == E else vv) for kk, vv in v.items()} for k, v in (("w13", p13), ("w2", p2))}
    pk2 = heal.PackedExperts(cfg); pk2.set_packed(sub, keep); pk2.add_lora(4, 8)
    idx2 = torch.stack([torch.tensor(keep)[torch.randperm(4)[:2]] for _ in range(N)])
    heal.STATE.student = True
    ya = pk2(x, idx2, w); heal.STATE.student = False; yb = ref(x, idx2, w)
    print(kind, "subset+zero-lora rel err", f"{((ya.float()-yb.float()).norm()/yb.float().norm()):.4f}")
    heal.STATE.student = True
    pk2(x, idx2, w).float().pow(2).sum().backward()
    print(kind, "lora grads", {k: float(v.grad.abs().sum() > 0) for k, v in pk2.lora.items()})
    heal.STATE.student = False
