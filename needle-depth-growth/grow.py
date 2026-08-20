"""Grow Needle 2's layer stack with identity-initialised blocks, and check that
the grown model is functionally identical to the original.

Every per-layer tensor in this checkpoint carries a leading axis of `num_layers`
(the stack is an `nn.scan`, and the MHC lane parameters are explicit `(L, ...)`
arrays), so adding depth is a concatenation along axis 0 plus a config bump.

A new layer is the identity when:
  * `attn_gate` -> -30, so `sigmoid` underflows to 0 in fp16 and the attention
    branch contributes nothing;
  * `hadamard_mlp/d3` -> 0, so the Hadamard branch returns exactly 0;
  together these make the block return its input, so the scan body's
  `y = block(u) - u` is 0;
  * `mhc_a_res` -> 0 and `mhc_b_res` -> 40*I, so the Sinkhorn lane-mixing
    matrix is the identity to ~1e-17 and `new_x = x`.
`mhc_a_pre/post` and the phi projections are irrelevant once `y` is 0.

Appending at the end also leaves the lane assignment `arange(L) % 4` unchanged
for every existing layer, and leaves the Engram sites at layers 2 and 15.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from needle.model.run import load_checkpoint
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

ADD = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CHECKPOINT = "checkpoints/needle2.pkl"  # fetched from Cactus-Compute/needle2 on first use
SRC = -1  # duplicate the last block, as depth up-scaling does


def grow(params, cfg, add):
    flat = dict(flatten_dict(params))
    out = {}
    for path, value in flat.items():
        a = np.asarray(value)
        if path[0] == "stack" and path[1] != "final_norm" and a.shape[:1] == (cfg.num_layers,):
            extra = np.repeat(a[SRC][None], add, axis=0).copy()
            leaf = path[-1]
            if leaf == "attn_gate":
                extra[:] = -30.0
            elif leaf == "d3":
                extra[:] = 0.0
            elif leaf == "mhc_a_res":
                extra[:] = 0.0
            elif leaf == "mhc_b_res":
                extra[:] = 40.0 * np.eye(a.shape[-1], dtype=a.dtype)
            out[path] = jnp.asarray(np.concatenate([a, extra], axis=0))
        else:
            out[path] = value
    return unflatten_dict(out)


params, cfg = load_checkpoint(CHECKPOINT)
tokens = jnp.asarray([[1, 40, 512, 77, 900, 12, 3, 250, 61, 4]], jnp.int32)

base = SimpleAttentionNetwork(cfg).apply({"params": params}, tokens)

grown_cfg = TransformerConfig(**{**cfg.__dict__, "num_layers": cfg.num_layers + ADD})
grown = SimpleAttentionNetwork(grown_cfg).apply({"params": grow(params, cfg, ADD)}, tokens)

n_before = sum(np.asarray(v).size for v in flatten_dict(params).values())
n_after = sum(np.asarray(v).size for v in flatten_dict(grow(params, cfg, ADD)).values())
d = np.abs(np.asarray(base, np.float32) - np.asarray(grown, np.float32))
print(f"layers {cfg.num_layers} -> {grown_cfg.num_layers}   params {n_before:,} -> {n_after:,} "
      f"(+{(n_after - n_before) / 1e6:.2f}M, +{100 * (n_after / n_before - 1):.1f}%)")
print(f"logits max |delta| {d.max():.3e}   mean |logit| {np.abs(np.asarray(base, np.float32)).mean():.3f}")
print(f"argmax identical: {(np.asarray(base).argmax(-1) == np.asarray(grown).argmax(-1)).all()}")
