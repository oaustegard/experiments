"""LoopCD-Hidden for Inkling 2 Mini in mlx-lm.

LoopCD (arXiv 2610.02185) is a decoding-time contrast between the first and
the last pass of a looped transformer's recurrent core:

    h' = h_R + omega * (h_R - h_1)

The guided state goes through the coda (the layers after the loop) once, so
the cost is a few vector operations per token and one extra copy of the
pass-1 state while a forward runs.

Inkling 2 Mini (model_type inkling2_loop) runs layers 0-5 once, layers 6-17
three times and layers 18-23 once. Two things differ from the paper's models:

* Every pass restarts from the prelude output and sees the previous pass only
  through a message bank, so "h_1" is the output of the first, message-free
  pass, which is the model run once with no loop.
* The coda reads a stack of block states (attention residuals over earlier
  block boundaries) as well as the hidden state. The stack is the same shape
  after every pass; it differs between passes only in the core's own entries.
  By default both are extrapolated. `guide_blocks=False` leaves the stack
  alone and guides the hidden state only.

The coda's KV caches are written from the guided state, so a prompt and a
generation see the same coda inputs (`guide_prompt=False` turns guidance off
for multi-token prefill chunks, to test whether that matters).

Needs mlx-lm with the inkling2 model package (ml-explore/mlx-lm#2086).

    from mlx_lm import load
    import loopcd
    model, tok = load("mlx-community/Inkling-2-Mini-4bit")
    loopcd.enable(model, omega=0.25)            # guided, 3 passes
    loopcd.enable(model, omega=0.25, passes=2)  # guided, one pass fewer
    loopcd.disable(model)                       # stock forward
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.inkling2 import model as _ink
from mlx_lm.models.inkling2.layers import unweighted_rms_norm

__all__ = ["LoopCD", "enable", "disable", "text_model"]

_ORIGINAL_RUN_LOOP = _ink.TextModel._run_loop


@dataclass
class LoopCD:
    omega: float = 0.25
    guide_blocks: bool = True
    guide_prompt: bool = True


def _guide(state: mx.array, ref: mx.array, omega: float) -> mx.array:
    s = state.astype(mx.float32)
    out = s + omega * (s - ref.astype(mx.float32))
    return out.astype(state.dtype)


def _run_loop_loopcd(self, hidden_states, cache, conv_mask):
    cfg: LoopCD | None = getattr(self, "loopcd", None)
    if cfg is None:
        return _ORIGINAL_RUN_LOOP(self, hidden_states, cache, conv_mask)

    # Same as TextModel._run_loop, plus a copy of the first pass's output.
    core = self.args.loop_core
    caches = iter(cache)

    def run(layers, hidden_states, block_states, bank=None):
        for i, layer in enumerate(layers):
            messages = None if bank is None else self.message_input(bank, i)
            hidden_states, block_states = layer(
                hidden_states,
                block_states,
                cache=next(caches),
                conv_mask=conv_mask,
                messages=messages,
            )
        return hidden_states, block_states

    prelude = run(self.layers[: core.start], hidden_states, None)
    hidden_states, block_states = prelude
    passes = self.args.loop_config["num_passes"]
    first = None
    for loop_pass in range(passes):
        bank = None
        if loop_pass > 0:
            bank = (
                block_states[prelude[1].shape[0] + 1 :],
                unweighted_rms_norm(hidden_states.astype(mx.float32)),
            )
            hidden_states, block_states = prelude
        hidden_states, block_states = run(
            self.layers[core.start : core.stop], hidden_states, block_states, bank
        )
        if loop_pass == 0:
            first = (hidden_states, block_states)

    guided = passes > 1 and cfg.omega != 0
    if guided and not cfg.guide_prompt and hidden_states.shape[1] > 1:
        guided = False
    if guided:
        h1, b1 = first
        hidden_states = _guide(hidden_states, h1, cfg.omega)
        if cfg.guide_blocks:
            block_states = _guide(block_states, b1, cfg.omega)
    return run(self.layers[core.stop :], hidden_states, block_states)


_ink.TextModel._run_loop = _run_loop_loopcd


def text_model(model) -> "_ink.TextModel":
    tm = model.model.language_model
    if tm.args.loop_config is None:
        raise ValueError("LoopCD needs a looped model (inkling2_loop).")
    return tm


def enable(
    model,
    omega: float = 0.25,
    *,
    passes: int | None = None,
    guide_blocks: bool = True,
    guide_prompt: bool = True,
) -> None:
    """Turn LoopCD on. `passes` changes the loop depth; set it before the
    prompt cache is made, since make_cache sizes the cache from it."""
    tm = text_model(model)
    if not hasattr(tm, "_loopcd_default_passes"):
        tm._loopcd_default_passes = tm.args.loop_config["num_passes"]
    tm.args.loop_config["num_passes"] = passes or tm._loopcd_default_passes
    tm.loopcd = LoopCD(omega, guide_blocks, guide_prompt)


def disable(model) -> None:
    tm = text_model(model)
    if hasattr(tm, "_loopcd_default_passes"):
        tm.args.loop_config["num_passes"] = tm._loopcd_default_passes
    tm.loopcd = None


def main() -> None:
    from mlx_lm import load, stream_generate

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="mlx-community/Inkling-2-Mini-4bit")
    p.add_argument("--prompt", required=True)
    p.add_argument("--omega", type=float, default=0.25)
    p.add_argument("--passes", type=int, default=None)
    p.add_argument("--hidden-only", action="store_true", help="don't guide block states")
    p.add_argument("--effort", default="low", help="Inkling reasoning_effort")
    p.add_argument("--max-tokens", type=int, default=2048)
    a = p.parse_args()

    model, tok = load(a.model)
    enable(model, a.omega, passes=a.passes, guide_blocks=not a.hidden_only)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": a.prompt}],
        add_generation_prompt=True,
        reasoning_effort=a.effort,
    )
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=a.max_tokens):
        print(r.text, end="", flush=True)
        last = r
    print(
        f"\n-- {last.generation_tokens} tokens, {last.generation_tps:.1f} tok/s, "
        f"peak {last.peak_memory:.1f} GB"
    )


if __name__ == "__main__":
    main()
