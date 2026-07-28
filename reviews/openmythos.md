# Review: kyegomez/OpenMythos

**Reviewed:** 2026-04-21 · commit at HEAD of `main` · MIT · Python/PyTorch
**Stars:** 5,016 · **Forks:** 1,180 · **Pushed:** 2026-04-20

## TL;DR

OpenMythos is a ~4 KLoC PyTorch implementation of a Recurrent-Depth Transformer
(Prelude → looped Recurrent Block with LTI-stable input injection → Coda) with
switchable MLA/GQA attention and a DeepSeek-style MoE FFN. The architecture
description in the README is coherent and mostly backed by real code. The core
model runs end-to-end — forward and `generate` both work in a CPU smoke test.

But the packaging, tests, and README make claims the code does not deliver on:

- **14 of 67 shipped tests fail out-of-the-box** on a clean checkout with
  correct dependencies — they pass unsliced RoPE tables to attention layers
  that contractually require them pre-sliced.
- `open_mythos/moda.py` is **~1,134 lines of dead code** — a parallel MoDA+MoE
  model never imported, never wired into `OpenMythos`, and unmentioned in the
  README.
- `open_mythos/__init__.py` **eagerly imports `transformers`** via the
  tokenizer, so the core model is unusable without installing an unrelated
  heavy dep. `__all__` also lists `load_tokenizer` and `get_vocab_size`, which
  are **never defined anywhere** — bare `from open_mythos import *` will break.
- The README's quickstart calls `mythos_7b()` — **that function does not
  exist**. Variants are 1b/3b/10b/50b/100b/500b/1t.
- The MoE router has a `router_bias` buffer for "aux-loss-free load balancing
  (DeepSeek-V3)" but **nothing in the codebase ever updates it**, and no
  auxiliary load-balance loss is computed either. The docstring advertises a
  mechanism that is mechanically inert. Even `moda.py`'s version — which
  does compute a balance loss — isn't used.
- The `LTIInjection.get_A()` "spectral radius < 1 by construction" claim has a
  real hole at the clamp boundary. `A = exp(-exp(clamp(log_dt + log_A, -20, 20)))`
  saturates to exactly `1.0` in float32 when the clamp floor is hit, and
  `test_spectral_radius_stable_after_large_grad_step` catches this: after a
  single aggressive SGD step, `A.max() == 1.0` trips `assert A.max() < 1.0`.
  The guarantee is "ρ(A) ≤ 1," not "< 1."
- `pyproject.toml` pins `torch = "2.11.0"` — a hard equality on a single
  patch-less version. Aggressive for a library.
- Training script is FSDP-based, but the README table says
  `Parallelism | PyTorch DDP via torchrun`. Minor, but indicates README lag.

## Epistemic framing (the real issue)

The repo is titled **OpenMythos** and described as "a theoretical
reconstruction of the Claude Mythos architecture, built from first principles
using the available research literature." Anthropic has not released a model
called "Claude Mythos." The top-of-README disclaimer is present and explicit
("not affiliated with, endorsed by, or connected to Anthropic"), which is
appropriate. Beyond that disclaimer the prose repeatedly slides from
speculation into asserted fact:

- "Claude Mythos is suspected to be a Recurrent-Depth Transformer..."
- "...the Parcae architecture (Prairie et al., 2026), and it represents the
  most likely class of solution Anthropic used to make Mythos trainable."
- "Mythos almost certainly has some version of this."
- "If trained under these scaling laws, Mythos could be dramatically more
  parameter-efficient than it appears."

Taken together, the project is a plausible-sounding **architectural collage**
of recent published techniques (RDT, MLA, DeepSeekMoE, LTI-stable injection,
ACT halting, RoPE-over-loop-index, depth-wise LoRA), branded with an
Anthropic-adjacent name. Whether that's interesting depends entirely on what
you want it for.

## Is it useful?

| Use case | Verdict |
|---|---|
| Clean reference implementation of Parcae-style LTI injection in PyTorch | **Yes** — `LTIInjection` in main.py is the cleanest single-file example I've seen. Minor fix needed on the clamp-boundary guarantee. |
| Tutorial-grade reading for loop-transformer + MoE + MLA composition | **Yes, with caveats** — docstrings are thorough and structurally literate; but you'll trip on the dead `moda.py` and the broken `__init__`. |
| Drop-in trainable LLM stack at the 3B scale claimed | **No, not without fixes** — the tests don't pass, the `router_bias` load-balancer is inert, the `torch==2.11.0` pin will fight most setups, and there is no integration test of the full training loop on tokens. |
| Evidence about Claude's actual architecture | **No** — zero evidence is presented; the entire claim is inferred from public papers and X posts. |

## Code quality — detailed

### What's good

- **Docstrings are genuinely informative.** Each class explains the mechanism,
  the shape invariants, and the citation. This is unusually disciplined.
- **Architecture is coherent.** Prelude/Recurrent/Coda is cleanly separated.
  `RecurrentBlock.forward` correctly implements the ACT remainder trick with
  a `halted` mask gating the `cumulative_p` accumulation — subtle, done right.
- **KV cache indirection is uniform.** Attention layers accept a mutable
  `kv_cache` dict keyed by a `cache_key` per call-site. `generate()` passes
  `start_pos` so RoPE frequencies align with the growing cache.
- **MLA is implemented faithfully.** Down/up projections around a latent
  `c_kv` that's what's cached, decoupled RoPE keys, separate `freqs_cis_mla`
  buffer — matches DeepSeek-V2 structure.
- **Weight tying** between embedding and output head is present.
- **The model actually runs.** Tiny config (`dim=64`, `T=8`, 3 loops) does a
  forward+`generate` on CPU with no errors for both `gqa` and `mla`.

### What's problematic

**1. Tests are broken at the API-contract level** (`test_main.py`).
`GQAttention.forward` and `MLAttention.forward` take pre-sliced `freqs_cis`
(the top-level `OpenMythos.forward` slices it before passing). The tests call
`self.attn(x, self.freqs)` with the full `max_seq_len` table. The first
`apply_rope` dies with `RuntimeError: size mismatch`. Fix is one character in
the test (`self.freqs[:T]`), but the fact that the shipped test suite doesn't
pass suggests the author never ran it. Affected tests: all `TestGQAttention`,
all `TestMLAttention`, `TestTransformerBlock`, `TestRecurrentBlock`.

**2. The `router_bias` load-balancer is decorative.** `MoEFFN` registers it
as a buffer and uses it for top-K selection:

```python
logits = self.router(flat)              # unbiased
scores = F.softmax(logits, dim=-1)
_, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
```

DeepSeek-V3's aux-loss-free routing requires updating this bias per step
based on expert utilization. The training script (`training/3b_fine_web_edu.py`)
never touches `router_bias`, and there is no balance-loss term added to the
LM loss. Net effect: `router_bias == 0` for the entire run, and routing
collapse is entirely possible. The docstring promise is unfulfilled.

Notably, `moda.py` — the unused parallel implementation — **does** compute a
balance loss. That code exists, just not where it's called.

**3. `LTIInjection` stability "by construction" has a boundary hole.**

```python
return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
```

The clamp at `-20` was added for float32 numerical safety (the docstring
notes: "Clamp keeps the product finite in float32 for any gradient step
size"). But `exp(-exp(-20)) = exp(-2.06e-9)` rounds to exactly `1.0` in
float32. After a single `SGD(lr=1e3)` step pushing `log_dt + log_A` to the
lower clamp, `A` is all-ones — precisely on the stability boundary, not
strictly inside it. The shipped test
`test_spectral_radius_stable_after_large_grad_step` catches this with
`assert A.max().item() < 1.0`, and it fails.

The honest fix is either (a) clamp to `(-20, 20)` on `log_A` only, keeping
the structural negativity guarantee intact without squashing `dt → 0`, or
(b) weaken the claim to "ρ(A) ≤ 1 by construction" and document the
fixed-point behaviour at the boundary.

**4. `open_mythos/moda.py` is orphaned.** 1,134 lines of independent
MoDA+DeepSeekMoE model with its own `MoDAConfig`, `MoDABlock`, `MoDAModel`,
`RMSNorm` (duplicate), `RotaryEmbedding` (duplicate), etc. No import of it
anywhere in the package, in tests, or in training. Either ship it as an
example in `examples/`, expose it via `__init__.py` as a second model class,
or remove it. Right now it just bloats installs and confuses readers.

**5. `open_mythos/__init__.py` imports `transformers` unconditionally** via
`from open_mythos.tokenizer import MythosTokenizer`. This makes the core
model unimportable without a 600MB-ish dependency that's irrelevant to model
construction. Tokenizer should be a lazy import or a separate module the
user opts into.

**6. `__all__` lists phantom symbols.** `"load_tokenizer"` and
`"get_vocab_size"` are declared in `__all__` but never imported or defined.
`from open_mythos import *` will succeed but the symbols will be missing.

**7. `mythos_7b()` is called in the README quickstart but doesn't exist.**
Variant functions are 1b/3b/10b/50b/100b/500b/1t.

**8. Training table / code mismatch.** README says `Parallelism | PyTorch
DDP via torchrun`. The script actually wraps with
`torch.distributed.fsdp.FullyShardedDataParallel`. FSDP != DDP.

**9. `torch == 2.11.0`** exact pin in `pyproject.toml` vs `torch>=2.1.0` in
`requirements.txt`. The Poetry pin will fight most installations.

## References section — nothing verified

I did not verify the arXiv IDs (`2604.07822`, `2604.12946`, `2603.21852`,
`2603.15619`) or the X status-ID links. The prefixes (26xx) are plausible for
an April-2026 repo, but the reader should independently verify each claim
before citing this README as secondary source material. In particular, the
"Parcae (Prairie et al., 2026)" paper does most of the architectural heavy
lifting in the README's argument and is worth a direct read.

## What I'd do if I forked this

- Fix the RoPE slicing in `test_main.py` — single-line change per affected
  test — and run the suite in CI. Test pass rate should be 100% before a
  release tag.
- Move `transformers` import inside `MythosTokenizer.__init__` (lazy).
- Delete `load_tokenizer`/`get_vocab_size` from `__all__`, or implement them.
- Either delete `moda.py` or expose a `MoDAModel` top-level option and
  deduplicate `RMSNorm`/`RotaryEmbedding` against `main.py`.
- Add `mythos_7b()` in `variants.py` (or fix the README).
- Wire the aux-loss-free bias update into the training loop (it's maybe 15
  lines): after each step, compute per-expert assignment counts and nudge
  `router_bias` toward balance. Or add the `moda.py`-style explicit balance
  loss. Ship one or the other; don't advertise both and deliver neither.
- Rework `LTIInjection.get_A()` to preserve a strict `ρ(A) < 1` or weaken
  the docstring claim to "≤ 1 at the clamp boundary."
- Relax the `torch == 2.11.0` pin to `torch >= 2.1`.
- Update the README training table to say FSDP, not DDP.

## Bottom line

The code is better than kyegomez's average — the docstrings are thoughtful,
the architecture composition is legible, the core model actually runs — but
it is not production-grade and it is not a reconstruction of anything; it is
a pastiche of 2024–2026 loop-transformer ideas under an Anthropic-adjacent
brand. Read the source for pedagogy, not for truth about Claude.
