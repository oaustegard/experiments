# The encoding contract

## Summary

A container is only meaningful together with a quantizer that agrees with the one that wrote it. This document owns the rule for what "agrees" means: which encoding parameters determine the codes, which of them a container records, which of those a quantizer checks before every decode or search, and what happens when one differs. Every other document links here instead of restating it.

The contract is enforced at the first decode or search, not when a container is created, loaded or passed around. It applies to every reading path: `decode`, `search`, `search_adc`, `search_twostage`, `search_batch`, `mse`, the IVF index and the GPU searcher.

## The simple case

The caller builds a quantizer, encodes, and reads the codes back with a quantizer built from the same arguments, in the same process or another one:

```python
Quantizer(d=384, bits=4, seed=7).encode(X).save("x.npz")
Quantizer(d=384, bits=4, seed=7).decode(CompressedVectors.load("x.npz"))
```

That works because the second quantizer rebuilds the same rotation and codebook from the same encoding parameters. Nothing is fitted to data, so there is nothing else to carry between processes.

## The parameters

| Parameter | Default | Determines the codes | Recorded in a container | Guarded |
| --- | --- | --- | --- | --- |
| `d` | required | yes | yes (`d`, and the codes' width) | no |
| `bits` | 4 | yes | yes | yes |
| `seed` | 42 | yes | no | no |
| `rotation` | `"rht"` for even `d`, `"haar"` for odd `d` | yes | yes | yes |
| `normalize` (the mode) | `True` | yes | yes, as whether `norms` is present | yes |
| `scale` | `1/sqrt(d)` in normalizing mode; 1.0 in scalar mode | yes (scalar mode only) | no | no |
| `mean` | none | yes | yes, with a `mean_dots` column | yes |
| `renorm` | `True` | no: it changes how codes are read | no | no |

`bits` must be one of 1, 2, 3, 4 or 8. Asking for 5, 6 or 7 is refused with a message that those widths "offer negligible benefit over 4-bit or 8-bit". An explicit `rotation="rht"` with an odd `d` is refused. An explicit `scale` in normalizing mode is refused, because it would be ignored.

## What is refused

Before reading any codes, every reading path compares the guarded parameters in this order and refuses at the first mismatch with a `ValueError` that names both values and says how to rebuild the quantizer:

1. **Rotation.** "rotation mismatch: these codes were encoded with 'rht' but this Quantizer uses 'haar'." A container with no rotation record is treated as `"haar"`.
2. **Mode.** "mode mismatch: these codes were encoded with normalize=True but this Quantizer uses normalize=False."
3. **Bits.** "bits mismatch: these codes are 4-bit but this Quantizer is 8-bit." A lower Matryoshka level is read with `precision=` from the quantizer that wrote the codes, never by building a quantizer at the lower width.
4. **Mean.** A centred container read without a mean, or the reverse, is refused; so is a mean that differs beyond `rtol=1e-5, atol=1e-6`, and a centred container written before remex 2.0 that lacks `mean_dots`.

A refused call has no effect: no cache is built on the container and nothing is returned.

## What a container does not record

Three of the encoding parameters are not protected, and each fails differently:

- **`seed`.** Not recorded in a `CompressedVectors`, a `PackedVectors` or a `.npz` file. A quantizer with the same `d`, `bits` and `rotation` and a different seed passes every check and decodes into a different rotation: the vectors it returns are unrelated to the originals (mean cosine −0.02 against 0.995 with the right seed, measured at `d = 64`, 4 bits), and search ranks by noise. Nothing is raised. Arrow files record the seed only when the caller passes `seed=` to `save_arrow`, and `RemexIndex` records it in its own files.
- **`scale`** (scalar mode). Not recorded. A quantizer with a different `scale` decodes every coordinate multiplied by the ratio of the two scales. Nothing is raised.
- **`d`.** Recorded but not compared. A mismatch usually fails, but with whatever error the first array operation happens to raise: `ValueError: Expected d=32, got shape (20, 64)` under `"rht"`, a NumPy `matmul` error under `"haar"`, an `IndexError` from `search_adc`. Under `rotation="none"` in scalar mode, `decode` returns the container's full width from a quantizer of another `d`, without error.

`renorm` is deliberately not part of the contract: two quantizers that differ only in `renorm` read the same codes and return slightly different vectors and scores, both valid.

## Interactions with other systems

**The encoding contract.** This document.

**Matryoshka precision.** `precision` is an argument to each reading call, not a parameter of the contract. It must be between 1 and the quantizer's `bits`, and the quantizer must match the container's `bits`.

**Determinism across machines.** Matching parameters give identical codes on every machine for `"rht"` and `"none"`; `"haar"` depends on the BLAS kernel and can differ in a few codes between machines ([determinism](../cross-cutting/determinism.md)).

**Memory and caches.** The decode cache on a container is built only after the checks pass. A cache built by one quantizer is not re-checked by name; the checks still run on every call, so a mismatched quantizer is refused even when a cache exists.

**Thread safety.** The checks read the container's recorded fields and take no lock.

**Other formats.** `.npz` and Arrow record `d`, `bits`, `rotation`, the mode and `mean`; `.pq` records the rotation and mode and refuses centred containers ([`.npz` files](../persistence/npz-files.md), [Arrow files](../persistence/arrow-files.md), [`.pq` files](../persistence/pq-files.md)).

**Version compatibility.** The default rotation changed from `"haar"` to `"rht"` in 1.0, which is why a missing rotation record means `"haar"` and not the current default. `"rht"` codes at power-of-two `d` changed in 1.0 and files written before it must be re-encoded; nothing records which version wrote a file, so this is not detected.

## Edge cases

- A quantizer built with `rotation=None` reports the rotation it picked in its `rotation` attribute; that value, not `None`, is what the container records.
- The guards read the container's attributes at call time, so a caller who reassigns `container.rotation` changes what is checked. This is the documented way to read a legacy container whose recorded rotation is wrong.
- The guards accept any object with the right attributes; they do not check its type.

## Open questions and verification

- The seed is part of the encoding by the library's own definition and is the one parameter whose mismatch is both likely (a default changed in calling code, a second service with its own default) and silent. May be worth treating as a bug rather than documenting ([B-01](../bug-triage.md#b-01-a-different-seed-decodes-without-error-to-unrelated-vectors)).
- `scale` in scalar mode has the same gap ([B-03](../bug-triage.md#b-03-scalar-mode-files-do-not-record-scale)).
- `d` is recorded and never compared ([B-06](../bug-triage.md#b-06-a-dimension-mismatch-is-not-guarded)).
- Files written before 1.0 with `"rht"` at power-of-two `d` decode wrongly and nothing detects it, because no file records the remex version that wrote it. Read from `CLAUDE.md` and the 1.0 changelog; not exercised.

Verified against oaustegard/remex commit `b3a0ed9`.
