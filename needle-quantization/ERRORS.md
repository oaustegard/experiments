# Errors — needle-quantization

## 1. The ternary size estimate was wrong by 3.5x, and it would have set the design

**What.** Scoping the sweep, I estimated ternary at ~1.58 bits per weight and
therefore a ~35% smaller blob than the shipped 13.74 MB — which framed "go below
2 bits" as the promising half of the experiment.

**Cause.** Reasoning from the name. `TERNARY_BITS = 1.58` is log2(3), the
information content; it is not the storage format. `_packed_row_bytes` returns
`in_pad * 2 // 8` for ternary and `_pack_ternary_crumbs` writes **2-bit crumbs**,
so ternary occupies exactly the storage of `bits=2`.

**How caught.** Exporting one probe blob before writing the sweep, rather than
after. `default=1.58` came back at 12.36 MB against a predicted ~8.9, which is
not a rounding disagreement.

**Direction.** Would have made "down" the headline axis of an experiment where
that axis does not exist. The finding survives — it is just a fact about the
format, provable from `export.py` in ten lines, rather than a measurement.

## 2. Four arms were scored on a loaded box and their latencies are junk

**What.** `all3`, `prot3`, `prot-tern` and `all-tern` report median turns of
8.1–13.6 s against 1.2–1.4 s for the other six.

**Cause.** The sweep ran in the background while other work continued on the
same 4-core container. `METHODS.md` already carries this exact gotcha from
`needle-bsky` — a background trainer inflated a five-tool turn from 284 ms to
3,644 ms.

**Direction.** None on accuracy: decoding is deterministic and the six clean
arms and four dirty ones interleave freely in the accuracy ranking. It does mean
this sweep contributes **no** latency number, which is stated in the writeup
rather than papered over with a caveat on a table that should not exist.

## 3. Three of seven predictions failed, one inverted

Not an error in execution, recorded because the base rate is the useful number.
**P2** (more bits help) and **P5** (ternary lands mid-range) were both wrong
about magnitude; **P3** (the 4-bit protection is load-bearing) was wrong about
*sign* — dropping it is free and 10% smaller. **P6** is unresolvable at n=54 and
is reported as such rather than claimed from a one-query difference.

The through-line with `needle-tool-naming`, whose central hypothesis also failed:
predictions formed from a plausible mechanism, and pre-registered, keep coming
back smaller or backwards. That is the pre-registration doing its job.
