# needle-quantization — Needle 2 is already at the knee

`needle-layer-pruning/` found essentially no slack in depth. Quantization looked
like the other axis, but it starts from a different place: **Needle 2 already
ships aggressively quantized.** The checkpoint carries
`weight_bits = "embedding=4,mhc=4,default=2"`, so the bulk of the model is at
**2 bits** with two tensor families protected at 4, and the 13.74 MB blob is
that spec.

Ten arms across the exporter's whole supported range — 2, 3 and 4 bits plus
ternary — on the same 62 queries the three sibling experiments use.

**There is one move available and it is worth 10% of the file: drop the 4-bit
protection.** Every arm at 2 bits or above lands within three queries of every
other, none significantly different from shipped, across a **1.9× range in
bytes**. Above 2 bits there is no headroom to buy. Below 2 bits there is nothing
to buy — ternary packs into the same bytes as 2-bit *by construction*, and costs
39 to 57 points.

Predictions were committed in [`PREREG.md`](PREREG.md) at `ffbf410`, before any
arm was exported. Three held, three failed, one is unresolvable at this n.

## The ladder

Routable top-1 over 54 queries, paired exact McNemar against `shipped`.

| arm | MB | vs shipped | routable | Δ | ship-only | arm-only | p | spec |
|---|---|---|---|---|---|---|---|---|
| `all3` | 17.77 | +29.3% | 0.648 | +0.037 | 5 | 7 | 0.77 | `default=3` |
| `prot3` | 18.46 | +34.4% | 0.648 | +0.037 | 4 | 6 | 0.75 | `embedding=4,mhc=4,default=3` |
| `engram-tern` | 13.74 | +0.0% | 0.648 | +0.037 | 0 | 2 | 0.50 | shipped + both Engram tables ternary |
| `all2` | **12.36** | **−10.0%** | 0.630 | +0.018 | 2 | 3 | 1.00 | `default=2` |
| `mhc2` | 13.41 | −2.4% | 0.630 | +0.018 | 2 | 3 | 1.00 | `embedding=4,mhc=2,default=2` |
| `shipped` | 13.74 | — | **0.611** | — | — | — | — | `embedding=4,mhc=4,default=2` |
| `emb2` | 12.69 | −7.6% | 0.611 | +0.000 | 4 | 4 | 1.00 | `embedding=2,mhc=4,default=2` |
| `all4` | 23.17 | +68.6% | 0.593 | −0.018 | 5 | 4 | 1.00 | `default=4` |
| `prot-tern` | 13.74 | +0.0% | 0.222 | −0.389 | 23 | 2 | **<0.0001** | `embedding=4,mhc=4,default=1.58` |
| `all-tern` | 12.36 | −10.0% | 0.037 | −0.574 | 31 | 0 | **<0.0001** | `default=1.58` |

**P1 held**: `shipped` reproduces 0.611 exactly, as it did in `needle-bsky`,
`needle-tool-naming` and `needle-layer-pruning`.

## Bit width does not matter above 2

The eight arms with no ternary in the bulk span **0.593 to 0.648 — 3.0 queries
of 54** — while their blobs span **12.36 to 23.17 MB**. **Zero of the eight**
differ significantly from `shipped`. A 1.9× swing in file size buys nothing
either way.

Two of those eight are worth naming.

**`all4` is nominally the worst of them.** 23.17 MB, +68.6%, and 0.593 — one
query below shipped, well inside noise, but the point is what is *absent*: there
is no headroom above the 2-bit default. `PREREG.md` predicted `all4 ≥ shipped`
by 0.00–0.06; the measured −0.018 falsifies **P2**, in the direction that
matters for anyone who was going to spend 9 MB on it. The checkpoint's own
`weight_bits` and the exporter's `qapt` stage say why: these weights were
produced quantization-aware for this spec, so reconstructing them more precisely
recovers nothing.

**`all2` is smaller than shipped and not worse.** 12.36 MB, −10.0%, 0.630,
p=1.00. Dropping `embedding` and `mhc` from 4 bits to 2 — the protection Cactus
ships — costs nothing measurable. **P3 predicted the opposite** (≥5 points
worse) and is falsified and inverted. Split apart: `emb2` (−7.6%) lands exactly
on shipped at 0.611, `mhc2` (−2.4%) at 0.630. **P6** guessed the embedding would
hurt more than the MHC lanes, since it is 4.19M parameters *and* weight-tied to
the output head (`logits = x @ embedding.T`); `emb2` is indeed one query below
`mhc2`, which is directionally consistent and not resolvable at n=54.

## Below 2 bits there is nothing to get, by construction

The exporter's ternary option is not a smaller format. `_packed_row_bytes`
returns `in_pad * 2 // 8` for ternary, and `_pack_ternary_crumbs` stores each
value as a **2-bit crumb** (using code 3 for zero). So ternary occupies exactly
the storage of `bits=2` while offering three codebook levels instead of four —
strictly dominated before a single query is run. The measured sizes confirm it:
`prot-tern` is byte-identical to `shipped` at 13.74 MB, `all-tern` to `all2` at
12.36 MB.

And it is not a survivable trade. `prot-tern` scores **0.222** at exactly the
shipped file size — a 39-point loss for zero bytes saved, p<0.0001. `all-tern`
scores **0.037** with a refusal rate of **1.000**: it emits Needle's empty call
for every one of the 62 queries, routable and off-topic alike. **P4 held**
(predicted below 0.20); **P5 failed** — it predicted `prot-tern` would land at
0.30–0.50 and it landed below the band.

## The Engram tables ternarize for free, which is a third witness

`engram-tern` puts both 4.19M-parameter n-gram tables — 18.6% of the model — on
the three-level codebook and leaves everything else at the shipped spec. It
scores **0.648**, two queries *above* shipped, at the same 13.74 MB. **P7 held**
(predicted a cost under 0.05).

That agrees with what the other two experiments found from different directions:
`needle-layer-pruning` showed that destroying an entire Engram table costs less
than removing four well-placed layers, and `needle-tool-naming` showed this
model's routing competence lives in selection over provided context rather than
in recalled facts. The n-gram memory is not what routes tools. It saves no bytes
to ternarize it, so this is a third piece of evidence rather than a lever.

## The answer

| | |
|---|---|
| **Can we quantize further?** | Effectively no — it is already done |
| **What is left** | `default=2` everywhere: **12.36 MB, −10.0%**, no measured cost (p=1.00) |
| **What more bits buy** | Nothing. +68.6% bytes for −1 query |
| **What fewer bits buy** | Nothing — ternary is the same bytes, and costs 39–57 points |

For a model whose entire argument is fitting on a phone or a microcontroller,
10% of a 14 MB file for free is worth taking, and it is the only thing on the
table. Combined with `needle-layer-pruning`'s one surviving cut the two compose
to about 18% off the blob, but that one costs 3.7 accuracy points and buys no
latency, so the quantization half is the half worth having.

## Reproduction

```bash
python3 -m pip install --break-system-packages cactus-needle
python3 run_quant.py            # 10 arms, export + score, ~25 min on 4 CPU cores
python3 analyze.py              # ranking + paired tests -> analysis.json
python3 recheck.py              # every number above against the artifacts
```

## Caveats

- **n=54 routable.** One query is 1.85 points. The claim here is a *null across
  eight arms*, which is what 0-of-8 significant means at this n — not a proof
  that bit width is exactly irrelevant, but enough that spending 9 MB on it is
  unjustified.
- **Latency is not reported.** Four arms were scored while the box was loaded
  and show medians of 8.1–13.6 s against 1.2 s for the rest. `METHODS.md`
  already records that a busy 4-core container inflates Needle latency by an
  order of magnitude; these numbers are that, not a quantization effect, and no
  timing conclusion is drawn from this sweep.
- **Confidence is out of scope.** Every `weights=` path reports `None`, as in
  the pruning sweep. A re-quantized model's gate calibration is unmeasured.
- **One task, one catalogue.** Routing over 18 Bluesky read tools. A harder
  generation task would likely separate the bit widths where this one does not.
- **The exporter's range is 2/3/4 plus ternary.** There is no 8-bit arm to run;
  `CQ_BITS = (2, 3, 4)`. The "up" direction is bounded by the format, not by
  choice.
