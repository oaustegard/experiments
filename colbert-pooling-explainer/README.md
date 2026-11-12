# colbert-pooling-explainer — a narrated video from ColBERT to pooling and remex

A 2¼-minute explainer, made with [Kinescript](https://github.com/drknowhow/kinescript) (main at `d254b14`, just past
2.5.0) and narrated by Kokoro in Muninn's voice. After a title card it takes nine steps:

1. a single-vector model stores one point per document; ColBERT stores a 128-number vector per token
2. MaxSim: each query token takes its best-matching document token, and the matches add up
3. what that costs: LateOn keeps about 230 vectors per document, 117,760 bytes in float32
4. the two ways to shrink it: fewer vectors (pooling) or fewer bits (quantizing)
5. pooling: Ward clustering merges tokens that point the same way into their mean
6. remex: subtract the corpus mean (LateOn's vectors sit in a narrow cone), rotate, snap each coordinate to
   Lloyd-Max levels (16 at 4 bits, 2 at 1 bit)
7. the measured trade at about 4 KB per document on SciFact: all tokens at 1 bit 0.718 nDCG@10, a quarter of the
   tokens at 4 bits 0.758, the full float index 0.763
8. letting the quantizer set the pooling: free at 3–4 bits, +0.013 on SciFact and −0.002 on NFCorpus against uniform
   pooling, so no general win
9. the takeaway: pool, keep at least 2 bits, center before quantizing

Every number comes from [`lateon-pool-remex`](../lateon-pool-remex/RESULTS.md) (the 4-rotation means in its
`summary.json`). The MaxSim similarities (0.93, 0.88, 0.91) and the dot layouts are illustrations, not measurements.

## Files

- `explainer.mp4`: 1920×1080, 30 fps, opens on the title as its poster frame
- `build_scene.py`: writes `scene.json` from one timeline (narration lines with `[[mark]]` word cues, layers cued
  to those words, karaoke captions per line)
- `scene.json`: the Kinescript scene

## Rebuild

```
python3 build_scene.py
kine voice scene.json      # Kokoro, speaker bm_fable,bm_fable,bm_lewis at rate 0.88 → voice/ (gitignored)
kine check scene.json      # 0 errors; one audio-peak warning remains (see below)
kine render scene.json -o explainer.mp4 -q high -j 4
```

`kine voice` needs the `kinescript-kokoro` plugin, which ships in the Kinescript repo (`plugins/`), not on PyPI, and
Kokoro 0.9.4 needs Python ≤ 3.12. Here it ran from a Python 3.12 venv through `KINE_VOICE_CMD`, with the Kokoro
weights fetched beforehand. In the claude-workspace setup `kine` runs inside `scripts/jail.sh`, which has no network.

## Notes

- **Pace.** Rate 0.95 measured 179 words per minute (word timings, first to last word per line). This is dense
  material, so it renders at 0.88.
- **"ColBERT" needs a phoneme link.** Written plainly, Kokoro spells it out letter by letter (`kˌɒlbˌiːˌiːˌɑːtˈiː`).
  `[ColBERT](/kQlbˈɛə/)` says "coal-bear", like the comedian it puns on. remex, MaxSim, SciFact and LateOn read
  correctly as written.
- **`kine check` caught the scene's real errors.** `hsl(...)` is a colour only inside an expression (`"=hsl(...)"`), and a layer
  with both `dur` and `until` drops `until`. Fixing them, moving two labels off the captions and lengthening the
  closing card left one warning: the mix peaks at −0.8 dBTP against a −1.5 ceiling, with `"mix": {"loudness": -17}`
  set.
