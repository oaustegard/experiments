# kokoro-demo

[`kokoro-demo.mp4`](kokoro-demo.mp4) is a 149 s, 1080p30 explainer called "Teaching Kokoro to talk". It shows nine ways to control Kokoro-82M speech, from the `making-kinescript-videos` skill's [`references/kokoro-speech.md`](https://github.com/oaustegard/claude-skills/blob/main/making-kinescript-videos/references/kokoro-speech.md). Each control is heard twice, before and after, in the voice `af_heart`. It was composed in Kinescript 1.6.1 and spoken with `kine voice` through `kinescript-kokoro` 0.1.0 (kokoro 0.9.4).

Oskar, 2026-11-02: "Use kinescript to render a video with kokoro voiceover demonstrating what the skill teaches you and the effect."

## Segments

1. Names: `[Muninn](/mˈunɪn/)` (Kokoro says MYOO-nin).
2. Script ɡ against keyboard g. The card shows the phonemes the model receives (`hˈuɪn` against `hˈuɡɪn`).
3. Emphasis: `[she](+2)` against the word as its own fragment, "Her? I never said she stole it."
4. Numbers and abbreviations: `007` and `St.`
5. Homographs: `[read](/ɹˈid/)`.
6. Pauses from punctuation.
7. Questions, which don't rise.
8. Pace: rate 1 against rate 0.8.
9. Voice blends by repetition.

The narrator is `bm_fable,bm_fable,bm_lewis` at rate 0.95.

## Pipeline

`build_scene.py` writes `kokoro-demo.json` from one list of segments. Run it with the Kokoro venv's Python: it prints each line's phonemes onto the cards, filtered through the model's vocabulary as `KModel` filters them. `kine voice` spoke all 29 sounds offline in 76 s. `kine check` reported 0 errors and 0 warnings. The render took `-q high -j 4` on the CPU backend: 4,469 frames in 103 s (43 fps), 6.6 MB, −16.4 LUFS through Kinescript's managed mix.

Kinescript and the plugin ran inside `claude-workspace/scripts/jail.sh`, with no network and `HF_HUB_OFFLINE=1`, using a copied model cache. `KINE_VOICE_CMD` was a JSON list that runs the Python 3.12 venv with its own `PYTHONPATH`. The jail's `PYTHONPATH` (cp313 wheels) would otherwise leak into it.

## Findings

- **The first cut "fixed" Kinescript to KIN-uh-script**, copying the `kinescript-kokoro` README's example. Oskar pointed out that Dimitri, who named it, says KYNE-script, which is Kokoro's default. The second cut uses Muninn as the names example and says Kinescript with no link.
- **`+2` is the stress ceiling.** Oskar heard that `[she](+2)` barely came through. misaki only adds or promotes one primary-stress mark, so `+3` and `+4` give the same phonemes. Of eleven text variants, none beat `+2` by more than about 1 st. The word as its own fragment did: in the final audio, "Her?" averaged +4.3 st over its line's median (peak +5.5), against −1.2 st (peak −0.1) for `[she](+2)`.

- **Narration pace.** At rate 0.78 the narration measured 139 wpm, not the ~180 that one-sentence tests predicted, because punctuation pauses slow real narration. Rate 0.95 gave 179 wpm. The skill's pace advice was corrected as a result (0.1.2).
- **Kokoro mispronounces its own name.** It says ko-KOR-oh. The narration uses `[Kokoro](/kˈQkəɹQ/)`, with British `Q`, for the British narrator.
- **ɡ and g look the same on screen.** In this font the two glyphs are identical, so a card showing source text cannot show the difference. The model-side phoneme line can.
