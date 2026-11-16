import base64
import html
import json
from pathlib import Path

here = Path(__file__).parent
presets = json.loads((here / "presets.json").read_text())
scores = json.loads((here / "scores.json").read_text())

NOTES = {
    "F2": "Stock voice. Median pitch 268 Hz is high for the adult woman it was designed as, which is why it reads young.",
    "F2-adult": "Same embedding, pitch down 4 semitones to 216 Hz, formants down 6%, slightly flatter and slower. Aimed at the mature voice the model card says is planned.",
    "F2-adult-mix": "Embedding blended 20% toward M2, so the network itself predicts lower pitch and different timing; then 2.5 semitones down and formants down 5%.",
    "F2-bright": "Embedding pushed 25% away from M2. The model predicts a higher, brighter delivery on its own.",
    "F2-breathy": "Noise branch up 6 dB and the harmonic spectrum tilted down 1.5 dB per octave: softer and airier, same pitch.",
    "blend-50": "Halfway between the two embeddings. Lands at 214 Hz with in-between timbre.",
    "M2": "Stock voice. Median pitch 175 Hz, at the top of the usual adult male range.",
    "M2-baritone": "Pitch down 5 semitones to 131 Hz, formants down 8%, a little more pitch movement and slower.",
    "M2-light": "Pitch up 2 semitones, formants up 5%: a lighter, younger male.",
    "M2-mix-bright": "Embedding pushed 20% away from F2. The network lowers its own pitch slightly (163 Hz).",
}


def recipe(p):
    parts = []
    mix = p.get("mix", {})
    if not (len(mix) == 1 and list(mix.values())[0] == 1.0):
        parts.append("mix " + " + ".join(f"{w:g}·{v}" for v, w in mix.items()))
    if p.get("semitones"):
        parts.append(f"pitch {p['semitones']:+g} st")
    if p.get("warp", 1.0) != 1.0:
        parts.append(f"formants ×{p['warp']:g}")
    if p.get("range", 1.0) != 1.0:
        parts.append(f"range ×{p['range']:g}")
    if p.get("speed", 1.0) != 1.0:
        parts.append(f"speed ×{p['speed']:g}")
    if p.get("tilt"):
        parts.append(f"tilt {p['tilt']:+g} dB/oct")
    if p.get("breath"):
        parts.append(f"breath {p['breath']:+g} dB")
    if p.get("jitter"):
        parts.append(f"jitter {p['jitter']:g} st")
    return parts or ["stock"]


def audio(name, i):
    data = base64.b64encode((here / f"samples/{name}-{i}.mp3").read_bytes()).decode()
    return f'<audio controls preload="none" src="data:audio/mpeg;base64,{data}"></audio>'


def card(name):
    p, s = presets[name], scores[name]
    family = "m" if name.startswith("M2") else ("b" if name.startswith("blend") else "f")
    base = "M2" if family == "m" else "F2"
    chips = "".join(f'<span class="chip">{html.escape(x)}</span>' for x in recipe(p))
    return f'''<article class="voice {family}" id="{name}">
  <header><h3>{name}</h3><div class="chips">{chips}</div></header>
  <p class="note">{html.escape(NOTES[name])}</p>
  <dl class="stats">
    <div><dt>Median F0</dt><dd>{s["f0"]} Hz</dd></div>
    <div><dt>Envelope vs {base}</dt><dd>×{s["formant_scale_vs_" + base]:.3f}</dd></div>
    <div><dt>Whisper WER</dt><dd>{s["wer_pct"]:.1f}%</dd></div>
  </dl>
  <div class="clips">
    <figure><figcaption>Sea</figcaption>{audio(name, 1)}</figure>
    <figure><figcaption>Meeting</figcaption>{audio(name, 2)}</figure>
  </div>
</article>'''


female = ["F2", "F2-adult", "F2-adult-mix", "F2-bright", "F2-breathy"]
male = ["M2", "M2-baritone", "M2-light", "M2-mix-bright"]
page = f'''<title>tiny-voca Voice Lab</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column of voice cards, grouped by source voice; data set in mono. */
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --fg: #1b1e26; --muted: #5d6475; --line: #dde0e7;
  --f: #6a3fb5; --m: #b5651d; --b: #2f7d74;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12141a; --surface: #1a1d25; --fg: #e8eaf0; --muted: #9aa1b2; --line: #2c303b;
  --f: #b195ea; --m: #e3a066; --b: #6cc3b7; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12141a; --surface: #1a1d25; --fg: #e8eaf0; --muted: #9aa1b2; --line: #2c303b;
  --f: #b195ea; --m: #e3a066; --b: #6cc3b7; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.55 var(--sans);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }}
main {{ max-width: 760px; margin: 0 auto; display: grid; gap: 40px; }}
h1 {{ font-size: 2rem; line-height: 1.15; margin: 0 0 8px; text-wrap: balance; }}
h2 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 0; }}
h3 {{ font: 600 1.15rem/1.2 var(--mono); margin: 0; }}
p {{ margin: 0; max-width: 65ch; }}
.lede {{ color: var(--muted); }}
section {{ display: grid; gap: 14px; }}
.controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap: 12px; }}
.controls div {{ border-top: 2px solid var(--line); padding-top: 8px; }}
.controls b {{ font-family: var(--mono); font-weight: 500; }}
.controls p {{ color: var(--muted); font-size: 0.93rem; }}
.voice {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--fam); border-radius: 6px; padding: 18px; display: grid; gap: 12px; min-width: 0; }}
.voice.f {{ --fam: var(--f); }} .voice.m {{ --fam: var(--m); }} .voice.b {{ --fam: var(--b); }}
.voice header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px; }}
.voice h3 {{ color: var(--fam); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ font: 0.78rem/1 var(--mono); padding: 5px 7px; border: 1px solid var(--line); border-radius: 4px; color: var(--muted); }}
.note {{ font-size: 0.95rem; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0; }}
.stats dt {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.stats dd {{ margin: 0; font: 500 1rem var(--mono); font-variant-numeric: tabular-nums; }}
.clips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 10px; }}
figure {{ margin: 0; display: grid; gap: 4px; min-width: 0; }}
figcaption {{ font-size: 0.75rem; color: var(--muted); }}
audio {{ width: 100%; }}
.method {{ font-size: 0.93rem; color: var(--muted); display: grid; gap: 10px; }}
</style>
<main>
  <header>
    <h1>tiny-voca Voice Lab</h1>
    <p class="lede">Ten voices from the two in <code>n4ze3m/tiny-voca</code>, made without retraining. Each plays two sentences; play F2 against F2-adult first.</p>
  </header>
  <section>
    <h2>Controls</h2>
    <div class="controls">
      <div><b>mix</b><p>Weighted sum of the learned F2 and M2 voice embeddings, at both the phoneme and frame level. The network re-predicts timing, pitch and timbre for the new point. A blend is a real new speaker row, so it exports.</p></div>
      <div><b>formants ×k</b><p>The waveform head predicts a cepstral filter per frame. Its spectrum is resampled along frequency, so every formant moves by k with pitch untouched. Below 1 sounds larger and older.</p></div>
      <div><b>pitch, range, speed, jitter</b><p>The model's own controls: semitone shift of the F0 track, scaling of pitch movement, rate, and a small random pitch drift.</p></div>
      <div><b>tilt, breath</b><p>dB per octave on the voiced branch around 1 kHz, and gain of the noise branch. Together they set brightness and breathiness.</p></div>
    </div>
  </section>
  <section><h2>From F2</h2>{"".join(card(n) for n in female)}</section>
  <section><h2>Between</h2>{card("blend-50")}</section>
  <section><h2>From M2</h2>{"".join(card(n) for n in male)}</section>
  <section class="method">
    <h2>How these were measured</h2>
    <p>Median F0 comes from Praat over all voiced frames of both sentences. WER is Whisper small.en against the input text; every voice transcribed both sentences exactly. The envelope figure is the frequency scale that best aligns a voice's cepstrally smoothed long-term spectrum with its source voice. It understates the warp: measured at the synthesis filter itself, a 0.90 setting moves the envelope by exactly 0.90, while the audio-level estimate reads 0.975 for F2, because F2's widely spaced harmonics sample the envelope coarsely.</p>
    <p>Nobody has listened to these yet. The numbers say they are intelligible and on target; whether they sound good is the open question.</p>
  </section>
</main>
'''
(here / "voice-lab.html").write_text(page)
print(len(page) // 1024, "KB")
