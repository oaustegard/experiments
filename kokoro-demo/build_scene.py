"""Build kokoro-demo.json: a Kinescript scene that demonstrates the Kokoro controls in
making-kinescript-videos/references/kokoro-speech.md, each as a before/after pair spoken by Kokoro.
Run with the Kokoro venv's Python (it prints each demo line's phonemes onto the cards)."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from kokoro import KPipeline

OUT = sys.argv[1]
G2P = {"a": KPipeline(lang_code="a", model=False), "b": KPipeline(lang_code="b", model=False)}

from huggingface_hub import hf_hub_download
VOCAB = json.load(open(hf_hub_download("hexgrad/Kokoro-82M", "config.json")))["vocab"]

def ps(text, voice):
    """The phonemes the model receives: misaki's output with every symbol outside Kokoro's vocabulary removed,
    as KModel does."""
    raw = " ".join(r.phonemes for r in G2P[voice[0]](text))
    return "".join(c for c in raw if c in VOCAB or c == " ")

NARRATOR = "bm_fable,bm_fable,bm_lewis"   # Muninn's voice: a 2:1 blend
N_RATE = 0.95                              # 139 wpm at 0.78, 182 at 0.97 on these lines
DEMO = "af_heart"
D_RATE = 0.85
KOKORO = "[Kokoro](/kˈQkəɹQ/)"
KINE = "[Kinescript](/kˈɪnəskɹˌɪpt/)"

BG, INK, DIM, ACC, OK, BAD = "#0b0d12", "#f2f0e8", "#8a91a0", "#ff633f", "#7ee0a1", "#ff8a7a"

# Each segment: narrator line, then a "before" and an "after" line (voice, rate, text), and an optional fact.
SEGS = [
    dict(key="names", kicker="1 · NAMES", head="Spell a name in phonemes",
         narr=f"Names first. {KOKORO} guesses words it doesn't know, and it guesses {KINE} wrong. "
              "A phoneme link in the text fixes it.",
         before=(DEMO, D_RATE, "Meet Kinescript."), after=(DEMO, D_RATE, "Meet [Kinescript](/kˈɪnəskɹˌɪpt/).")),
    dict(key="gee", kicker="2 · THE ALPHABET", head="Script ɡ (U+0261), not keyboard g",
         narr=f"Write those phonemes in {KOKORO}'s own alphabet. A plain keyboard g is not in it, "
              "and the model drops it without a word.",
         before=(DEMO, D_RATE, "[Huginn](/hˈugɪn/) and Muninn."), after=(DEMO, D_RATE, "[Huginn](/hˈuɡɪn/) and Muninn."),
         fact="ASCII g (U+0067) is missing from the vocabulary · use ɡ (U+0261)"),
    dict(key="stress", kicker="3 · EMPHASIS", head="Raise the stress",
         narr="Stress is the only emphasis it has. Raise a word two steps, or write it in capitals.",
         before=(DEMO, D_RATE, "I never said she stole it."), after=(DEMO, D_RATE, "I never said [she](+2) stole it."),
         fact="“she”: 40% longer and 3.6 semitones higher"),
    dict(key="numbers", kicker="4 · NUMBERS", head="Watch the zeros and abbreviations",
         narr="Numbers and abbreviations trip it up. Leading zeros vanish, and St. becomes a saint.",
         before=(DEMO, D_RATE, "Agent 007 lives on Main St."),
         after=(DEMO, D_RATE, "Agent [007](/dˈʌbᵊl ˈO sˈɛvən/) lives on Main Street.")),
    dict(key="homograph", kicker="5 · HOMOGRAPHS", head="Pin the reading you mean",
         narr="It picks between homographs by grammar, and habits fool it.",
         before=(DEMO, D_RATE, "I read books every night."), after=(DEMO, D_RATE, "I [read](/ɹˈid/) books every night.")),
    dict(key="pauses", kicker="6 · PAUSES", head="Punctuation is the pause control",
         narr="Punctuation sets the pauses, and a quarter of a second is the most you get.",
         before=(DEMO, D_RATE, "Write the scene, check it, render it."),
         after=(DEMO, D_RATE, "Write the scene. Check it. Render it."),
         fact="comma 0.11 s · semicolon 0.18 s · colon 0.20 s · period 0.26 s"),
    dict(key="questions", kicker="7 · INTONATION", head="Questions don't rise",
         narr="One thing it can't do. Questions don't rise. Every voice we tested fell at the end.",
         before=(DEMO, D_RATE, "You're ready."), after=(DEMO, D_RATE, "You're ready?"),
         labels=("STATEMENT", "QUESTION"), fact="both fall about ten semitones on “ready”"),
    dict(key="pace", kicker="8 · PACE", head="Slow it down",
         narr="At its default rate, it talks fast. A rate near point eight lands in a comfortable range.",
         before=(DEMO, 1.0, "The checker tells you what is wrong before you render it."),
         after=(DEMO, 0.8, "The checker tells you what is wrong before you render it."),
         labels=("RATE 1", "RATE 0.8"), fact="af_heart: 212 → 167 words a minute"),
    dict(key="blend", kicker="9 · VOICES", head="Blend by repeating",
         narr="Last, voices blend. Repeat a voice to weight it. The voice you have been hearing "
              "is two parts fable to one part lewis.",
         before=("bm_fable", N_RATE, "I keep what matters."),
         after=(NARRATOR, N_RATE, "I keep what matters."),
         labels=("ONE VOICE", "2:1 BLEND")),
]

audio, layers = [], []
layers += [
    {"id": "bg", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%",
     "fill": {"linear": [[0, 0], [1, 1]], "stops": ["#0b0d12", "#151a24"]}},
    {"use": "progress", "id": "progress", "color": ACC, "h": 6},
]

# Intro
audio.append({"id": "n0", "role": "voice", "at": 0.6, "speaker": NARRATOR, "rate": N_RATE,
              "text": f"{KOKORO} reads what you write, not what you mean. "
                      "Here are nine ways to make it say what you mean."})
layers += [
    {"id": "t-kicker", "type": "text", "text": "KOKORO TTS IN KINESCRIPT", "x": "50%", "y": "34%", "size": 34,
     "weight": 700, "letterSpacing": 6, "fill": ACC, "at": 0.3, "until": "n0.out+0.4", "enter": "fade", "exit": "fade"},
    {"id": "t-title", "type": "text", "text": "Teaching **Kokoro** to talk", "x": "50%", "y": "47%", "size": 120,
     "weight": 800, "fill": INK, "emph": ACC, "at": 0.4, "until": "n0.out+0.4", "split": "words",
     "enter": {"name": "rise", "dur": 0.6, "stagger": 0.12}, "exit": "fade"},
    {"id": "t-sub", "type": "text", "text": "nine controls, each heard before and after", "x": "50%", "y": "60%",
     "size": 44, "fill": DIM, "at": 1.2, "until": "n0.out+0.4", "enter": "fade", "exit": "fade"},
    {"id": "cc0", "type": "captions", "voice": "n0", "mode": "karaoke", "active": ACC, "size": 40},
]

prev, tail = "n0", 0.8
CARD_W, CARD_H = 790, 360
for i, s in enumerate(SEGS, 1):
    n, b, a = f"n{i}", f"b{i}", f"a{i}"
    audio += [
        {"id": n, "role": "voice", "at": f"{prev}.out+{tail + 0.3:g}", "speaker": NARRATOR, "rate": N_RATE, "text": s["narr"]},
        {"id": b, "role": "voice", "at": f"{n}.out+0.5", "speaker": s["before"][0], "rate": s["before"][1],
         "text": s["before"][2]},
        {"id": a, "role": "voice", "at": f"{b}.out+0.6", "speaker": s["after"][0], "rate": s["after"][1],
         "text": s["after"][2]},
    ]
    tail = 2.4 if s.get("fact") else 0.8
    end = f"{a}.out+{tail:g}"
    lab_b, lab_a = s.get("labels", ("BEFORE", "AFTER"))
    layers += [
        {"id": f"k{i}", "type": "text", "text": s["kicker"], "x": 150, "y": 150, "anchor": "left", "size": 34,
         "weight": 700, "letterSpacing": 4, "fill": ACC, "at": f"{n}.in", "until": end, "enter": "fade", "exit": "fade"},
        {"id": f"h{i}", "type": "text", "text": s["head"], "x": 150, "y": 235, "anchor": "left", "size": 76,
         "weight": 800, "fill": INK, "at": f"{n}.in", "until": end, "enter": {"name": "rise", "dur": 0.5},
         "exit": "fade"},
    ]
    for side, sid, (voice, rate, text), lab, col, x in (
            ("b", b, s["before"], lab_b, BAD, 150), ("a", a, s["after"], lab_a, OK, 980)):
        g = f"{side}{i}"
        layers += [
            {"id": f"{g}-card", "type": "rect", "x": x, "y": 340, "anchor": "topleft", "w": CARD_W, "h": CARD_H,
             "radius": 18, "fill": "#161b26", "stroke": "#2a3142", "strokeWidth": 2,
             "at": f"{n}.in+0.3", "until": end, "enter": "fade", "exit": "fade"},
            {"id": f"{g}-live", "type": "rect", "x": x, "y": 340, "anchor": "topleft", "w": CARD_W, "h": CARD_H,
             "radius": 18, "fill": "#00000000", "stroke": col, "strokeWidth": 4,
             "at": f"{sid}.in", "until": f"{sid}.out", "enter": {"name": "fade", "dur": 0.15},
             "exit": {"name": "fade", "dur": 0.2}},
            {"id": f"{g}-lab", "type": "text", "text": lab, "x": x + 40, "y": 390, "anchor": "left", "size": 28,
             "weight": 700, "letterSpacing": 3, "fill": col, "at": f"{n}.in+0.3", "until": end,
             "enter": "fade", "exit": "fade"},
            {"id": f"{g}-voice", "type": "text", "text": f"{voice} · rate {rate:g}", "x": x + CARD_W - 40, "y": 390,
             "anchor": "right", "size": 24, "font": "IBM Plex Mono", "fill": DIM, "at": f"{n}.in+0.3",
             "until": end, "enter": "fade", "exit": "fade"},
            {"id": f"{g}-src", "type": "text", "text": text, "x": x + 40, "y": 490, "anchor": "left", "size": 34,
             "font": "IBM Plex Mono", "fill": INK, "align": "left", "maxWidth": CARD_W - 80,
             "at": f"{n}.in+0.3", "until": end, "enter": "fade", "exit": "fade"},
            {"id": f"{g}-ps", "type": "text", "text": "→ " + ps(text, voice), "x": x + 40, "y": 620, "anchor": "left",
             "size": 30, "fill": DIM, "align": "left", "maxWidth": CARD_W - 80,
             "at": f"{sid}.in", "until": end, "enter": "type", "exit": "fade"},
        ]
    if s.get("fact"):
        layers.append({"id": f"f{i}", "type": "text", "text": s["fact"], "x": "50%", "y": 790, "size": 34,
                       "fill": INK, "box": {"fill": "#ff633f22", "pad": [12, 24], "radius": 12},
                       "at": f"{a}.in", "until": end, "enter": "rise", "exit": "fade"})
    layers.append({"id": f"cc{i}", "type": "captions", "voice": n, "mode": "karaoke", "active": ACC, "size": 40})
    prev = a

# Outro
audio.append({"id": "n10", "role": "voice", "at": f"{prev}.out+{tail + 0.3:g}", "speaker": NARRATOR, "rate": N_RATE,
              "text": f"All nine are written up, with the measurements, in the making {KINE} videos skill."})
layers += [
    {"id": "o-title", "type": "text", "text": "making-kinescript-videos", "x": "50%", "y": "44%", "size": 88,
     "weight": 800, "font": "IBM Plex Mono", "fill": INK, "at": "n10.in", "until": "n10.out+1.5",
     "enter": {"name": "type", "dur": 1.0}, "caret": True, "exit": "fade"},
    {"id": "o-sub", "type": "text", "text": "references/kokoro-speech.md · oaustegard/claude-skills", "x": "50%",
     "y": "56%", "size": 38, "fill": DIM, "at": "n10.in+1.0", "until": "n10.out+1.5", "enter": "fade",
     "exit": "fade"},
    {"id": "cc10", "type": "captions", "voice": "n10", "mode": "karaoke", "active": ACC, "size": 40},
]

scene = {"canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": BG}, "audio": audio, "layers": layers}
json.dump(scene, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"{len(audio)} sounds, {len(layers)} layers -> {OUT}")
