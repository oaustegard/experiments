"""Writes test/fixtures/tok.json: Whisper tokenizer decode cases from HF, for tok_check.mjs."""
import json, random
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import WhisperTokenizer

TEXTS = [
    "Hello, world!", " Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "It costs $1,234.56 on 2024-03-05.", "  two leading spaces", "trailing space ", "No leading space here",
    "He said: \"wait... what?!\" (really)", "100% sure; 3.14159 is pi, e = 2.718.", "Mr. O'Neil's e-mail: a@b.co",
    "Jeg heter Åse og bor i Tromsø, nær Ålesund.", " Blåbærsyltetøy på brødskiva, så godt!", "æøå ÆØÅ",
    "Dies ist ein Satz mit Umlauten: Müller, Größe, Straße.", " Über die Brücke gehen wir später.",
    "Voilà, un café crème à la française, où êtes-vous allé ?", " L'été dernier, nous avons mangé des crêpes.",
    "日本語のテキストをここに書きます。", " 今日はいい天気ですね。", "東京タワーは333メートルです。",
    "مرحبا بالعالم، كيف حالك اليوم؟", " اللغة العربية جميلة", "Привет, мир! Как дела?", "Ελληνικά κείμενα",
    "I love it 😀🎉", " 👍🏽 family 👨‍👩‍👧‍👦 flag 🇳🇴", "Mixed: hello 世界 مرحبا мир æøå 🙂",
    "Café au lait, naïve façade, jalapeño.", "你好，世界！这是一个测试。", "한국어 문장입니다.",
    "हिन्दी वाक्य", "ภาษาไทย", "tabs\tand\nnewlines\n\nhere", "ALL CAPS SHOUTING TEXT",
    "x" * 50, "a b c d e f g h", "https://example.com/path?q=1&r=2#frag", "snake_case and CamelCase",
    "1 2 3 4 5 6 7 8 9 10", "\"Quoted\" 'single' `back`", "— em dash – en dash … ellipsis",
    "Ça va très bien, merci beaucoup.", " Ich möchte ein Bier, bitte.", "Hvordan har du det i dag?",
    " Det var en gang en liten gutt som het Ola.", "Zażółć gęślą jaźń", "Čeština je krásná",
    "Türkçe karakterler: ğüşiöç", "Tiếng Việt có dấu", "עברית כתיבה", "ℝ ∑ ∫ √ ≠ ≤",
    "The year 1999 and the year 2025.", " I'm, you're, we've, they'd.", "Dr. Smith went to Washington, D.C.",
    "Okay. Um, so, like, yeah.", " thank you", "Thank you.", " you", "...", "?", " ", "a",
    "Good morning, everyone. Today we will talk about speech recognition systems and how they work.",
    "Bokmål og nynorsk: ikkje, ikke, eg, jeg.", "Straße — Maße — Fuß", "naïve résumé coöperate",
]
MODELS = ["openai/whisper-tiny", "openai/whisper-tiny.en"]
rng = random.Random(1234)


def build(model):
    tok = WhisperTokenizer.from_pretrained(model)
    path = hf_hub_download(model, "tokenizer.json")
    vocab_size = len(tok.get_vocab())
    cases = []

    def add(ids, skip, kind):
        ids = [int(i) for i in ids]
        cases.append({"kind": kind, "skip_special": skip, "ids": ids,
                      "text": tok.decode(ids, skip_special_tokens=skip)})

    def enc(t):
        return tok.encode(t, add_special_tokens=False)

    for t in TEXTS:
        add(enc(t), True, "text")
    # multi-byte split at the end / middle: cut at every byte-token boundary that breaks a char
    n_cut = 0
    for t in TEXTS:
        ids = enc(t)
        if len(ids) < 2:
            continue
        for _ in range(3):
            c = rng.randrange(1, len(ids))
            add(ids[:c], True, "cut")
            n_cut += 1
    # force emoji/CJK cuts
    for t in ["日本語のテキスト", "😀🎉👍", "æøå", "مرحبا بالعالم", "Mixed 世界 🙂"]:
        ids = enc(t)
        for c in range(1, len(ids)):
            add(ids[:c], True, "cut-forced")
            add(ids[c:], True, "cut-head")
    # specials and timestamps
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")
    t0 = tok.convert_tokens_to_ids("<|0.00|>")
    specials = [sot, eot, nots]
    if "<|en|>" in tok.get_vocab() and model.endswith("tiny"):
        specials += [tok.convert_tokens_to_ids(x) for x in ("<|en|>", "<|no|>", "<|ja|>", "<|transcribe|>")]
    for i in range(30):
        body = enc(rng.choice(TEXTS))
        a = t0 + rng.randrange(0, 1500)
        b = t0 + rng.randrange(0, 1500)
        seq = [sot] + rng.sample(specials[3:], min(2, len(specials[3:]))) + [nots] if i % 2 else [sot]
        if i % 3 == 0:
            seq += [a] + body + [b] + enc(rng.choice(TEXTS)) + [t0 + 1500, eot]
        else:
            seq += body + [eot]
        for skip in (True, False):
            add(seq, skip, "special")
    # random valid ids
    for i in range(20):
        n = rng.randrange(1, 40)
        ids = [rng.randrange(0, vocab_size) for _ in range(n)]
        add(ids, True, "random")
        add(ids, False, "random")
    return {"tokenizer_json": path, "t0": t0, "cases": cases}


out = {m: build(m) for m in MODELS}
d = Path(__file__).resolve().parent / "fixtures"
d.mkdir(exist_ok=True)
(d / "tok.json").write_text(json.dumps(out, ensure_ascii=False))
for m, v in out.items():
    print(m, len(v["cases"]), "cases")
