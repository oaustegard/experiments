"""Compare every arm in out/ with the reference arm (torch312, the parent session's working setup).

Word times: per-word |start - ref start| and |end - ref end| over all ten lines, after checking the word lists
match. Phones: whether each arm's phone list (misaki's phonemes per word) equals the reference's. Pace: words per
minute of speaking time (first word start to last word end). Audio: length difference, waveform correlation and
log-spectrogram distance (mean |dB| difference over a 1024/256 STFT, bins above -60 dB of the loudest), each also
for a second run of the reference arm (torch312b), because Kokoro's vocoder adds random noise on every call.

    python3 compare.py            -> results/compare.json and a table on stdout
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REF = "torch312"


def load(arm):
    return json.loads((OUT / arm / "result.json").read_text())


def wav(arm, sid):
    with wave.open(str(OUT / arm / f"{sid}.wav")) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32767


def logspec(x, n=1024, hop=256):
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)


def spec_db(a, b):
    m = min(len(a), len(b))
    sa, sb = logspec(a[:m]), logspec(b[:m])
    keep = np.maximum(sa, sb) > max(sa.max(), sb.max()) - 60
    return float(np.abs(sa - sb)[keep].mean())


def wpm(words):
    return len(words) / (words[-1]["end"] - words[0]["start"]) * 60


def compare(arm, ref):
    a, r = load(arm), load(ref)
    ds, de, same_words, same_phones, corr, sdb, dlen = [], [], True, True, [], [], []
    for la, lr in zip(a["lines"], r["lines"]):
        if [w["word"] for w in la["words"]] != [w["word"] for w in lr["words"]]:
            same_words = False
            continue
        ds += [abs(x["start"] - y["start"]) for x, y in zip(la["words"], lr["words"])]
        de += [abs(x["end"] - y["end"]) for x, y in zip(la["words"], lr["words"])]
        same_phones &= [p["ipa"] for p in la["phones"]] == [p["ipa"] for p in lr["phones"]]
        xa, xr = wav(arm, la["id"]), wav(ref, lr["id"])
        m = min(len(xa), len(xr))
        dlen.append((len(xa) - len(xr)) / 24000)
        corr.append(float(np.corrcoef(xa[:m], xr[:m])[0, 1]))
        sdb.append(spec_db(xa, xr))
    ds, de = np.array(ds), np.array(de)
    words = [w for line in a["lines"] for w in line["words"]]
    return {
        "arm": arm, "python": a["python"], "engine": a["engine"],
        "same_words": same_words, "same_phones": bool(same_phones), "n_words": len(words),
        "start_mean_ms": round(1000 * ds.mean(), 2), "start_max_ms": round(1000 * ds.max(), 1),
        "end_mean_ms": round(1000 * de.mean(), 2), "end_max_ms": round(1000 * de.max(), 1),
        "words_exact": int(((ds == 0) & (de == 0)).sum()),
        "words_within_25ms": int(((ds <= 0.025) & (de <= 0.025)).sum()),
        "wpm": round(float(np.mean([wpm(line["words"]) for line in a["lines"]])), 1),
        "audio_s": a["audio_s"], "len_diff_max_s": round(float(np.max(np.abs(dlen))), 3),
        "corr_mean": round(float(np.mean(corr)), 4), "spec_db_mean": round(float(np.mean(sdb)), 2),
        "load_s": a["load_s"], "synth_s": a["synth_s"], "rtf": round(a["synth_s"] / a["audio_s"], 3),
        "peak_rss_mb": a["peak_rss_mb"],
    }


arms = sorted(p.name for p in OUT.iterdir() if (p / "result.json").exists())
rows = [compare(arm, REF) for arm in arms]
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "compare.json").write_text(json.dumps({r["arm"]: r for r in rows}, indent=1))
cols = ["arm", "python", "same_phones", "words_exact", "words_within_25ms", "start_mean_ms", "end_max_ms", "wpm",
        "audio_s", "corr_mean", "spec_db_mean", "load_s", "synth_s", "rtf", "peak_rss_mb"]
print(" | ".join(cols))
for r in rows:
    print(" | ".join(str(r[c]) for c in cols))
