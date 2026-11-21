"""Build fixed evaluation sets -> ../cache/<set>.npz, MANIFEST.json, ../results/eval-sets.json."""
import hashlib, io, json, shutil
from pathlib import Path
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
SR = 16000
FLEURS = {"es_419": "es", "de_de": "de", "fr_fr": "fr", "nb_no": "no", "ja_jp": "ja"}


def decode(b):
    a, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(a, dtype=np.float32)


def sample(table, n, idcol, textcol):
    idx = np.random.RandomState(0).choice(table.num_rows, size=min(n, table.num_rows), replace=False)
    idx = np.sort(idx)
    rows = table.take(idx).to_pylist()
    audio = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        audio[i] = decode(r["audio"]["bytes"])
    return audio, [r[textcol] for r in rows], [str(r[idcol]) for r in rows]


def build(name):
    if name.startswith("ls-"):
        cfg = {"ls-clean": "clean", "ls-other": "other"}[name]
        p = hf_hub_download("openslr/librispeech_asr", f"{cfg}/test/0000.parquet", repo_type="dataset")
        a, txt, ids = sample(pq.read_table(p), 300, "id", "text")
        return a, txt, ids, "en"
    lang = name.split("-", 1)[1]
    p = hf_hub_download("google/fleurs", f"{lang}/test/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    a, txt, ids = sample(pq.read_table(p), 100, "id", "raw_transcription")
    return a, txt, ids, FLEURS[lang]


def main():
    CACHE.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    man = {}
    for name in ["ls-clean", "ls-other"] + [f"fleurs-{l}" for l in FLEURS]:
        a, txt, ids, lang = build(name)
        np.savez(CACHE / f"{name}.npz", audio=a, text=np.array(txt, dtype=object),
                 id=np.array(ids, dtype=object), language=np.array(lang))
        man[name] = {"count": len(ids), "audio_seconds": round(float(sum(len(x) for x in a)) / SR, 2),
                     "language": lang,
                     "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
        print(name, man[name], flush=True)
    (CACHE / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    shutil.copy(CACHE / "MANIFEST.json", RESULTS / "eval-sets.json")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
