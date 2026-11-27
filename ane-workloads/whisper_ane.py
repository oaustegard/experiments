#!/usr/bin/env python3
"""Whisper large-v3-turbo on the Neural Engine, GPU idle and GPU busy.

Three engines on the same audio, each with the GPU idle and then under a llama-bench decode:

  whisperkit   WhisperKit's CLI (Core ML encoder and decoder, both pinned to
               cpuAndNeuralEngine), model openai_whisper-large-v3-v20240930_turbo
  whispercpp   whisper.cpp built with -DWHISPER_COREML=1 and run with --no-gpu: the
               Core ML encoder plus a CPU decoder. Upstream loads the encoder with
               MLComputeUnitsAll, which lets Core ML pick the GPU; this script patches that
               one line to MLComputeUnitsCPUAndNeuralEngine before building
               (--whispercpp-units all keeps upstream's choice)
  mlx          mlx-whisper, mlx-community/whisper-large-v3-turbo, on the GPU (the contrast)

Audio: LibriSpeech test-clean, consecutive utterances of one chapter concatenated into
windows of at most --window seconds (Whisper's encoder always processes 30 s, so a short
clip costs as much as a full one), --windows of them spread evenly over the set.

Reported per engine and condition: WER (jiwer, both sides through Whisper's English text
normaliser), realtime factor (processing seconds per audio second, model load excluded),
and the load time of the first invocation separately from a second one. The first Core ML
load on a machine compiles the model for the Neural Engine and the OS caches the result, so
"first" is a compile only if this model has never been loaded on this machine; the script
records whether its model folders existed before it started.

    python3 whisper_ane.py                       # all three, about 6 GB of downloads
    python3 whisper_ane.py --only whisperkit     # repeatable
    python3 whisper_ane.py --windows 8 --no-gpu-busy   # a quick check

Requires: python3 >= 3.11, Xcode command line tools, cmake, and `brew install whisperkit-cli`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

import common
from common import log, run

PINS = ["jiwer==4.0.0", "whisper-normalizer==1.0.1", "soundfile==0.14.0", "numpy",
        "huggingface_hub>=0.30"]
MLX_PINS = ["mlx-whisper==0.4.3"]
LIBRISPEECH = "https://www.openslr.org/resources/12/test-clean.tar.gz"
WHISPERKIT_MODEL = "large-v3-v20240930_turbo"  # argmaxinc/whisperkit-coreml, openai_whisper-<this>
WHISPERCPP_REPO = "https://github.com/ggml-org/whisper.cpp"
WHISPERCPP_COMMIT = "79b11911d9d38c4be2fc7dc8b4efeff3324c090f"  # 2026-11-26
GGML_HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
GGML_BIN = "ggml-large-v3-turbo.bin"
GGML_ENCODER_ZIP = "ggml-large-v3-turbo-encoder.mlmodelc.zip"
MLX_REPO = "mlx-community/whisper-large-v3-turbo"
UNITS_LINE = "config.computeUnits = MLComputeUnitsAll;"
UNITS_NE = "config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;"
SR = 16000


# ---------------------------------------------------------------- audio

def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    log(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)
    return dest


def plan_windows(utts, max_s):
    """utts: [(chapter, utt_id, seconds, text)] in corpus order. Greedy: consecutive
    utterances of one chapter while the total stays within max_s. Returns lists of indices."""
    out, cur, cur_s, cur_ch = [], [], 0.0, None
    for i, (ch, _, sec, _) in enumerate(utts):
        if cur and (ch != cur_ch or cur_s + sec > max_s):
            out.append(cur)
            cur, cur_s = [], 0.0
        if sec > max_s:
            continue
        cur.append(i)
        cur_s += sec
        cur_ch = ch
    if cur:
        out.append(cur)
    return out


def pick_evenly(items, n):
    if n >= len(items):
        return list(items)
    return [items[(2 * k + 1) * len(items) // (2 * n)] for k in range(n)]


def prepare_audio(args):
    """Windows as 16 kHz mono PCM16 wav files plus their reference text."""
    import numpy as np
    import soundfile as sf

    adir = common.WORK / "whisper" / f"windows-{args.windows}x{args.window:g}s"
    manifest = adir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    root = common.WORK / "whisper" / "LibriSpeech" / "test-clean"
    if not root.exists():
        tgz = download(LIBRISPEECH, common.WORK / "whisper" / "test-clean.tar.gz")
        log("unpacking LibriSpeech test-clean")
        with tarfile.open(tgz) as t:
            t.extractall(common.WORK / "whisper", filter="data")
    utts = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            info = sf.info(str(flac))
            utts.append((trans.parent.name, str(flac), info.frames / info.samplerate, text))
    plans = pick_evenly([p for p in plan_windows(utts, args.window)
                         if sum(utts[i][2] for i in p) >= args.window * 0.6], args.windows)
    adir.mkdir(parents=True, exist_ok=True)
    wins = []
    for k, p in enumerate(plans):
        pcm = []
        for i in p:
            x, sr = sf.read(utts[i][1], dtype="int16")
            if sr != SR:
                raise RuntimeError(f"{utts[i][1]}: {sr} Hz, expected {SR}")
            pcm.append(x)
        x = np.concatenate(pcm)
        path = adir / f"w{k:03d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(x.tobytes())
        wins.append({"id": path.stem, "path": str(path), "seconds": len(x) / SR,
                     "ref": " ".join(utts[i][3] for i in p),
                     "utterances": [Path(utts[i][1]).stem for i in p]})
    manifest.write_text(json.dumps(wins, indent=1))
    return wins


def score(wins, hyps):
    """Corpus WER over the windows that have a hypothesis; a missing one counts as empty."""
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    refs = [norm(w["ref"]) for w in wins]
    hyp = [norm(hyps.get(w["id"]) or "") for w in wins]
    o = jiwer.process_words(refs, hyp)
    return {"wer": round(o.wer, 5), "substitutions": o.substitutions, "deletions": o.deletions,
            "insertions": o.insertions, "ref_words": o.hits + o.substitutions + o.deletions,
            "windows_missing": sum(1 for w in wins if not hyps.get(w["id"]))}


# ---------------------------------------------------------------- WhisperKit

class WhisperKit:
    name = "whisperkit"

    def setup(self, args):
        self.bin = args.whisperkit_cli or shutil.which("whisperkit-cli")
        if not self.bin:
            raise RuntimeError("whisperkit-cli not found: brew install whisperkit-cli")
        self.models = common.WORK / "whisper" / "whisperkit-models"
        self.units = args.whisperkit_units
        info = {"binary": str(self.bin), "version": common.sh([self.bin, "--version"]),
                "model": WHISPERKIT_MODEL, "compute_units": self.units,
                "models_existed_before": self.models.exists()}
        return info

    def transcribe(self, wins, raw, tag):
        rep = raw / f"whisperkit-{tag}"
        if rep.exists():
            shutil.rmtree(rep)
        rep.mkdir(parents=True)
        cmd = [self.bin, "transcribe", "--model", WHISPERKIT_MODEL,
               "--download-model-path", self.models,
               "--download-tokenizer-path", self.models / "tokenizer",
               "--audio-encoder-compute-units", self.units,
               "--text-decoder-compute-units", self.units,
               "--language", "en", "--chunking-strategy", "none",
               "--concurrent-worker-count", "1", "--report", "--report-path", rep]
        for w in wins:
            cmd += ["--audio-path", w["path"]]
        t = time.time()
        rc, out, err = run(cmd, timeout=7200)
        wall = time.time() - t
        (rep / "log.txt").write_text(f"{cmd}\nrc={rc}\n--- stdout\n{out[-20000:]}\n--- stderr\n{err[-20000:]}")
        r = {"rc": rc, "wall_s": round(wall, 2), "hyps": {}, "t0": t, "t1": t + wall}
        if rc != 0:
            r["error"] = (err or out)[-600:]
        proc = audio = 0.0
        for w in wins:
            f = rep / f"{w['id']}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            r["hyps"][w["id"]] = d.get("text", "")
            tm = d.get("timings") or {}
            proc += tm.get("fullPipeline", 0.0)
            audio += tm.get("inputAudioSeconds", 0.0)
            for k in ("modelLoading", "encoderLoadTime", "decoderLoadTime", "prewarmLoadTime",
                      "encoderSpecializationTime", "decoderSpecializationTime", "tokenizerLoadTime"):
                if k in tm:
                    r.setdefault("load", {})[k] = round(tm[k], 3)
            for k in ("encoding", "decodingLoop", "logmels", "audioLoading"):
                if k in tm:
                    r.setdefault("stages_s", {})[k] = round(r.get("stages_s", {}).get(k, 0) + tm[k], 3)
        r["engine_s"] = round(proc, 3) if proc else None
        r["load_s"] = (r.get("load") or {}).get("modelLoading")
        return r


# ---------------------------------------------------------------- whisper.cpp

class WhisperCpp:
    name = "whispercpp"

    def setup(self, args):
        if shutil.which("cmake") is None:
            raise RuntimeError("cmake not found: brew install cmake")
        self.units = args.whispercpp_units
        src = common.WORK / "whisper" / "whisper.cpp"
        build = f"build-{self.units}"
        self.bin = src / build / "bin" / "whisper-cli"
        mdir = common.WORK / "whisper" / "ggml"
        self.model = mdir / GGML_BIN
        enc = mdir / GGML_BIN.replace(".bin", "-encoder.mlmodelc")
        info = {"commit": WHISPERCPP_COMMIT, "encoder_compute_units": self.units,
                "models_existed_before": enc.exists()}
        if not src.exists():
            log("cloning whisper.cpp")
            for cmd in (["git", "init", "-q", src],
                        ["git", "-C", src, "fetch", "-q", "--depth", "1", WHISPERCPP_REPO, WHISPERCPP_COMMIT],
                        ["git", "-C", src, "checkout", "-q", "FETCH_HEAD"]):
                rc, _, err = run(cmd, timeout=1800)
                if rc != 0:
                    raise RuntimeError(f"{cmd[1:4]} failed: {err[-1500:]}")
        mm = src / "src" / "coreml" / "whisper-encoder.mm"
        text = mm.read_text()
        want = UNITS_NE if self.units == "ne" else UNITS_LINE
        if want not in text:
            other = UNITS_LINE if self.units == "ne" else UNITS_NE
            if other not in text:
                raise RuntimeError(f"{mm}: compute-units line not found; upstream changed")
            mm.write_text(text.replace(other, want))
        info["encoder_units_line"] = want
        if not self.bin.exists():
            log(f"building whisper.cpp with Core ML ({build})")
            rc, _, err = run(["cmake", "-B", build, "-DWHISPER_COREML=1", "-DCMAKE_BUILD_TYPE=Release"],
                             cwd=src, timeout=1800)
            if rc != 0:
                raise RuntimeError("cmake configure failed: " + err[-3000:])
            rc, _, err = run(["cmake", "--build", build, "-j", "--config", "Release",
                              "--target", "whisper-cli"], cwd=src, timeout=3600)
            if rc != 0:
                raise RuntimeError("build failed: " + err[-3000:])
        mdir.mkdir(parents=True, exist_ok=True)
        download(GGML_HF + GGML_BIN, self.model)
        if not enc.exists():
            z = download(GGML_HF + GGML_ENCODER_ZIP, mdir / GGML_ENCODER_ZIP)
            rc, _, err = run(["unzip", "-q", "-o", z, "-d", mdir], timeout=1800)
            if rc != 0 or not enc.exists():
                raise RuntimeError(f"unzip did not produce {enc.name}: {err[-600:]}")
        return info

    def transcribe(self, wins, raw, tag):
        cmd = [self.bin, "-m", self.model, "-ng", "-l", "en", "-nt", "-bs", "1", "-bo", "1", "-oj"]
        for w in wins:
            cmd += ["-f", w["path"]]
            j = Path(w["path"] + ".json")
            if j.exists():
                j.unlink()
        t = time.time()
        rc, out, err = run(cmd, timeout=7200)
        wall = time.time() - t
        (raw / f"whispercpp-{tag}.log").write_text(f"{cmd}\nrc={rc}\n--- stdout\n{out[-20000:]}\n--- stderr\n{err[-40000:]}")
        r = {"rc": rc, "wall_s": round(wall, 2), "hyps": {}, "t0": t, "t1": t + wall,
             "coreml_loaded": "Core ML model loaded" in err, "coreml_in_build": "COREML = 1" in err}
        if rc != 0:
            r["error"] = err[-600:]
        for w in wins:
            j = Path(w["path"] + ".json")
            if j.exists():
                d = json.loads(j.read_text())
                r["hyps"][w["id"]] = "".join(s.get("text", "") for s in d.get("transcription", []))
                j.rename(raw / f"whispercpp-{tag}-{w['id']}.json")
        ms = {m.group(1): float(m.group(2)) for m in
              re.finditer(r"whisper_print_timings:\s+(\w+) time =\s*([\d.]+) ms", err)}
        r["timings_ms"] = ms
        stages = [ms[k] for k in ("mel", "sample", "encode", "decode", "batchd", "prompt") if k in ms]
        r["engine_s"] = round(sum(stages) / 1e3, 3) if stages else None
        # "total" runs from context creation, so what is neither a stage nor the ggml load is
        # mostly the Core ML encoder load (plus audio reading)
        if "total" in ms and stages:
            r["load_s"] = round((ms["total"] - sum(stages)) / 1e3, 3)
            r["load"] = {"ggml_load_s": round(ms.get("load", 0) / 1e3, 3)}
        return r


# ---------------------------------------------------------------- mlx-whisper

MLX_WORKER = r'''
import json, sys, time, wave
import numpy as np
t = time.time()
import mlx_whisper
from huggingface_hub import snapshot_download
repo, wins, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
snapshot_download(repo)
res = {"import_s": round(time.time() - t, 3), "hyps": {}, "per_window_s": []}
for i, w in enumerate(wins):
    with wave.open(w["path"], "rb") as f:  # 16 kHz mono PCM16; a path would need ffmpeg
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    t = time.time()
    r = mlx_whisper.transcribe(pcm, path_or_hf_repo=repo, language="en", verbose=None)
    res["per_window_s"].append(round(time.time() - t, 3))
    res["hyps"][w["id"]] = r["text"]
json.dump(res, open(out, "w"))
'''


class Mlx:
    name = "mlx"

    def setup(self, args):
        self.py = common.ensure_venv("mlx-whisper", MLX_PINS)
        self.worker = common.WORK / "whisper" / "mlx_worker.py"
        self.worker.write_text(MLX_WORKER)
        log("mlx-whisper: fetching the model")
        rc, _, err = run([self.py, "-c", "import sys; from huggingface_hub import snapshot_download as d; "
                          "print(d(sys.argv[1]))", MLX_REPO], timeout=3600)
        if rc != 0:
            raise RuntimeError("model download failed: " + err[-800:])
        return {"model": MLX_REPO, "mlx_whisper": MLX_PINS[0],
                "versions": common.sh([self.py, "-m", "pip", "freeze"])}

    def transcribe(self, wins, raw, tag):
        wj, oj = raw / f"mlx-{tag}-in.json", raw / f"mlx-{tag}.json"
        # the first window is transcribed twice; the first pass carries the model load
        wj.write_text(json.dumps([wins[0]] + list(wins)))
        t = time.time()
        rc, out, err = run([self.py, self.worker, MLX_REPO, wj, oj], timeout=7200)
        wall = time.time() - t
        r = {"rc": rc, "wall_s": round(wall, 2), "hyps": {}, "t0": t, "t1": t + wall}
        if rc != 0 or not oj.exists():
            r["error"] = err[-600:]
            return r
        d = json.loads(oj.read_text())
        r["hyps"] = d["hyps"]
        per = d["per_window_s"]
        r["engine_s"] = round(sum(per[1:]), 3)
        r["load_s"] = round(per[0] - per[1], 3)  # first pass minus the same window warm
        r["load"] = {"first_call_s": per[0], "same_window_warm_s": per[1], "import_s": d["import_s"]}
        return r


ENGINES = {"whisperkit": WhisperKit, "whispercpp": WhisperCpp, "mlx": Mlx}


# ---------------------------------------------------------------- run

def measure(engine, wins, raw, cond, load):
    """One window alone (the load), then every window in one invocation."""
    if load is not None:
        load.wait_busy()
    one = engine.transcribe(wins[:1], raw, f"{cond}-one")
    if load is not None:
        load.wait_busy()
    full = engine.transcribe(wins, raw, f"{cond}-all")
    audio = sum(w["seconds"] for w in wins)
    r = {"audio_s": round(audio, 1), "windows": len(wins), "wall_s": full["wall_s"],
         "engine_s": full.get("engine_s"), "load_s": full.get("load_s"), "load": full.get("load"),
         "single_window_wall_s": one["wall_s"], "single_window_load_s": one.get("load_s"),
         "error": full.get("error") or one.get("error")}
    if full.get("engine_s"):
        r["rtf"] = round(full["engine_s"] / audio, 5)
        r["speed_x_realtime"] = round(audio / full["engine_s"], 1)
    if len(wins) > 1 and not r["error"]:
        # wall-clock cross-check: the extra windows' cost, with process start and load cancelled
        extra = (full["wall_s"] - one["wall_s"]) / (audio - wins[0]["seconds"])
        r["rtf_wall_marginal"] = round(extra, 5)
    for k in ("stages_s", "timings_ms", "coreml_loaded", "coreml_in_build"):
        if k in full:
            r[k] = full[k]
    r.update(score(wins, full["hyps"]))
    r.update(common.gpu_record(load, full["t0"], full["t1"]))
    r["hyps"] = full["hyps"]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=list(ENGINES), action="append")
    ap.add_argument("--windows", type=int, default=40, help="number of ~30 s windows")
    ap.add_argument("--window", type=float, default=29.0, help="longest window, seconds")
    ap.add_argument("--whisperkit-cli", help="path to whisperkit-cli (default: on PATH)")
    ap.add_argument("--whisperkit-units", default="cpuAndNeuralEngine",
                    choices=["cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly", "all"])
    ap.add_argument("--whispercpp-units", default="ne", choices=["ne", "all"],
                    help="ne: patch the Core ML encoder to CPU + Neural Engine; all: upstream")
    ap.add_argument("--cooldown", type=int, default=20, help="seconds between engines")
    common.add_gpu_args(ap)
    args = ap.parse_args()
    if sys.version_info < (3, 11):
        sys.exit("python3 >= 3.11 needed (whisper-normalizer); try `brew install python@3.12`")
    if not common.is_mac():
        sys.exit("This script needs Apple Silicon macOS.")
    common.reexec_in_venv("whisper", PINS)

    env, out = common.start("whisper", args)
    raw = out / "raw"
    S = {"env": env, "args": {k: str(v) for k, v in vars(args).items()}, "engines": {}}
    wins = prepare_audio(args)
    S["audio"] = {"windows": len(wins), "seconds": round(sum(w["seconds"] for w in wins), 1),
                  "source": LIBRISPEECH, "window_ids": [w["utterances"] for w in wins]}
    log(f"{len(wins)} windows, {S['audio']['seconds']:.0f} s of audio")

    engines = []
    for name in args.only or list(ENGINES):
        e = ENGINES[name]()
        try:
            S["engines"][name] = {"setup": e.setup(args)}
        except Exception as ex:  # noqa: BLE001 - one engine failing must not lose the others
            S["engines"][name] = {"error": repr(ex)}
            log(f"{name} setup failed: {ex!r}")
            continue
        log(f"{name}: first invocation (one window; compiles for the Neural Engine if never loaded)")
        first = e.transcribe(wins[:1], raw, "first")
        first.pop("hyps", None)
        S["engines"][name]["first_invocation"] = first
        engines.append(e)
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

    for cond, load in common.gpu_conditions(args, raw):
        if isinstance(load, Exception):
            S["busy_error"] = repr(load)
            log(f"GPU-busy condition skipped: {load!r}")
            continue
        for e in engines:
            log(f"{e.name}, GPU {cond}")
            try:
                r = measure(e, wins, raw, cond, load)
            except Exception as ex:  # noqa: BLE001
                r = {"error": repr(ex)}
            hyps = r.pop("hyps", None)
            if hyps is not None:
                (raw / f"{e.name}-{cond}-hyps.json").write_text(json.dumps(hyps, indent=1))
            S["engines"][e.name][cond] = r
            (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
            time.sleep(args.cooldown)
    common.finish(out, S, summarize(S))


def summarize(S):
    e, a = S["env"], S["audio"]
    L = [f"# Whisper large-v3-turbo: {e.get('chip')}, {e['timestamp']}", "",
         f"macOS {e.get('macos')}, AC {e.get('on_ac_power')}, low power {e.get('low_power_mode')}. "
         f"LibriSpeech test-clean, {a['windows']} windows, {a['seconds']} s of audio.", "",
         "RTF is processing seconds per audio second with the model load excluded (engine-reported; "
         "the wall column is the same from wall-clock, all windows minus one window). "
         "Load is one invocation's model load: first is this run's first invocation.", "",
         "| engine | GPU | WER % | RTF | x realtime | RTF wall | load s first | load s later | GPU tok/s alone -> during |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, E in S["engines"].items():
        if "error" in E:
            L.append(f"| {name} | | setup failed: {E['error'][:160]} | | | | | | |")
            continue
        first = (E.get("first_invocation") or {}).get("load_s")
        for cond in ("idle", "busy"):
            r = E.get(cond)
            if not r:
                continue
            if r.get("error") and "wer" not in r:
                L.append(f"| {name} | {cond} | failed: {r['error'][:160]} | | | | | | |")
                continue
            g = (f"{r.get('gpu_alone_tok_s')} -> {r.get('gpu_during_tok_s')} (busy {r.get('gpu_busy_fraction')})"
                 if cond == "busy" else "")
            L.append(f"| {name} | {cond} | {100 * r['wer']:.2f} | {r.get('rtf')} | {r.get('speed_x_realtime')} | "
                     f"{r.get('rtf_wall_marginal')} | {first} | {r.get('load_s')} | {g} |")
    L.append("")
    for name, E in S["engines"].items():
        s = E.get("setup") or {}
        if s:
            L.append(f"- {name}: " + ", ".join(f"{k} {v}" for k, v in s.items() if k != "versions"))
    if S.get("busy_error"):
        L += ["", f"GPU-busy condition skipped: {S['busy_error']}"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
