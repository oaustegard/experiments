#!/usr/bin/env python3
"""Benchmark Ternary Bonsai 2.5 27B on an Apple Silicon Mac, against its bandwidth ceiling.

Run on the Mac, from this directory, plugged in, not in Low Power Mode:

    python3 bench.py                  # everything, ~40-60 min, ~30 GB of downloads
    python3 bench.py --only llama     # llama.cpp phases only
    python3 bench.py --only mlx       # MLX phases only

Phases (each records its own errors and the rest carry on):
  env      chip, memory, macOS, power source and mode, GPU wired limit
  bw       MLX weight-read bandwidth probe (fp16 matvec and sum over 4 GiB)
  llama    PrismML llama.cpp fork, built from source: llama-bench on PQ2_0 and PTQ1_0
           - prefill 512 / 2048 / 8192 tokens
           - 128-token decode at depth 0 / 8192 / 32768 (KV and GDN-state share)
           - prefill ubatch sweep 256..2048 at 8192 tokens (where ms/token stops falling)
  mtp      llama-server on the PQ2_0-mtp file, the released Metal preset, MTP off vs on,
           six real prompts, greedy, plus one ~8K-token prefill prompt
  mlx      the prism-ml MLX pack (affine 2-bit) through its bundled loader: prefill and
           decode at 512 / 2048 / 8192 / 32768, and its MTP head off vs on

Output: results/<timestamp>-<chip>/ with env.json, raw/*.json|log, summary.json, summary.md.
Commit that directory, or paste summary.md back.

Requires: Xcode command line tools, cmake (brew install cmake), git, python3 >= 3.9.
The MLX phase makes its own venv in work/venv with the pack's pinned requirements.
"""

import argparse
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
GGUF_REPO = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
MLX_REPO = "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit"
FORK = "https://github.com/PrismML-Eng/llama.cpp"
GGUFS = {
    "PQ2_0": "Ternary-Bonsai-2.5-27B-PQ2_0.gguf",
    "PTQ1_0": "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf",
    "PQ2_0-mtp": "Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf",
}
# Released Metal preset (presets/llama.cpp/mtp-metal.json, build prism-b12128).
MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-min", "0",
            "--spec-draft-n-max", "1", "--spec-draft-p-min", "0.6"]
SERVER_ARGS = ["-ngl", "99", "-fa", "on", "-c", "16384", "-np", "1"]

# Architecture constants for the roofline (Qwen3.8-27B, from the pack's config.json).
FULL_ATTN_LAYERS, KV_HEADS, HEAD_DIM = 16, 4, 256
KV_BYTES_PER_TOKEN_F16 = FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM * 2 * 2  # K and V, fp16 = 64 KiB
GDN_STATE_BYTES = 48 * 48 * 128 * 128 * 4  # 48 layers x 48 heads x 128x128 fp32 = ~151 MB
EMBED_ROWS, HIDDEN = 248320, 5120  # one embedding row is read per token, not the table

PROMPTS = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into "
    "total seconds. Include type hints, a docstring and three doctest examples.",
    "You have tools get_weather(city: str) and send_email(to: str, subject: str, body: str). "
    "The user says: 'Check the weather in Oslo and Boston and email a comparison to "
    "kari@example.com.' Respond with the JSON tool calls you would make, in order, then a "
    "one-paragraph explanation.",
    "Explain how a write-ahead log lets a database recover after a crash. Cover checkpoints, "
    "redo versus undo, and what fsync guarantees. About 300 words.",
    "Refactor this into idiomatic Rust with error handling instead of unwrap:\n\n"
    "fn main() { let s = std::fs::read_to_string(\"in.txt\").unwrap(); let n: Vec<i64> = "
    "s.lines().map(|l| l.trim().parse().unwrap()).collect(); let t: i64 = n.iter().sum(); "
    "println!(\"{}\", t / n.len() as i64); }",
    "List twenty concrete, testable acceptance criteria for a password-reset flow in a web "
    "application, one line each.",
    "Translate into Norwegian (bokmål), keeping the tone: 'We moved the release to Thursday "
    "because the migration script failed on the staging copy. Nothing is lost; we just need "
    "one more dry run.' Then explain two word choices.",
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, timeout=None, env=None, cwd=None):
    """Run a command; return (returncode, stdout, stderr). Never raises on failure."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", repr(e)


def sh(cmd):
    rc, out, _ = run(cmd, timeout=60)
    return out.strip() if rc == 0 else None


# ---------------------------------------------------------------- env

def capture_env():
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": sh(["sysctl", "-n", "hw.model"]),
        "memsize_bytes": sh(["sysctl", "-n", "hw.memsize"]),
        "macos": sh(["sw_vers", "-productVersion"]),
        "iogpu_wired_limit_mb": sh(["sysctl", "-n", "iogpu.wired_limit_mb"]),
        "pmset_batt": sh(["pmset", "-g", "batt"]),
        "pmset_therm": sh(["pmset", "-g", "therm"]),
        "python": platform.python_version(),
    }
    custom = sh(["pmset", "-g"]) or ""
    env["low_power_mode"] = None
    for line in custom.splitlines():
        parts = line.split()
        if parts and parts[0] in ("lowpowermode", "powermode") and len(parts) > 1:
            env[parts[0]] = parts[1]
            if parts[0] == "lowpowermode":
                env["low_power_mode"] = parts[1] == "1"
            if parts[0] == "powermode":  # 0 auto, 1 low power, 2 high power
                env["low_power_mode"] = parts[1] == "1"
    gpu = sh(["system_profiler", "SPDisplaysDataType", "-json"])
    if gpu:
        try:
            cards = json.loads(gpu).get("SPDisplaysDataType", [])
            env["gpu_cores"] = cards[0].get("sppci_cores") if cards else None
        except (ValueError, IndexError, AttributeError):
            env["gpu_cores"] = None
    batt = env["pmset_batt"] or ""
    env["on_ac_power"] = ("AC Power" in batt) if batt else None
    return env


# ---------------------------------------------------------------- downloads

def venv_python():
    return WORK / "venv" / "bin" / "python"


def ensure_venv(pack_dir=None):
    py = venv_python()
    if not py.exists():
        log("creating venv in work/venv")
        rc, _, err = run([sys.executable, "-m", "venv", str(WORK / "venv")], timeout=300)
        if rc != 0:
            raise RuntimeError("venv failed: " + err[-2000:])
        run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout=600)
    rc, _, err = run([str(py), "-m", "pip", "install", "-q", "huggingface_hub>=0.30"],
                     timeout=900)
    if rc != 0:
        raise RuntimeError("pip huggingface_hub failed: " + err[-2000:])
    if pack_dir is not None:
        req = pack_dir / "runtime" / "requirements.txt"
        log("installing the MLX pack's pinned requirements")
        rc, _, err = run([str(py), "-m", "pip", "install", "-q", "-r", str(req)], timeout=1800)
        if rc != 0:
            raise RuntimeError("pip pack requirements failed: " + err[-2000:])
    return py


def hf_download(repo, filename=None, local_dir=None):
    """Download one file (or a whole repo snapshot) with huggingface_hub in the venv."""
    py = ensure_venv()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        code = ("import sys; from huggingface_hub import hf_hub_download as d; "
                "print(d(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))")
        args = [repo, filename, str(local_dir)]
    else:
        code = ("import sys; from huggingface_hub import snapshot_download as d; "
                "print(d(sys.argv[1], local_dir=sys.argv[2]))")
        args = [repo, str(local_dir)]
    log(f"downloading {repo} {filename or '(snapshot)'}")
    p = subprocess.run([str(py), "-c", code] + args, text=True, stdout=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"download failed: {repo} {filename}")
    return Path(p.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- llama.cpp

def build_fork():
    src = WORK / "llama.cpp"
    bindir = src / "build" / "bin"
    if (bindir / "llama-bench").exists() and (bindir / "llama-server").exists():
        return bindir, sh(["git", "-C", str(src), "rev-parse", "HEAD"])
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found: brew install cmake")
    if not src.exists():
        log("cloning the PrismML llama.cpp fork")
        rc, _, err = run(["git", "clone", "--depth", "1", FORK, str(src)], timeout=1800)
        if rc != 0:
            raise RuntimeError("clone failed: " + err[-2000:])
    log("building llama-bench and llama-server (Metal)")
    rc, _, err = run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                      "-DLLAMA_CURL=OFF", "-DGGML_METAL=ON"], cwd=str(src), timeout=900)
    if rc != 0:
        raise RuntimeError("cmake configure failed: " + err[-3000:])
    rc, _, err = run(["cmake", "--build", "build", "-j", "--config", "Release",
                      "--target", "llama-bench", "llama-server"], cwd=str(src), timeout=3600)
    if rc != 0:
        raise RuntimeError("build failed: " + err[-3000:])
    return bindir, sh(["git", "-C", str(src), "rev-parse", "HEAD"])


def llama_bench(bindir, model, extra, raw_path, timeout=5400):
    cmd = [str(bindir / "llama-bench"), "-m", str(model), "-ngl", "99", "-fa", "1",
           "-o", "json"] + extra
    log("llama-bench " + " ".join(extra))
    rc, out, err = run(cmd, timeout=timeout)
    raw_path.write_text(json.dumps({"cmd": cmd, "rc": rc, "stdout": out,
                                    "stderr_tail": err[-4000:]}, indent=1))
    if rc != 0:
        return {"error": f"rc={rc}: {err[-600:]}"}
    try:
        start = out.index("[")
        rows = json.loads(out[start:])
    except ValueError as e:
        return {"error": f"unparsable llama-bench output: {e}"}
    keep = ("n_prompt", "n_gen", "n_depth", "n_batch", "n_ubatch", "avg_ts", "stddev_ts",
            "avg_ns", "type_k", "build_commit")
    return {"rows": [{k: r.get(k) for k in keep} for r in rows]}


def phase_llama(out, raw, args):
    res = {}
    bindir, commit = build_fork()
    res["fork_commit"] = commit
    models = {}
    for key in ("PQ2_0", "PTQ1_0"):
        path = hf_download(GGUF_REPO, GGUFS[key], WORK / "gguf")
        models[key] = path
        res.setdefault("file_bytes", {})[key] = path.stat().st_size
    for key, path in models.items():
        r = {}
        r["prefill"] = llama_bench(bindir, path, ["-p", "512,2048,8192", "-n", "0",
                                                  "-r", str(args.reps)],
                                   raw / f"llama-{key}-prefill.json")
        r["decode"] = llama_bench(bindir, path, ["-p", "0", "-n", "128",
                                                 "-d", args.depths, "-r", "2"],
                                  raw / f"llama-{key}-decode.json")
        if key == "PQ2_0" or args.ubatch_all:
            r["ubatch_sweep"] = llama_bench(
                bindir, path, ["-p", "8192", "-n", "0", "-b", "2048",
                               "-ub", "256,512,1024,2048", "-r", "2"],
                raw / f"llama-{key}-ubatch.json")
        res[key] = r
        time.sleep(args.cooldown)
    return res


# ---------------------------------------------------------------- llama-server MTP A/B

def http_json(url, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def long_prompt(target_tokens):
    """About target_tokens of real code: this script repeated, then a short task."""
    text = Path(__file__).read_text()
    reps = max(1, int(target_tokens * 3.2 / len(text)) + 1)  # ~3.2 chars/token for code
    body = (text + "\n") * reps
    body = body[: int(target_tokens * 3.2)]
    return body + "\n\nIn three sentences, say what the script above measures."


def serve_once(bindir, model, port, spec, raw_path, args):
    cmd = [str(bindir / "llama-server"), "-m", str(model), "--port", str(port),
           "--host", "127.0.0.1"] + SERVER_ARGS + (MTP_ARGS if spec else [])
    logf = open(str(raw_path) + ".log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    results = {"cmd": cmd, "prompts": []}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                return {"error": f"server exited rc={proc.returncode}, see {raw_path}.log"}
            try:
                if http_json(base + "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() - t0 > 900:
                return {"error": "server did not become healthy in 900 s"}
            time.sleep(2)
        work = [("p%d" % i, p, 512) for i, p in enumerate(PROMPTS)]
        work.append(("long8k", long_prompt(8192), 64))
        for rnd in range(args.mtp_rounds):
            for name, prompt, n in work:
                body = {"messages": [{"role": "user", "content": prompt}],
                        "max_tokens": n, "temperature": 0, "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    t = time.time()
                    r = http_json(base + "/v1/chat/completions", body)
                    tim = r.get("timings", {})
                    results["prompts"].append({
                        "round": rnd, "name": name, "wall_s": round(time.time() - t, 3),
                        "prompt_n": tim.get("prompt_n"),
                        "prompt_per_second": tim.get("prompt_per_second"),
                        "predicted_n": tim.get("predicted_n"),
                        "predicted_per_second": tim.get("predicted_per_second"),
                        "draft_n": tim.get("draft_n"),
                        "draft_n_accepted": tim.get("draft_n_accepted"),
                        "text_head": (r.get("choices") or [{}])[0]
                        .get("message", {}).get("content", "")[:200],
                    })
                except (urllib.error.URLError, OSError, ValueError) as e:
                    results["prompts"].append({"round": rnd, "name": name,
                                               "error": repr(e)})
        return results
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        raw_path.write_text(json.dumps(results, indent=1))


def summarize_server(r):
    if "error" in r:
        return r
    gen = [p["predicted_per_second"] for p in r["prompts"]
           if p.get("name", "").startswith("p") and p.get("predicted_per_second")]
    pre = [p["prompt_per_second"] for p in r["prompts"]
           if p.get("name") == "long8k" and p.get("prompt_per_second")]
    dn = sum(p.get("draft_n") or 0 for p in r["prompts"])
    da = sum(p.get("draft_n_accepted") or 0 for p in r["prompts"])
    return {
        "decode_tok_s_median": round(statistics.median(gen), 2) if gen else None,
        "prefill_8k_tok_s_median": round(statistics.median(pre), 1) if pre else None,
        "draft_acceptance": round(da / dn, 3) if dn else None,
        "errors": sum(1 for p in r["prompts"] if "error" in p),
    }


def phase_mtp(out, raw, args):
    bindir, commit = build_fork()
    model = hf_download(GGUF_REPO, GGUFS["PQ2_0-mtp"], WORK / "gguf")
    res = {"fork_commit": commit, "file_bytes": model.stat().st_size}
    for label, spec in (("mtp_off", False), ("mtp_on", True)):
        log(f"llama-server {label}")
        r = serve_once(bindir, model, args.port, spec, raw / f"server-{label}.json", args)
        res[label] = summarize_server(r)
        time.sleep(args.cooldown)
    a, b = res["mtp_off"].get("decode_tok_s_median"), res["mtp_on"].get("decode_tok_s_median")
    res["mtp_decode_ratio"] = round(b / a, 3) if a and b else None
    return res


# ---------------------------------------------------------------- MLX (runs in the venv)

MLX_WORKER = r'''
import json, sys, time, statistics
from pathlib import Path
pack = Path(sys.argv[1]); out = Path(sys.argv[2]); spec = json.loads(sys.argv[3])
res = {}
import mlx.core as mx
res["mlx_version"] = mx.__version__
res["metal_device"] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                       for k, v in mx.metal.device_info().items()} if hasattr(mx, "metal") else None

def save():
    out.write_text(json.dumps(res, indent=1))

# Bandwidth probe: the read pattern of decode is a matvec over the weights.
if spec.get("bw"):
    bw = {}
    try:
        W = mx.random.normal((32768, 65536)).astype(mx.float16)   # 4 GiB
        x = mx.random.normal((65536, 1)).astype(mx.float16)
        mx.eval(W, x)
        for name, fn in (("matvec_f16", lambda: W @ x), ("sum_f16", lambda: W.sum(axis=1))):
            for _ in range(3):
                mx.eval(fn())
            ts = []
            for _ in range(15):
                t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
            bw[name + "_GBps_best"] = round(W.nbytes / min(ts) / 1e9, 1)
            bw[name + "_GBps_median"] = round(W.nbytes / statistics.median(ts) / 1e9, 1)
        del W, x
    except Exception as e:
        bw["error"] = repr(e)
    res["bw"] = bw; save()

if not spec.get("model"):
    sys.exit(0)

sys.path.insert(0, str(pack / "runtime"))
try:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from vision_artifact import chat_config, load_vl_model
    t = time.perf_counter()
    model, processor, config = load_vl_model(pack)
    res["load_s"] = round(time.perf_counter() - t, 1)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
except Exception as e:
    res["load_error"] = repr(e); save(); sys.exit(0)

filler = Path(spec["filler"]).read_text()

def prompt_of(n, task):
    ids = tok.encode(filler * (n // 1000 + 2))
    text = tok.decode(ids[: max(1, n - 40)])
    return apply_chat_template(processor, chat_config(config), text + "\n\n" + task,
                               num_images=0, enable_thinking=False)

def one(prompt, max_tokens, **kw):
    t = time.perf_counter()
    r = generate(model, processor, prompt, max_tokens=max_tokens, temperature=0.0,
                 verbose=False, **kw)
    wall = time.perf_counter() - t
    g = lambda k: getattr(r, k, None)
    return {"wall_s": round(wall, 2), "prompt_tokens": g("prompt_tokens"),
            "prompt_tps": g("prompt_tps"), "generation_tokens": g("generation_tokens"),
            "generation_tps": g("generation_tps"), "peak_memory_gb": g("peak_memory")}

TASK = "Rewrite the code above in full, adding a comment above every function."
try:
    one(prompt_of(256, TASK), 16)   # warmup: compile kernels
except Exception as e:
    res["warmup_error"] = repr(e)
runs = {}
for n in spec["sizes"]:
    rows = []
    for rep in range(spec["reps"]):
        try:
            rows.append(one(prompt_of(n, TASK), 128))
        except Exception as e:
            rows.append({"error": repr(e)})
    runs[str(n)] = rows
    res["runs"] = runs; save()

if spec.get("mtp"):
    try:
        from mtp_artifact import load_mtp_head
        draft = load_mtp_head(model, pack / "mtp" / "mtp-v2-head.safetensors")
        p = prompt_of(1024, TASK)
        ab = {"off": [], "on": []}
        for rep in range(spec["reps"]):
            ab["off"].append(one(p, 256))
            ab["on"].append(one(p, 256, draft_model=draft, draft_kind="mtp",
                                draft_block_size=2))
        res["mtp"] = ab
    except Exception as e:
        res["mtp_error"] = repr(e)
    save()
'''


def phase_mlx(out, raw, args, bw_only=False):
    pack = None
    if not bw_only:
        pack = hf_download(MLX_REPO, None, WORK / "mlx-pack")
    py = ensure_venv(pack)
    if bw_only:
        # same MLX the pack pins, so the probe and the model runs share a runtime
        run([str(py), "-m", "pip", "install", "-q", "mlx==0.32.0"], timeout=900)
    worker = WORK / "mlx_worker.py"
    worker.write_text(MLX_WORKER)
    dest = raw / ("mlx-bw.json" if bw_only else "mlx.json")
    spec = {"bw": bw_only, "model": not bw_only,
            "sizes": [int(s) for s in args.mlx_sizes.split(",")],
            "reps": args.reps, "mtp": not args.skip_mtp, "filler": str(Path(__file__))}
    log("MLX worker: " + ("bandwidth probe" if bw_only else "model runs"))
    p = subprocess.run([str(py), str(worker), str(pack or WORK), str(dest), json.dumps(spec)],
                       text=True, stderr=subprocess.PIPE, timeout=7200)
    (raw / (dest.stem + ".stderr.log")).write_text(p.stderr[-20000:])
    if not dest.exists():
        return {"error": f"worker rc={p.returncode}: {p.stderr[-800:]}"}
    return json.loads(dest.read_text())


# ---------------------------------------------------------------- summary

def per_token_bytes(file_bytes, depth=0, kv_bytes=KV_BYTES_PER_TOKEN_F16):
    """Bytes decode reads per token: the whole GGUF file (an overcount, since only one
    embedding row is read per token, so the ceiling comes out slightly low), the GDN state
    read and written, and the attention KV at this depth."""
    return file_bytes + 2 * GDN_STATE_BYTES + depth * kv_bytes


def best(rows, **match):
    for r in rows or []:
        if all(r.get(k) == v for k, v in match.items()):
            return r.get("avg_ts")
    return None


def summarize(S):
    lines = [f"# Bonsai 2.5 27B on {S['env'].get('chip')} — {S['env']['timestamp']}", ""]
    e = S["env"]
    lines += [f"- memory {int(e['memsize_bytes'] or 0) / 2**30:.0f} GiB, GPU cores "
              f"{e.get('gpu_cores')}, macOS {e.get('macos')}, AC power {e.get('on_ac_power')}, "
              f"low power mode {e.get('low_power_mode')}, wired limit "
              f"{e.get('iogpu_wired_limit_mb')} MB", ""]
    bw = (S.get("bw") or {}).get("bw") or {}
    gbps = max([v for k, v in bw.items() if k.endswith("_best") and isinstance(v, (int, float))]
               or [0]) or None
    lines.append(f"Bandwidth probe (best of matvec/sum over 4 GiB): **{gbps} GB/s**")
    lines.append("")
    L = S.get("llama") or {}
    if "error" in L:
        lines.append(f"llama.cpp phase failed: {L['error']}")
    elif L:
        lines.append(f"## llama.cpp (PrismML fork {str(L.get('fork_commit'))[:10]})")
        lines.append("")
        lines.append("| packing | file GiB | pp512 | pp2048 | pp8192 | tg128 @0 | @8192 | "
                     "@32768 | ceiling @0 | % of ceiling @0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key in ("PQ2_0", "PTQ1_0"):
            r = L.get(key) or {}
            fb = (L.get("file_bytes") or {}).get(key)
            pre = (r.get("prefill") or {}).get("rows")
            dec = (r.get("decode") or {}).get("rows")
            pp = [best(pre, n_prompt=n) for n in (512, 2048, 8192)]
            tg = [best(dec, n_gen=128, n_depth=d) for d in (0, 8192, 32768)]
            ceil = round(gbps * 1e9 / per_token_bytes(fb), 1) if gbps and fb else None
            frac = f"{100 * tg[0] / ceil:.0f}%" if ceil and tg[0] else "–"
            fmt = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "–"
            lines.append(f"| {key} | {fb / 2**30:.2f} | " if fb else f"| {key} | – | ")
            lines[-1] += " | ".join(fmt(v) for v in pp + tg) + f" | {fmt(ceil)} | {frac} |"
            S.setdefault("derived", {})[key] = {"pp": pp, "tg": tg, "ceiling_tg0": ceil}
        sweep = ((L.get("PQ2_0") or {}).get("ubatch_sweep") or {}).get("rows")
        if sweep:
            lines.append("")
            lines.append("Prefill ubatch sweep, PQ2_0, 8192-token prompt: " + ", ".join(
                f"ub {r['n_ubatch']}: {r['avg_ts']:.0f} tok/s" for r in sweep if r.get("avg_ts")))
        lines.append("")
        lines.append("Decode ceiling = bandwidth / (file bytes + GDN state read+write + "
                     "64 KiB x depth of fp16 KV). Using the whole file overstates bytes "
                     "slightly (the embedding table is not read per token).")
        lines.append("")
    M = S.get("mtp") or {}
    if "error" in M:
        lines.append(f"MTP phase failed: {M['error']}")
    elif M:
        lines.append("## llama-server MTP A/B (PQ2_0-mtp, released Metal preset, greedy)")
        lines.append("")
        lines.append("| | decode tok/s (median, 6 prompts) | prefill tok/s (8K prompt) | "
                     "draft acceptance |")
        lines.append("|---|---:|---:|---:|")
        for k in ("mtp_off", "mtp_on"):
            r = M.get(k) or {}
            lines.append(f"| {k} | {r.get('decode_tok_s_median')} | "
                         f"{r.get('prefill_8k_tok_s_median')} | {r.get('draft_acceptance')} |")
        lines.append(f"\nMTP decode ratio: **{M.get('mtp_decode_ratio')}**\n")
    X = S.get("mlx") or {}
    if X:
        lines.append(f"## MLX pack (mlx {X.get('mlx_version')})")
        lines.append("")
        if X.get("load_error") or X.get("error"):
            lines.append(f"failed: {X.get('load_error') or X.get('error')}")
        for n, rows in (X.get("runs") or {}).items():
            ok = [r for r in rows if "error" not in r]
            if ok:
                pt = max(r["prompt_tps"] or 0 for r in ok)
                gt = max(r["generation_tps"] or 0 for r in ok)
                pk = max(r["peak_memory_gb"] or 0 for r in ok)
                lines.append(f"- {n}-token prompt: prefill {pt:.0f} tok/s, decode {gt:.1f} "
                             f"tok/s, peak {pk:.1f} GB")
            else:
                lines.append(f"- {n}-token prompt: failed ({rows[0].get('error', '')[:200]})")
        mt = X.get("mtp")
        if mt:
            off = statistics.median([r["generation_tps"] for r in mt["off"]])
            on = statistics.median([r["generation_tps"] for r in mt["on"]])
            lines.append(f"- MTP head (draft block 2): decode {off:.1f} -> {on:.1f} tok/s "
                         f"({on / off:.2f}x)")
        if X.get("mtp_error"):
            lines.append(f"- MTP failed: {X['mtp_error'][:300]}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["env", "bw", "llama", "mtp", "mlx"], action="append",
                    help="run only these phases (repeatable); default all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depths", default="0,8192,32768", help="llama-bench decode depths")
    ap.add_argument("--mlx-sizes", default="512,2048,8192,32768")
    ap.add_argument("--mtp-rounds", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between heavy phases")
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--skip-mtp", action="store_true", help="skip the MLX MTP A/B")
    ap.add_argument("--ubatch-all", action="store_true", help="ubatch sweep on PTQ1_0 too")
    ap.add_argument("--force", action="store_true",
                    help="run on battery or in Low Power Mode (results are not comparable)")
    args = ap.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit("This benchmark is for Apple Silicon macOS.")
    phases = args.only or ["env", "bw", "llama", "mtp", "mlx"]
    if "env" not in phases:
        phases = ["env"] + phases

    S = {"env": capture_env(), "args": vars(args)}
    e = S["env"]
    log(f"{e['chip']}, {int(e['memsize_bytes'] or 0) / 2**30:.0f} GiB, GPU cores "
        f"{e.get('gpu_cores')}, AC={e['on_ac_power']}, low power={e['low_power_mode']}")
    if (e["on_ac_power"] is False or e["low_power_mode"]) and not args.force:
        sys.exit("On battery or in Low Power Mode, which halves GPU throughput. Plug in and "
                 "set Energy Mode to Automatic or High Power, or pass --force.")

    tag = f"{datetime.now():%Y%m%d-%H%M}-{(e['chip'] or 'mac').replace(' ', '-')}"
    out = HERE / "results" / tag
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    (out / "env.json").write_text(json.dumps(e, indent=1))

    def guarded(name, fn):
        t = time.time()
        try:
            S[name] = fn()
        except Exception as ex:  # one phase failing must not lose the others
            S[name] = {"error": repr(ex)}
            log(f"{name} failed: {ex!r}")
        S.setdefault("wall_s", {})[name] = round(time.time() - t)
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

    if "bw" in phases:
        guarded("bw", lambda: phase_mlx(out, raw, args, bw_only=True))
    if "llama" in phases:
        guarded("llama", lambda: phase_llama(out, raw, args))
    if "mtp" in phases:
        guarded("mtp", lambda: phase_mtp(out, raw, args))
    if "mlx" in phases:
        guarded("mlx", lambda: phase_mlx(out, raw, args))

    S["env_after"] = {"pmset_therm": sh(["pmset", "-g", "therm"])}
    md = summarize(S)
    (out / "summary.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n" + md)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
