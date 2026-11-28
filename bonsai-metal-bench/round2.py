#!/usr/bin/env python3
"""Round 2 tasks for bonsai-metal-bench (see PLAN.md): three runs on the Mac.

  python3 round2.py tensor   Task 2: prefill with the Metal tensor API on and off, fork and upstream
  python3 round2.py mtp      Task 3: where the MTP gain goes (draft length, cut-off, kernel env vars)
  python3 round2.py kv       Task 4: decode at depth with an fp16 and a q8_0 KV cache

Each writes results/<timestamp>-round2-<task>-<chip>/ with summary.md, summary.json, env.json and
raw/ (every llama-bench stdout and its full stderr). It reuses bench.py's helpers, so it needs the
same Mac, the fork build and the GGUFs that `bench.py` put under work/. Stdlib only.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402  (stdlib helpers: run, log, capture_env, build_fork, hf_download, ...)

HERE, WORK = bench.HERE, bench.WORK
UPSTREAM = "https://github.com/ggml-org/llama.cpp"
PARAMS = 24.35e9          # matmul parameters per token, PLAN.md Task 2: FLOPs = 2 x PARAMS per token
log, run = bench.log, bench.run


def tflops(tok_s):
    return 2 * PARAMS * tok_s / 1e12


def start(task, args):
    env = bench.capture_env()
    log(f"{env['chip']}, AC={env['on_ac_power']}, low power={env['low_power_mode']}")
    if (env["on_ac_power"] is False or env["low_power_mode"]) and not args.force:
        sys.exit("On battery or in Low Power Mode. Plug in, set Energy Mode to Automatic, or pass --force.")
    tag = f"{datetime.now():%Y%m%d-%H%M}-round2-{task}-{(env['chip'] or 'mac').replace(' ', '-')}"
    out = HERE / "results" / tag
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(env, indent=1))
    WORK.mkdir(exist_ok=True)
    return env, out


def bench_run(bindir, model, extra, name, raw, env_extra=None, cooldown=20, timeout=7200):
    """One llama-bench invocation. Keeps stdout json and the full stderr in raw/. Returns rows."""
    cmd = [str(bindir / "llama-bench"), "-m", str(model), "-ngl", "99", "-fa", "1", "-o", "json"] + extra
    env = dict(os.environ)
    env.update(env_extra or {})
    log(f"{name}: " + " ".join(f"{k}={v}" for k, v in (env_extra or {}).items()) + " " + " ".join(extra))
    t = time.time()
    rc, out, err = run(cmd, timeout=timeout, env=env)
    (raw / f"{name}.stderr.log").write_text(err)
    (raw / f"{name}.json").write_text(json.dumps({"cmd": cmd, "env": env_extra or {}, "rc": rc,
                                                   "wall_s": round(time.time() - t, 1), "stdout": out}, indent=1))
    time.sleep(cooldown)
    if rc != 0:
        return {"error": f"rc={rc}: {err[-500:]}"}
    try:
        rows = json.loads(out[out.index("["):])
    except ValueError as e:
        return {"error": f"unparsable: {e}"}
    keep = ("n_prompt", "n_gen", "n_depth", "n_batch", "n_ubatch", "type_k", "type_v", "avg_ts",
            "stddev_ts", "build_commit")
    return {"rows": [{k: r.get(k) for k in keep} for r in rows],
            "stderr_notes": stderr_notes(err)}


def stderr_notes(err):
    """The matmul kernels a run loaded, plus any line about the tensor API or its metallib."""
    kernels = sorted(set(re.findall(r"loaded (kernel_mul_mm\w*|kernel_mul_mat\w*)", err)))
    tensor = []
    for line in err.splitlines():
        if re.search(r"tensor|metallib", line, re.I) and line not in tensor:
            tensor.append(line.strip())
    return {"mul_mm_kernels": kernels, "tensor_lines": tensor[:12]}


# ---------------------------------------------------------------- Task 2: tensor API on and off

def build_upstream(tag):
    src = WORK / "llama.cpp-upstream"
    bindir = src / "build" / "bin"
    if not (bindir / "llama-bench").exists():
        if not src.exists():
            log(f"cloning upstream llama.cpp {tag}")
            rc, _, err = run(["git", "clone", "--depth", "1", "--branch", tag, UPSTREAM, str(src)], timeout=1800)
            if rc != 0:
                raise RuntimeError("clone failed: " + err[-1500:])
        log("building upstream llama-bench (Metal)")
        rc, _, err = run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
                          "-DGGML_METAL=ON"], cwd=str(src), timeout=900)
        if rc != 0:
            raise RuntimeError("cmake failed: " + err[-2500:])
        rc, _, err = run(["cmake", "--build", "build", "-j", "--config", "Release", "--target", "llama-bench"],
                         cwd=str(src), timeout=3600)
        if rc != 0:
            raise RuntimeError("build failed: " + err[-2500:])
    return bindir, bench.sh(["git", "-C", str(src), "rev-parse", "HEAD"])


def task_tensor(args):
    env, out = start("tensor", args)
    raw = out / "raw"
    S = {"env": env, "args": vars(args), "params_per_token": PARAMS}
    fork_bin, fork_commit = bench.build_fork()
    S["fork_commit"] = fork_commit
    models = {k: bench.hf_download(bench.GGUF_REPO, bench.GGUFS[k], WORK / "gguf") for k in ("PQ2_0", "PTQ1_0")}
    q2 = bench.hf_download(bench.GGUF_REPO, "Ternary-Bonsai-2.5-27B-Q2_0.gguf", WORK / "gguf")
    plans = [("fork", fork_bin, "PQ2_0", models["PQ2_0"], True), ("fork", fork_bin, "PTQ1_0", models["PTQ1_0"], False)]
    try:
        up_bin, up_commit = build_upstream(args.upstream_tag)
        S["upstream"] = {"tag": args.upstream_tag, "commit": up_commit}
        plans.append(("upstream", up_bin, "Q2_0", q2, False))
    except Exception as ex:  # the comparison point failing must not lose the fork runs
        S["upstream"] = {"error": repr(ex)}
        log(f"upstream build failed: {ex!r}")
    S["runs"] = {}
    for build, bindir, key, model, sweep in plans:
        for mode, envx in (("tensor_on", {}), ("tensor_off", {"GGML_METAL_TENSOR_DISABLE": "1"})):
            name = f"{build}-{key}-{mode}"
            r = {"prefill": bench_run(bindir, model, ["-p", "512,2048,8192", "-n", "0", "-r", str(args.reps)],
                                      name + "-prefill", raw, envx)}
            if sweep:
                r["ubatch"] = bench_run(bindir, model, ["-p", "8192", "-n", "0", "-b", "2048",
                                                        "-ub", "256,512,1024,2048", "-r", "2"],
                                        name + "-ubatch", raw, envx)
            S["runs"][name] = r
            (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    md = summarize_tensor(S)
    (out / "summary.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n" + md)
    log(f"wrote {out}")


def summarize_tensor(S):
    L = ["# Task 2: prefill with the Metal tensor API on and off", "",
         f"Apple chip: {S['env']['chip']}. Fork commit {S['fork_commit'][:10]}; upstream "
         f"{json.dumps(S.get('upstream'))}. TFLOPS = 2 x {PARAMS / 1e9:.2f}e9 x tok/s.", "",
         "| build | file | tensor API | pp512 tok/s (TFLOPS) | pp2048 | pp8192 |", "|---|---|---|---|---|---|"]
    for name, r in S["runs"].items():
        build, key, mode = name.split("-", 2)
        rows = r["prefill"].get("rows") if isinstance(r["prefill"], dict) else None
        if not rows:
            L.append(f"| {build} | {key} | {mode} | {r['prefill']} | | |")
            continue
        by = {x["n_prompt"]: x["avg_ts"] for x in rows}
        cell = lambda n: f"{by[n]:.1f} ({tflops(by[n]):.1f})" if n in by else "n/a"  # noqa: E731
        L.append(f"| {build} | {key} | {mode.replace('tensor_', '')} | {cell(512)} | {cell(2048)} | {cell(8192)} |")
    L += ["", "Prefill ratio, tensor on / off:", ""]
    for build, key in (("fork", "PQ2_0"), ("fork", "PTQ1_0"), ("upstream", "Q2_0")):
        a, b = S["runs"].get(f"{build}-{key}-tensor_on"), S["runs"].get(f"{build}-{key}-tensor_off")
        try:
            on = {x["n_prompt"]: x["avg_ts"] for x in a["prefill"]["rows"]}
            off = {x["n_prompt"]: x["avg_ts"] for x in b["prefill"]["rows"]}
            L.append(f"- {build} {key}: " + ", ".join(f"pp{n} {on[n] / off[n]:.2f}x" for n in sorted(on)))
        except (KeyError, TypeError):
            L.append(f"- {build} {key}: missing")
    L += ["", "Ubatch sweep, fork PQ2_0, 8192-token prompt (tok/s):", "", "| tensor API | ub 256 | ub 512 | ub 1024 | ub 2048 |",
          "|---|---|---|---|---|"]
    for mode in ("tensor_on", "tensor_off"):
        try:
            rows = S["runs"][f"fork-PQ2_0-{mode}"]["ubatch"]["rows"]
            L.append(f"| {mode.replace('tensor_', '')} | " + " | ".join(f"{x['avg_ts']:.1f}" for x in rows) + " |")
        except (KeyError, TypeError):
            L.append(f"| {mode} | missing |")
    L += ["", "Kernels and tensor-API lines reported on stderr:", ""]
    for name, r in S["runs"].items():
        n = (r["prefill"] or {}).get("stderr_notes") if isinstance(r["prefill"], dict) else None
        if n:
            L.append(f"- `{name}`: {', '.join(n['mul_mm_kernels']) or 'no mul_mm kernel lines'}; "
                     f"tensor lines: {n['tensor_lines'][:4]}")
    return "\n".join(L)


def task_kernels(args):
    """Task 2, second pass: `llama-bench -v` prints each pipeline as it is compiled, which the default
    log level does not. One short prefill per configuration; keeps the stderr and lists the matmul
    pipelines (`kernel_mul_mm*`, `kernel_mul_mv*`) it asks for. Adds kernels.json to --into."""
    out = args.into
    raw = out / "raw"
    S = json.loads((out / "summary.json").read_text())
    fork_bin, _ = bench.build_fork()
    up_bin, _ = build_upstream(args.upstream_tag)
    models = {"PQ2_0": bench.hf_download(bench.GGUF_REPO, bench.GGUFS["PQ2_0"], WORK / "gguf"),
              "PTQ1_0": bench.hf_download(bench.GGUF_REPO, bench.GGUFS["PTQ1_0"], WORK / "gguf"),
              "Q2_0": bench.hf_download(bench.GGUF_REPO, "Ternary-Bonsai-2.5-27B-Q2_0.gguf", WORK / "gguf")}
    K = {}
    for build, bindir, key in (("fork", fork_bin, "PQ2_0"), ("fork", fork_bin, "PTQ1_0"), ("upstream", up_bin, "Q2_0")):
        for mode, envx in (("tensor_on", {}), ("tensor_off", {"GGML_METAL_TENSOR_DISABLE": "1"})):
            name = f"{build}-{key}-{mode}-kernels"
            cmd = [str(bindir / "llama-bench"), "-m", str(models[key]), "-ngl", "99", "-fa", "1", "-p", "512",
                   "-n", "0", "-r", "1", "-v", "-o", "json"]
            env = dict(os.environ)
            env.update(envx)
            log(name)
            rc, o, err = run(cmd, timeout=1800, env=env)
            (raw / f"{name}.stderr.log").write_text(err)
            names = re.findall(r"base = '(kernel_[A-Za-z0-9_]+)'", err)
            pipes = sorted({n for n in names if re.match(r"kernel_mul_m[mv]", n)})
            K[f"{build}-{key}-{mode}"] = {"rc": rc, "matmul_pipelines": pipes,
                                          "libs": sorted(set(re.findall(r"compiled '(\w*mul_mm\w*)' library", err)))}
            time.sleep(10)
    S["kernels"] = K
    (out / "kernels.json").write_text(json.dumps(K, indent=1))
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    md = (out / "summary.md").read_text().split("\nKernels and tensor-API lines")[0].rstrip() + \
        "\n\nMatmul pipelines requested (`llama-bench -v`, one 512-token prefill each; `kernels.json`):\n\n"
    for k, v in K.items():
        md += f"- `{k}`: {', '.join(v['matmul_pipelines']) or 'none printed'}\n"
    (out / "summary.md").write_text(md)
    print(md)


# ---------------------------------------------------------------- Task 3: MTP verify cost

def serve_setting(bindir, model, port, spec_args, envx, name, raw, rounds):
    cmd = [str(bindir / "llama-server"), "-m", str(model), "--port", str(port), "--host", "127.0.0.1"] \
        + bench.SERVER_ARGS + spec_args
    env = dict(os.environ)
    env.update(envx)
    logf = open(str(raw / f"{name}.server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{port}"
    res = {"cmd": cmd, "env": envx, "prompts": []}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                return {"error": f"server exited rc={proc.returncode}"}
            try:
                if bench.http_json(base + "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() - t0 > 900:
                return {"error": "server not healthy in 900 s"}
            time.sleep(2)
        work = [("p%d" % i, p, 512) for i, p in enumerate(bench.PROMPTS)]
        work.append(("long8k", bench.long_prompt(8192), 64))
        for rnd in range(rounds):
            for pname, prompt, n in work:
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0,
                        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    r = bench.http_json(base + "/v1/chat/completions", body)
                    tm = r.get("timings", {})
                    res["prompts"].append({"round": rnd, "name": pname, **{k: tm.get(k) for k in (
                        "prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n",
                        "draft_n_accepted")},
                        "text_head": (r.get("choices") or [{}])[0].get("message", {}).get("content", "")[:120]})
                except (urllib.error.URLError, OSError, ValueError) as e:
                    res["prompts"].append({"round": rnd, "name": pname, "error": repr(e)})
        return res
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        (raw / f"{name}.json").write_text(json.dumps(res, indent=1))


def summarize_setting(r):
    if "error" in r:
        return r
    gen = [p for p in r["prompts"] if p["name"].startswith("p") and p.get("predicted_per_second")]
    pre = [p["prompt_per_second"] for p in r["prompts"] if p["name"] == "long8k" and p.get("prompt_per_second")]
    tokens = sum(p["predicted_n"] or 0 for p in gen)
    drafted = sum(p["draft_n"] or 0 for p in gen)
    accepted = sum(p["draft_n_accepted"] or 0 for p in gen)
    steps = tokens - accepted                      # each verify step yields one token plus its accepted drafts
    return {"decode_tok_s_median": round(statistics.median(p["predicted_per_second"] for p in gen), 2),
            "prefill_8k_tok_s_median": round(statistics.median(pre), 1) if pre else None,
            "tokens": tokens, "draft_n": drafted, "draft_n_accepted": accepted,
            "acceptance": round(accepted / drafted, 3) if drafted else None,
            "tokens_per_step": round(tokens / steps, 3) if steps else None,
            "errors": sum(1 for p in r["prompts"] if "error" in p)}


def fork_has(var):
    rc, o, _ = run(["git", "-C", str(WORK / "llama.cpp"), "grep", "-l", var], timeout=60)
    return rc == 0 and bool(o.strip())


def task_mtp(args):
    env, out = start("mtp", args)
    raw = out / "raw"
    S = {"env": env, "args": vars(args)}
    bindir, commit = bench.build_fork()
    model = bench.hf_download(bench.GGUF_REPO, bench.GGUFS["PQ2_0-mtp"], WORK / "gguf")
    S["fork_commit"] = commit
    kernel_vars = ["GGML_METAL_PQ2_COEF", "LLAMA_MTP_PREFILL_LEAN", "GGML_METAL_SSM_CONV_ROWS"]
    S["env_vars_in_fork"] = {v: fork_has(v) for v in kernel_vars}
    present = [v for v in kernel_vars if S["env_vars_in_fork"][v]]

    def spec(nmax, pmin):
        return ["--spec-type", "draft-mtp", "--spec-draft-n-min", "0", "--spec-draft-n-max", str(nmax),
                "--spec-draft-p-min", str(pmin)]

    settings = [("off", [], {})]
    for pmin in (0.6, 0.3):
        for nmax in (1, 2, 3):
            settings.append((f"n{nmax}-p{pmin}", spec(nmax, pmin), {}))
    for v in present:
        settings.append((f"n1-p0.6+{v}", spec(1, 0.6), {v: "1"}))
    if len(present) > 1:
        settings.append(("n1-p0.6+all", spec(1, 0.6), {v: "1" for v in present}))
    settings.append(("off-again", [], {}))          # drift check on the baseline
    S["settings"] = {}
    for name, sa, envx in settings:
        log(f"llama-server {name}")
        r = bench_server = serve_setting(bindir, model, args.port, sa, envx, name, raw, args.rounds)
        S["settings"][name] = {"spec_args": sa, "env": envx, **summarize_setting(r)}
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
        time.sleep(args.cooldown)
    md = summarize_mtp(S)
    (out / "summary.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n" + md)
    log(f"wrote {out}")


def summarize_mtp(S):
    st = S["settings"]
    offs = [st[k]["decode_tok_s_median"] for k in ("off", "off-again") if "decode_tok_s_median" in st.get(k, {})]
    base = statistics.mean(offs) if offs else None
    L = ["# Task 3: where the MTP gain goes", "",
         f"Fork commit {S['fork_commit'][:10]}, PQ2_0-mtp, round-1 prompts. Baseline (no drafting) is the mean of "
         f"the two off runs: {base} tok/s ({offs}). Env vars present in the fork: {S['env_vars_in_fork']}.", "",
         "Verify cost = tokens per step / speedup; 1.0 means verifying is free. Tokens per step = tokens / "
         "(tokens - accepted drafts).", "",
         "| setting | decode tok/s | speedup | draft_n | accepted | acceptance | tokens/step | verify cost | prefill 8K |",
         "|---|---|---|---|---|---|---|---|---|"]
    best = None
    for name, r in st.items():
        if "decode_tok_s_median" not in r:
            L.append(f"| {name} | {r.get('error')} | | | | | | | |")
            continue
        sp = r["decode_tok_s_median"] / base if base else None
        tps = r["tokens_per_step"]
        vc = tps / sp if (sp and tps) else None
        f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
        L.append(f"| {name} | {r['decode_tok_s_median']} | {f(sp)} | {r['draft_n']} | {r['draft_n_accepted']} | "
                 f"{f(r['acceptance'])} | {f(tps)} | {f(vc, 2)} | {r['prefill_8k_tok_s_median']} |")
        if not name.startswith("off") and (best is None or r["decode_tok_s_median"] > best[1]):
            best = (name, r["decode_tok_s_median"], sp)
    if best:
        L += ["", f"Best setting: **{best[0]}** at {best[1]} tok/s, {best[2]:.3f}x the no-drafting baseline."]
    return "\n".join(L)


# ---------------------------------------------------------------- Task 4: q8_0 KV at depth

def task_kv(args):
    env, out = start("kv", args)
    raw = out / "raw"
    S = {"env": env, "args": vars(args)}
    bindir, commit = bench.build_fork()
    model = bench.hf_download(bench.GGUF_REPO, bench.GGUFS["PQ2_0"], WORK / "gguf")
    S["fork_commit"], S["file_bytes"] = commit, model.stat().st_size
    S["runs"] = {}
    for kv in ("f16", "q8_0"):
        S["runs"][kv] = bench_run(bindir, model, ["-p", "0", "-n", "128", "-d", args.depths, "-r", "2",
                                                  "-ctk", kv, "-ctv", kv], f"kv-{kv}", raw)
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    # a second f16 run last, to see run-to-run drift against the gain being measured
    S["runs"]["f16_again"] = bench_run(bindir, model, ["-p", "0", "-n", "128", "-d", args.depths, "-r", "2",
                                                       "-ctk", "f16", "-ctv", "f16"], "kv-f16-again", raw)
    S["warnings"] = {}
    for kv in ("f16", "q8_0"):
        err = (raw / f"kv-{kv}.stderr.log").read_text()
        S["warnings"][kv] = [l.strip() for l in err.splitlines() if re.search(r"warn|quality|degrad", l, re.I)][:10]
    md = summarize_kv(S)
    (out / "summary.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n" + md)
    log(f"wrote {out}")


def predicted_gain(file_bytes, depth):
    """Roofline: bytes read per token with fp16 KV over bytes with q8_0 KV (34 bytes per 32 values)."""
    f16 = bench.per_token_bytes(file_bytes, depth, bench.KV_BYTES_PER_TOKEN_F16)
    q8 = bench.per_token_bytes(file_bytes, depth, bench.KV_BYTES_PER_TOKEN_F16 * 34 / 64)
    return f16 / q8 - 1


def summarize_kv(S):
    def by(r):
        return {x["n_depth"]: x["avg_ts"] for x in r["rows"]} if isinstance(r, dict) and "rows" in r else {}
    a, b, a2 = by(S["runs"]["f16"]), by(S["runs"]["q8_0"]), by(S["runs"]["f16_again"])
    L = ["# Task 4: does 8-bit KV pay at long context?", "",
         f"Fork commit {S['fork_commit'][:10]}, PQ2_0, `llama-bench -p 0 -n 128 -r 2`, decode tok/s by depth. "
         "Predicted gain is the roofline bytes ratio (file + GDN state + 64 KiB x depth of KV).", "",
         "| depth | f16 | q8_0 | f16 (again) | measured gain | predicted gain | f16 drift |", "|---|---|---|---|---|---|---|"]
    for d in sorted(a):
        g = (b[d] / a[d] - 1) * 100 if d in b else None
        drift = (a2[d] / a[d] - 1) * 100 if d in a2 else None
        pg = predicted_gain(S["file_bytes"], d) * 100
        L.append(f"| {d} | {a[d]:.1f} | {b.get(d, float('nan')):.1f} | {a2.get(d, float('nan')):.1f} | "
                 f"{'n/a' if g is None else f'{g:+.1f}%'} | {pg:+.1f}% | {'n/a' if drift is None else f'{drift:+.1f}%'} |")
    L += ["", f"Warnings printed with q8_0 KV: {S['warnings']['q8_0'] or 'none'}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=["tensor", "kernels", "mtp", "kv"])
    ap.add_argument("--into", type=Path, help="kernels: the round2-tensor results directory to add to")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2, help="mtp: rounds over the prompt set per setting")
    ap.add_argument("--depths", default="0,8192,32768,65536")
    ap.add_argument("--upstream-tag", default="b12823")
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--cooldown", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"tensor": task_tensor, "kernels": task_kernels, "mtp": task_mtp, "kv": task_kv}[args.task](args)


if __name__ == "__main__":
    main()
