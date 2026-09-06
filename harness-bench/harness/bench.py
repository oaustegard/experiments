#!/usr/bin/env python3
"""Aider-Polyglot driver: materialize exercises, emit prompts, grade from pristine.

Subcommands:
  sample   pick the pilot task set (deterministic)
  prepare  build work/<arm>/<task>/ with stub + instructions, no test files
  prompts  write one prompt file per task (and a batch manifest)
  grade    rebuild pristine, overlay only solution files, run tests, emit JSON
"""
import argparse, json, os, random, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PG = Path(os.environ.get("POLYGLOT_ROOT", ROOT / "polyglot-benchmark"))

LANGS = {
    "python": dict(cmd=["python3", "-m", "pytest", "-x", "-q"], timeout=180),
    "go":     dict(cmd=["go", "test", "./..."],                 timeout=420),
    "rust":   dict(cmd=["cargo", "test", "--", "--include-ignored"], timeout=600),
}


def ex_dir(lang, task):
    return PG / lang / "exercises" / "practice" / task


def cfg(lang, task):
    return json.loads((ex_dir(lang, task) / ".meta" / "config.json").read_text())


def solution_files(lang, task):
    return cfg(lang, task)["files"]["solution"]


def test_files(lang, task):
    return cfg(lang, task)["files"].get("test", [])


def instructions(lang, task):
    docs = ex_dir(lang, task) / ".docs"
    parts = []
    for name in ("introduction.md", "instructions.md", "instructions.append.md"):
        p = docs / name
        if p.exists():
            parts.append(p.read_text())
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------- sample
def cmd_sample(a):
    rng = random.Random(a.seed)
    out = []
    for lang in a.langs.split(","):
        pool = sorted(p.name for p in (PG / lang / "exercises" / "practice").iterdir() if p.is_dir())
        out += [dict(lang=lang, task=t) for t in rng.sample(pool, a.n)]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(out)} tasks -> {a.out}")
    for t in out:
        print(" ", t["lang"], t["task"])


# ---------------------------------------------------------------- prepare
def cmd_prepare(a):
    tasks = json.loads(Path(a.tasks).read_text())
    base = ROOT / "work" / a.arm
    if base.exists():
        shutil.rmtree(base)
    for t in tasks:
        lang, task = t["lang"], t["task"]
        dst = base / lang / task
        shutil.copytree(ex_dir(lang, task), dst)
        # the agent must not see the tests or the reference solution
        shutil.rmtree(dst / ".meta", ignore_errors=True)
        for tf in test_files(lang, task):
            (dst / tf).unlink(missing_ok=True)
        if lang == "rust":
            shutil.rmtree(dst / "tests", ignore_errors=True)
        if lang == "go":
            for p in dst.glob("*_test.go"):
                p.unlink()
    print(f"prepared {len(tasks)} tasks under {base}")


# ---------------------------------------------------------------- prompts
PROMPT = """# Task: {task} ({lang})

Working directory: {wd}

Edit ONLY this file (create nothing else):
  {sol}

## Instructions

{instr}

## Current contents of {sol}

```{lang}
{stub}
```

## Rules

- Write a complete, correct implementation into {sol}.
- Keep the public names and signatures the stub declares; a hidden test suite
  imports them exactly as written.
- Do not create, delete or edit any other file.
{extra}"""

ONESHOT_EXTRA = "- Do not run any command. Write the file and stop."
RETRY_EXTRA = """- Your previous attempt failed the hidden test suite. Its output:

```
{feedback}
```

- Fix the implementation. Do not run any command. Write the file and stop."""


def cmd_prompts(a):
    tasks = json.loads(Path(a.tasks).read_text())
    fb = json.loads(Path(a.feedback).read_text()) if a.feedback else {}
    base = ROOT / "work" / a.arm
    outdir = ROOT / "prompts" / a.arm
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for t in tasks:
        lang, task = t["lang"], t["task"]
        key = f"{lang}/{task}"
        if fb and key not in fb:
            continue
        wd = base / lang / task
        sol = solution_files(lang, task)[0]
        extra = ONESHOT_EXTRA if not fb else RETRY_EXTRA.format(feedback=fb[key][:4000])
        body = PROMPT.format(task=task, lang=lang, wd=wd, sol=sol,
                             instr=instructions(lang, task),
                             stub=(wd / sol).read_text(), extra=extra)
        p = outdir / f"{lang}__{task}.md"
        p.write_text(body)
        written.append(str(p))
    (outdir / "MANIFEST.json").write_text(json.dumps(written, indent=1))
    print(f"{len(written)} prompts -> {outdir}")


# ---------------------------------------------------------------- grade
def cmd_grade(a):
    tasks = json.loads(Path(a.tasks).read_text())
    base = ROOT / "work" / a.arm
    graded = ROOT / "work" / f"_graded_{a.arm}"
    if graded.exists():
        shutil.rmtree(graded)
    results = {}
    for t in tasks:
        lang, task = t["lang"], t["task"]
        src, dst = base / lang / task, graded / lang / task
        shutil.copytree(ex_dir(lang, task), dst)          # pristine, incl. tests
        sols = solution_files(lang, task)
        for s in sols:
            if (src / s).exists():
                shutil.copy2(src / s, dst / s)
        # what did the agent touch outside its solution set?
        allowed = set(sols)
        stray = []
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(src))
            if rel in allowed:
                continue
            orig = ex_dir(lang, task) / rel
            if not orig.exists() or orig.read_bytes() != p.read_bytes():
                stray.append(rel)
        spec = LANGS[lang]
        # Per-task, per-arm target dir. A shared CARGO_TARGET_DIR let a todo!()
        # stub reuse the gold arm's compiled test binary and report 23 passed
        # (2026-09-06); cargo's fingerprint did not separate the two trees.
        env = dict(os.environ, GOFLAGS="-mod=mod", GOPATH="/tmp/gopath",
                   CARGO_TARGET_DIR=str(dst / "_target"))
        try:
            r = subprocess.run(spec["cmd"], cwd=dst, capture_output=True, text=True,
                               timeout=spec["timeout"], env=env)
            ok, out = r.returncode == 0, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired as e:
            ok, out = False, f"TIMEOUT after {spec['timeout']}s\n{(e.stdout or b'')[-2000:]}"
        results[f"{lang}/{task}"] = dict(passed=ok, stray=stray, output=out[-6000:])
        print(f"{'PASS' if ok else 'FAIL':4} {lang}/{task}" + (f"  STRAY={stray}" if stray else ""))
    Path(a.out).write_text(json.dumps(results, indent=1))
    n = sum(1 for v in results.values() if v["passed"])
    print(f"\n{a.arm}: {n}/{len(results)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="c", required=True)
    s = sub.add_parser("sample"); s.add_argument("--seed", type=int, default=20260906)
    s.add_argument("--n", type=int, default=4); s.add_argument("--langs", default="python,go,rust")
    s.add_argument("--out", required=True); s.set_defaults(f=cmd_sample)
    s = sub.add_parser("prepare"); s.add_argument("--tasks", required=True)
    s.add_argument("--arm", required=True); s.set_defaults(f=cmd_prepare)
    s = sub.add_parser("prompts"); s.add_argument("--tasks", required=True)
    s.add_argument("--arm", required=True); s.add_argument("--feedback")
    s.set_defaults(f=cmd_prompts)
    s = sub.add_parser("grade"); s.add_argument("--tasks", required=True)
    s.add_argument("--arm", required=True); s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_grade)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
