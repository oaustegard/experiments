"""Fan a JSONL of tasks out to headless Haiku children.

Each task: {"id", "prompt", "system", "schema"?, "tools"?, "model"?, "add_dir"?}
Writes <out>/<id>.json = {"id", "output", "cost", "secs"}; existing files are skipped,
so a re-run resumes. Usage: python3 fan.py tasks.jsonl outdir [--conc 24]
"""
import asyncio, json, os, pathlib, sys, time

SCRATCH = pathlib.Path(os.environ.get("FAN_CWD", "/tmp/claude-0/fan-cwd")); SCRATCH.mkdir(parents=True, exist_ok=True)
MODEL = "claude-haiku-5-5"

async def run_one(t, out, sem, stats):
    dest = out / f"{t['id']}.json"
    if dest.exists():
        stats["skip"] += 1; return
    cmd = ["claude", "-p", "--model", t.get("model", MODEL), "--output-format", "json",
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--system-prompt", t["system"]]
    tools = t.get("tools", "")
    cmd += ["--tools", tools] if tools else ["--tools", ""]
    if tools:
        cmd += ["--allowedTools", tools]
    for d in t.get("add_dir", []):
        cmd += ["--add-dir", d]
    if t.get("schema"):
        cmd += ["--json-schema", json.dumps(t["schema"])]
    async with sem:
        for attempt in range(3):
            t0 = time.time()
            try:
                p = await asyncio.create_subprocess_exec(*cmd, cwd=SCRATCH, stdin=asyncio.subprocess.PIPE,
                                                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                so, se = await asyncio.wait_for(p.communicate(t["prompt"].encode()), timeout=t.get("timeout", 420))
                d = json.loads(so)
                outp = d.get("structured_output") if t.get("schema") else d.get("result")
                if d.get("is_error") or outp is None:
                    raise RuntimeError(f"child error: {str(d.get('result'))[:200]}")
                dest.write_text(json.dumps(dict(id=t["id"], output=outp, cost=d.get("total_cost_usd", 0),
                                                secs=round(time.time() - t0, 1))))
                stats["ok"] += 1; stats["cost"] += d.get("total_cost_usd", 0)
                break
            except Exception as e:
                try: p.kill()
                except Exception: pass
                stats["retry"] += 1
                err = f"{type(e).__name__}: {e}"[:300]
                await asyncio.sleep(3 * (attempt + 1))
        else:
            stats["fail"] += 1
            (out / f"{t['id']}.err").write_text(err)
        done = stats["ok"] + stats["fail"] + stats["skip"]
        if done % 10 == 0 or done == stats["n"]:
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{stats['n']} ok={stats['ok']} fail={stats['fail']} "
                  f"retry={stats['retry']} cost=${stats['cost']:.3f}", flush=True)

async def main(tasks_path, out, conc):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    stats = dict(n=len(tasks), ok=0, fail=0, skip=0, retry=0, cost=0.0)
    sem = asyncio.Semaphore(conc)
    t0 = time.time()
    await asyncio.gather(*(run_one(t, out, sem, stats) for t in tasks))
    print(f"DONE {stats} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    conc = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 24
    asyncio.run(main(sys.argv[1], sys.argv[2], conc))
