"""Self-conditioning test (arXiv 2509.09677 §3.2) via claude -p --resume on crafted transcripts.

Builds a session JSONL whose assistant turns carry running sums with an injected error
rate e, resumes it with one more key, and scores TURN accuracy: correct iff the reply
equals (last reported sum in history) + D[new key].
"""
import json, os, random, re, subprocess, sys, uuid, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
PROJ = os.path.expanduser("~/.claude/projects/" + re.sub(r"[^A-Za-z0-9]", "-", WORK))
TEMPLATE = os.path.join(HERE, "template.jsonl")  # real probe session, used for field shapes

WORDS = ["apple","brick","chair","delta","eagle","flame","grape","house","ivory","joker","knife","lemon","mango","night","ocean","piano","queen","river","stone","tiger","umbra","vivid","whale","xenon","yacht","zebra","amber","blaze","cloud","drift","ember","frost","glove","heart","index","jelly","koala","laser","maple","nerve","olive","pearl","quilt","radar","sheep","torch","ultra","vapor","wheat","yield"]
SYSTEM = ("You maintain a running sum. Starting from 0, each user turn gives one key; add its "
          "dictionary value to your previous answer. Reply with ONLY <answer>N</answer>.")

def templates():
    u = a = None
    for l in open(TEMPLATE):
        d = json.loads(l)
        if d.get("type") == "user" and u is None: u = d
        if d.get("type") == "assistant" and a is None: a = d
    return u, a

def build(sid, D, keys, reported):
    u0, a0 = templates()
    dtxt = "\n".join(f"{w}: {v}" for w, v in D.items())
    lines, parent = [], None
    t0 = time.time() - 3600
    for i, (k, s) in enumerate(zip(keys, reported)):
        uu, au = str(uuid.uuid4()), str(uuid.uuid4())
        text = (f"Dictionary:\n{dtxt}\n\nKey: {k}" if i == 0 else f"Key: {k}")
        u = dict(u0, uuid=uu, parentUuid=parent, sessionId=sid, cwd=WORK, promptId=str(uuid.uuid4()),
                 timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t0 + 2 * i)),
                 message={"role": "user", "content": text})
        m = dict(a0["message"]); m.update(id="msg_" + uuid.uuid4().hex[:24],
                 content=[{"type": "text", "text": f"<answer>{s}</answer>"}])
        a = dict(a0, uuid=au, parentUuid=uu, sessionId=sid, cwd=WORK, message=m,
                 requestId="req_" + uuid.uuid4().hex[:24],
                 timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t0 + 2 * i + 1)))
        lines += [u, a]; parent = au
    os.makedirs(PROJ, exist_ok=True)
    with open(os.path.join(PROJ, sid + ".jsonl"), "w") as f:
        for d in lines: f.write(json.dumps(d) + "\n")

def sample(seed, H, e):
    r = random.Random(seed)
    D = {w: r.randint(-99, 99) for w in WORDS}
    keys = [r.choice(WORDS) for _ in range(H + 1)]
    reported, s = [], 0
    for k in keys[:H]:
        s = s + D[k]
        if r.random() < e:
            s += r.choice([-1, 1]) * r.randint(1, 30)   # planted error; later turns build on it
        reported.append(s)
    return D, keys, reported

def run_one(job):
    model, think, e, seed, H = job
    D, keys, reported = sample(seed, H, e)
    sid = str(uuid.uuid4())
    build(sid, D, keys[:H], reported)
    expect = reported[-1] + D[keys[H]]
    env = dict(os.environ)
    cmd = ["claude", "-p", f"Key: {keys[H]}", "--resume", sid, "--model", model,
           "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--output-format", "json", "--effort", think]
    t = time.time()
    p = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    try:
        d = json.loads(p.stdout); res = d.get("result", "")
        u = d.get("usage", {}); th = (u.get("output_tokens_details") or {}).get("thinking_tokens")
        cost = d.get("total_cost_usd")
    except Exception:
        res, th, cost = "PARSE_FAIL " + p.stdout[:200] + p.stderr[:200], None, None
    m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", res or "")
    got = int(m.group(1)) if m else None
    return dict(model=model, think=think, e=e, seed=seed, H=H, expect=expect, got=got,
                ok=(got == expect), thinking_tokens=th, cost=cost, secs=round(time.time() - t, 1),
                raw=(res or "")[:120])

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sanity":
        D, keys, rep = sample(1, 3, 0.0)
        sid = str(uuid.uuid4()); build(sid, D, keys[:3], rep)
        p = subprocess.run(["claude", "-p", "Ignore the task for one turn: quote your previous three answers verbatim.",
                            "--resume", sid, "--model", "haiku", "--system-prompt", SYSTEM, "--tools", "",
                            "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"],
                           cwd=WORK, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        print("planted:", rep); print("model:", json.loads(p.stdout).get("result") if p.stdout else p.stderr[:400])
        sys.exit()
    H, n, par = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    models = sys.argv[5].split(","); thinks = sys.argv[6].split(","); es = [float(x) for x in sys.argv[7].split(",")]
    jobs = [(m, t, e, 1000 + i, H) for m in models for t in thinks for e in es for i in range(n)]
    out = os.path.join(HERE, f"results_{mode}.jsonl")
    with ThreadPoolExecutor(par) as ex, open(out, "a") as f:
        for r in ex.map(run_one, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
    open(os.path.join(HERE, f"{mode}.done"), "w").write("ok")
