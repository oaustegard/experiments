"""Directional-duality probe, paired: claude-fable-5 vs claude-fable-5-1.

Rebuild of the 2026-07-22 in-container self-probe (memory 84edd20f), which was
never filed. Same design: single-direction walks (digit length of addition,
run count) vs a decorated-frame control at matched difficulty. Both models run
in the same harness, same seeded operands, so the comparison is within-run.

Harness rules carried over from July 22/23: no temperature param (deprecated on
fable-5), generous max_tokens (thinking shares the budget), empty text is a
harness event not a model error, score = last integer in text, exact match.
"""
import json, os, random, re, sys, time, concurrent.futures as cf
import urllib.request

API_KEY = os.environ["API_KEY"]
MODELS = ["claude-fable-5", "claude-fable-5-1"]
N = 3
SEED = 42
MAX_TOKENS = 16000

PLAIN = "Compute: {a} + {b}\nReply with only the integer."
DECOR = ("You are a sentient fog bank mediating a treaty between rival lighthouse "
         "unions on Neptune. Clause 9 requires the exact sum of {a} and {b} lumens "
         "of condensed moonlight, or the herring parliament dissolves. Reply with "
         "only the integer.")
COUNT = ("Here is a list of words:\n\n{s}\n\nHow many times does the word 'hare' "
         "appear in the list? Reply with only the integer.")

def rnd_int(rng, d):
    return int("".join([str(rng.randint(1, 9))] + [str(rng.randint(0, 9)) for _ in range(d - 1)]))

def build_cells():
    rng = random.Random(SEED)
    cells = []
    for d in (16, 40, 60, 90):
        for t in range(N):
            a, b = rnd_int(rng, d), rnd_int(rng, d)
            cells.append(dict(task="add", frame="plain", n=d, trial=t, prompt=PLAIN.format(a=a, b=b), truth=a + b))
    for t in range(N):
        a, b = rnd_int(rng, 40), rnd_int(rng, 40)
        cells.append(dict(task="add", frame="decor", n=40, trial=t, prompt=DECOR.format(a=a, b=b), truth=a + b))
    for k in (80, 200, 500):
        for t in range(N):
            cells.append(dict(task="count", frame="plain", n=k, trial=t, prompt=COUNT.format(s=" ".join(["hare"] * k)), truth=k))
    return cells

def call(model, prompt):
    body = json.dumps({"model": model, "max_tokens": MAX_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            b = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt); continue
            return {"type": "error", "error": {"status": e.code, "message": b[:500]}}
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt); continue
            return {"type": "error", "error": {"message": repr(e)}}

def score(cell, resp):
    rec = dict(cell); rec.pop("prompt")
    if resp.get("type") != "message":
        rec.update(cls="harness_error", raw=json.dumps(resp.get("error"))[:300]); return rec
    text = "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text").strip()
    u = resp.get("usage", {})
    rec.update(stop=resp.get("stop_reason"), out_tokens=u.get("output_tokens"),
               think_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens"),
               raw=text[:120])
    ints = re.findall(r"\d[\d,]*", text)
    if not text:
        rec.update(cls="empty_text", answer=None); return rec
    if not ints:
        rec.update(cls="no_integer", answer=None); return rec
    ans = int(ints[-1].replace(",", ""))
    rec["answer"] = ans
    if ans == cell["truth"]:
        rec["cls"] = "ok"
    elif cell["task"] == "add" and len(str(ans)) == len(str(cell["truth"])):
        rec["cls"] = "nearmiss"
    else:
        rec["cls"] = "wrong"
    return rec

def main():
    cells = build_cells()
    jobs = [(m, c) for m in MODELS for c in cells]
    out = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(call, m, c["prompt"]): (m, c) for m, c in jobs}
        for f in cf.as_completed(futs):
            m, c = futs[f]
            rec = score(c, f.result()); rec["model"] = m
            out.append(rec)
            print(f"{m:16} {c['task']:5} {c['frame']:5} n={c['n']:<3} t{c['trial']} {rec['cls']:13} think={rec.get('think_tokens')}", flush=True)
    out.sort(key=lambda r: (r["model"], r["task"], r["frame"], r["n"], r["trial"]))
    json.dump(out, open("results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
