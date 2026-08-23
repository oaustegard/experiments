"""Stream a random sample of SYNTH rows via the HF datasets-server rows API."""
import gzip, json, random, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import httpx

OUT = sys.argv[1]
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LEN = 100
TOTAL = 1_089_584          # rows exposed by the (partial) datasets-server index
BASE = "https://datasets-server.huggingface.co/rows"
KEEP = ("synth_id", "language", "exercise", "query", "query_seed_text",
        "constraints", "synthetic_reasoning", "synthetic_answer", "words")

random.seed(20260823)
offsets = random.sample(range(0, TOTAL - LEN), N_REQ)
client = httpx.Client(timeout=120.0)


def fetch(off):
    url = (f"{BASE}?dataset={urllib.parse.quote('PleIAs/SYNTH', safe='')}"
           f"&config=default&split=train&offset={off}&length={LEN}")
    for attempt in range(4):
        try:
            r = client.get(url)
            if r.status_code == 200:
                return [{k: row["row"].get(k) for k in KEEP} for row in r.json()["rows"]]
            print(f"  off={off} HTTP {r.status_code}", flush=True)
        except Exception as e:
            print(f"  off={off} {type(e).__name__}: {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    return []


t0 = time.time()
n = 0
with gzip.open(OUT, "wt") as fh, ThreadPoolExecutor(max_workers=4) as ex:
    for i, rows in enumerate(ex.map(fetch, offsets)):
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += len(rows)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{N_REQ} reqs, {n} docs, {time.time()-t0:.0f}s", flush=True)
            fh.flush()
print(f"DONE {n} docs in {time.time()-t0:.0f}s -> {OUT}", flush=True)
