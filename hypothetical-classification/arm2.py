"""Two more arms: the classic constrained-choice baseline, and batched hallucination.

structured  ship all 860 legal labels to the model, ask it to pick one.
batched     one call hallucinates classes for BATCH queries at once.
Both report input-token cost so the pattern's economics are measured, not asserted.
"""
import json, sys, time, re, numpy as np, concurrent.futures as cf
import bench

A = json.load(open("artifacts_lite.json"))
vocab, queries, goldn = A["vocab"], A["queries"], A["gold"]
gold = np.array([vocab.index(c) for c in goldn])
VOCAB_BLOB = "\n".join(vocab)

STRUCTURED = """Classify the search query into EXACTLY ONE of the legal product classes listed
below. Output the class verbatim, exactly as it appears in the list. No other text.

LEGAL PRODUCT CLASSES:
{vocab}

QUERY: {query}

CLASS:"""

BATCHED = """Your task is to create a novel, never-seen-before furniture, home goods, or
hardware product classification that best fits each search query.

Product classifications might look like: Coffee Tables / Throw Pillows /
Dressers & Chests / Food Storage & Canisters / Stackable Chairs / Kids Beds

Output one line per query, in the same order, formatted exactly as:
<n>. <classification>

QUERIES:
{numbered}"""

def structured_arm(qs, model="lite"):
    def one(q):
        r = bench.gem(prompt=STRUCTURED.format(vocab=VOCAB_BLOB, query=q), model=model,
                      max_output_tokens=200, thinking_level="minimal", temperature=0.0)
        return (r or "").strip().strip('"').split("\n")[0]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        return list(ex.map(one, qs))

def batched_arm(qs, model="lite", batch=40):
    chunks = [qs[i:i+batch] for i in range(0, len(qs), batch)]
    def one(ch):
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(ch))
        r = bench.gem(prompt=BATCHED.format(numbered=numbered), model=model,
                      max_output_tokens=4000, thinking_level="minimal", temperature=0.7) or ""
        got = {}
        for line in r.splitlines():
            m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if m: got[int(m.group(1))] = m.group(2).strip().strip('"')
        return [got.get(i+1, "") for i in range(len(ch))]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        return [x for ch in ex.map(one, chunks) for x in ch]

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    t = time.time(); st = structured_arm(queries); t_s = time.time() - t
    t = time.time(); bt = batched_arm(queries);   t_b = time.time() - t
    json.dump({"structured": st, "batched": bt}, open("artifacts_arm2.json", "w"))

    m = SentenceTransformer("all-MiniLM-L6-v2")
    L = m.encode(vocab, batch_size=128, show_progress_bar=False)

    # structured output is supposed to already BE a legal label; measure both
    # exact-match and what it scores if you snap it like everything else.
    exact = float(np.mean([s.strip() == g for s, g in zip(st, goldn)]))
    legal = float(np.mean([s.strip() in set(vocab) for s in st]))
    S = m.encode(st, batch_size=128, show_progress_bar=False)
    B = m.encode(bt, batch_size=128, show_progress_bar=False)
    print(f"{'arm':46} {'acc@1':>6} {'acc@3':>6}")
    print("-" * 62)
    print(f"{'structured: verbatim exact match':46} {exact:6.3f} {'-':>6}   (legal-output rate {legal:.3f})")
    for nm, V in (("structured: output snapped to label", S),
                  ("hallucination BATCHED x40 -> label", B)):
        idx = bench.snap(V, L, k=3)
        print(f"{nm:46} {bench.acc(idx,gold,1):6.3f} {bench.acc(idx,gold,3):6.3f}")
    print(f"\nwall-clock  structured {t_s:.0f}s   batched {t_b:.0f}s")
    ntok = lambda s: len(s) / 4
    print(f"input tokens/query  structured ~{ntok(STRUCTURED.format(vocab=VOCAB_BLOB, query='x')):.0f}"
          f"   single-hallucination ~{ntok(bench.HALL_PROMPT.format(query='x')):.0f}"
          f"   batched(40) ~{ntok(BATCHED.format(numbered='x'))/40 + 4:.0f}")
