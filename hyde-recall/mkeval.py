import sys, os, json, random, time, threading, concurrent.futures as cf
sys.path.insert(0,'/mnt/skills/user'); sys.path.insert(0,os.path.expanduser('~'))
sys.path.append('/mnt/skills/user/invoking-gemini/scripts')
from muninn_utils.memory_tfidf import MemoryIndex
from gemini_client import invoke_gemini

_sem = threading.Semaphore(3)
def gem(**kw):
    for a in range(6):
        try:
            with _sem:
                return invoke_gemini(**kw)
        except Exception as e:
            if "429" not in str(e) and "Rate limited" not in str(e): raise
            time.sleep(1.5 * (2 ** a))
    return None

INSTR="""Below is one entry from an engineering memory store.

Write ONE question that this entry answers - the question someone would actually type months
later when they half-remember it.

HARD CONSTRAINT: do not reuse the entry's distinctive vocabulary. Avoid its proper nouns,
tool names, error codes, file paths, acronyms and project names. Describe the thing in
ordinary words instead ("the version control tool where every save is a commit", not "jj").

12-25 words. Output the question only, no quotes, no preamble."""

if __name__ == "__main__":
    idx=MemoryIndex(); idx.build()
    random.seed(1729)
    pool=[(idx.ids[i], idx.summaries[i]) for i in range(len(idx.ids)) if len(idx.summaries[i])>400]
    sample=random.sample(pool, 80)
    def mk(item):
        mid, summ = item
        r = gem(prompt=f"{INSTR}\n\nENTRY:\n{summ[:1800]}", model="lite",
                max_output_tokens=300, thinking_level="minimal", temperature=0.8)
        return (mid, r.strip().strip('"').replace("\n"," ")) if r else None
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        qs=[x for x in ex.map(mk, sample) if x and len(x[1].split())>=6]
    json.dump(qs, open('evalset.json','w'), indent=1)
    print("questions", len(qs))
    for m,q in qs[:10]: print(f"  {m[:8]} {q}")
