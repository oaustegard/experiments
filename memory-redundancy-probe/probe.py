import sys; sys.path.insert(0, '/mnt/skills/user/remembering')
from muninn_utils.memory_tfidf import MemoryIndex
from scripts import memory
from collections import Counter

rows = memory._exec("SELECT id,summary,type,tags,priority,t,last_accessed,access_count FROM memories WHERE deleted_at IS NULL AND is_superseded=0", [])
N=len(rows)
ac = [int(m['access_count'] or 0) for m in rows]
never = sum(1 for x in ac if x==0)
print(f"{N} active memories")
print(f"never recalled (access_count=0): {never} ({100*never/N:.0f}%)")
print(f"recalled >=1: {N-never} ({100*(N-never)/N:.0f}%)")
import statistics as st
print(f"access_count: max={max(ac)} median={st.median(ac)} mean={st.mean(ac):.1f}")
# of recalled ones, distribution
rec = sorted([x for x in ac if x>0], reverse=True)
print(f"top access counts: {rec[:15]}")

# the Valid stubs
print("\n=== 'Valid' stubs ===")
for m in rows:
    if (m['summary'] or '').strip()=='Valid':
        print(f"  {m['id'][:8]} type={m['type']} pri={m['priority']} t={str(m['t'])[:10]} tags={m['tags']}")

# largest 0.55 cluster
idx=MemoryIndex(); idx.build()
cl=idx.clusters(threshold=0.55)
cl.sort(key=len, reverse=True)
big=cl[0]
print(f"\n=== largest cluster ({len(big)} mems) ===")
for item in big[:14]:
    # item is dict
    sid = item.get('id','?')[:8]
    prev = (item.get('summary') or item.get('preview') or '')[:80]
    print(f"  {sid}  {prev}")
