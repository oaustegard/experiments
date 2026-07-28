import sys; sys.path.insert(0, '/mnt/skills/user/remembering')
from scripts import memory
import re
rows = memory._exec("SELECT id,summary,type,priority,t FROM memories WHERE deleted_at IS NULL AND is_superseded=0", [])
# ACTIVE only. Two questions:
# Q1: any active memory claiming the MEMORY STORE uses embeddings/vector/semantic clustering?
emb = re.compile(r'(embed|vector|cosine|knn|nearest.neighb|sentence.transform)', re.I)
store = re.compile(r'(consolidat|curat|dedup|near.?dup|recall|memory store|memory maintenance|sleep session|memory_tfidf|pruning)', re.I)
q1 = [m for m in rows if emb.search(m['summary'] or '') and store.search(m['summary'] or '')]
print(f"Q1 active memories mentioning BOTH embeddings AND store-machinery: {len(q1)}")
for m in q1:
    print(f"   {m['id'][:8]} {str(m['t'])[:10]} {m['type']}: {(m['summary'] or '')[:160]}")
print()
# Q2: active memories about memory maintenance / dedup claims
q2 = [m for m in rows if store.search(m['summary'] or '')]
print(f"Q2 active store-machinery memories: {len(q2)}")
for m in sorted(q2, key=lambda x:str(x['t'])):
    s=(m['summary'] or '')
    dup = ' [DUP-CLAIM]' if re.search(r'dup', s, re.I) else ''
    print(f"   {m['id'][:8]} {str(m['t'])[:10]} {m['type']:10s} pri={m['priority']}{dup}: {s[:120]}")
