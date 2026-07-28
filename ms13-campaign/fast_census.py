import sys, itertools, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_proof as k

def unrooted_sig(edges, m):
    """Canonical signature of a free tree: AHU from the center(s)."""
    return k.ahu_signature(k.adjacency(edges, m), m)

def free_trees(mmax):
    """Non-isomorphic free trees per m, by leaf extension: every tree on m
    nodes is a tree on m-1 nodes plus a pendant leaf."""
    trees={1:[[]], 2:[[(0,1)]]}
    for m in range(3, mmax+1):
        seen={}
        for edges in trees[m-1]:
            for attach in range(m-1):
                e2 = list(edges)+[(attach, m-1)]
                sig = unrooted_sig(e2, m)
                if sig not in seen: seen[sig]=e2
        trees[m]=list(seen.values())
    return trees

t0=time.time()
T=free_trees(10)
counts=[len(T[m]) for m in range(1,11)]
print('free-tree counts m=1..10:', counts)
print('expected (A000055):      [1, 1, 1, 2, 3, 6, 11, 23, 47, 106]')
ok = counts == [1,1,1,2,3,6,11,23,47,106]
print('MATCHES known sequence:', ok, f'({time.time()-t0:.1f}s)')
if not ok: sys.exit(1)

# now redo the census using these shapes, m=2..10, via k3_proof's own row machinery
found={}
for m in range(2,11):
    before=len(found)
    for edges in T[m]:
        masks=k.subtree_masks(edges, m, 0)
        for markers in k._marker_multisets(m):
            rows=k.rows_from_markers(masks, markers)
            key=k.canonical_rowset(rows)
            if key and key not in found: found[key]={'m':m,'markers':markers}
    print(f'  m={m}: +{len(found)-before} -> {len(found)} total  ({time.time()-t0:.1f}s)', flush=True)
print('\nEXHAUSTIVE THROUGH m=10 total types:', len(found))
mx=k.find_maximal(found)
print('maximal types:', len(mx))
for t in mx: print('   ', list(t))
json.dump({'total':len(found),'counts':counts,
           'maximal':[[list(r) for r in t] for t in mx]},
          open('/tmp/fast_census.json','w'), indent=1, default=str)
