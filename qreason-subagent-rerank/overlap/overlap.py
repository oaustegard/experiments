"""How many BRIGHT test queries appear verbatim in ReasonRank's training queries (reasonrank_data_13k),
which QReason also trains its rewriter on. Match = a 60-char window of the normalised BRIGHT query
text (from its start, or chars 100-160) occurs in the normalised concatenation of that domain's
training queries. BRIGHT text comes from QueryGym's genqr files (original query + appended keywords),
so a miss is possible where the original is short; hits are the floor, not the ceiling.
The id_query/*.json files are small enough that huggingface.co serves them directly (no CDN)."""
import csv
import json
import os
import re
import urllib.request

csv.field_size_limit(10**9)
dom = {'biology': 'biology', 'earth-science': 'earth_science', 'economics': 'economics',
       'robotics': 'robotics', 'stackoverflow': 'stackoverflow', 'sustainable-living': 'sustainable_living'}
os.makedirs('data', exist_ok=True)
def get(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'muninn-raven'})) as r:
            open(path, 'wb').write(r.read())
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
out, T, B = {}, 0, 0
for qg, rr in dom.items():
    get(f'https://raw.githubusercontent.com/ls3-lab/QueryGym/HEAD/reproducibility/data/runs/bright-{qg}/genqr/openai/gpt-4.1/bm25/bb1e250c.queries.tsv', f'data/bright_{qg}.tsv')
    get(f'https://huggingface.co/datasets/liuwenhan/reasonrank_data_13k/resolve/main/id_query/{rr}.json', f'data/train_{rr}.json')
    bq = {r[0]: r[1] for r in csv.reader(open(f'data/bright_{qg}.tsv', newline=''), delimiter='\t') if len(r) >= 2}
    train = ' || '.join(norm(v) for v in json.load(open(f'data/train_{rr}.json')).values())
    hits = 0
    for v in bq.values():
        n = norm(v)
        probes = [n[:60], n[100:160]] if len(n) > 200 else [n[:60]]
        hits += any(len(p) == 60 and p in train for p in probes)
    out[qg] = [hits, len(bq)]; T += hits; B += len(bq)
    print(f'{qg:18s} {hits:3d}/{len(bq):3d} = {100*hits/len(bq):3.0f}%')
print(f'total {T}/{B} of these six subsets; {T}/1384 = {100*T/1384:.0f}% of BRIGHT')
json.dump(out, open('overlap_counts.json', 'w'), indent=1)
