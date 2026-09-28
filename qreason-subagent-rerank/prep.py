"""Build the BRIGHT-biology inputs for the subagent reranking arms.

Sources (all reachable without the Hugging Face CDN, which this container's egress blocks):
  - candidates: ReasonIR top-100 per query, retrieved with BRIGHT's GPT-4 reasoning queries,
    with passage text. GitHub LFS copy in JOHNNY-fans/TFRank:
    https://media.githubusercontent.com/media/JOHNNY-fans/TFRank/HEAD/evaluation/input/BRIGHT_reasonir_gpt4/biology.yesno_score.jsonl
    First-stage NDCG@10 over all 103 queries from this file: 43.40 (QReason Table 1: 43.49).
  - qrels: https://raw.githubusercontent.com/texttron/tevatron/HEAD/examples/ReasonIR/bright_qrels/biology.tsv
  - original query text: common case-insensitive prefix of two QueryGym expansions of the same
    query (genqr = original + keywords, csqe = lowercased original + expansion):
    ls3-lab/QueryGym reproducibility/data/runs/bright-biology/{genqr,csqe}/openai/gpt-4.1/bm25/*.queries.tsv
    Whitespace is normalised (newlines lost).

Run from this directory: python3 prep.py  (downloads ~92 MB into data/)
"""
import collections
import csv
import json
import os
import random
import urllib.request

D = 'data/'
URLS = {
    'cands.jsonl': 'https://media.githubusercontent.com/media/JOHNNY-fans/TFRank/HEAD/evaluation/input/BRIGHT_reasonir_gpt4/biology.yesno_score.jsonl',
    'qrels.tsv': 'https://raw.githubusercontent.com/texttron/tevatron/HEAD/examples/ReasonIR/bright_qrels/biology.tsv',
    'genqr.tsv': 'https://raw.githubusercontent.com/ls3-lab/QueryGym/HEAD/reproducibility/data/runs/bright-biology/genqr/openai/gpt-4.1/bm25/bb1e250c.queries.tsv',
    'csqe.tsv': 'https://raw.githubusercontent.com/ls3-lab/QueryGym/HEAD/reproducibility/data/runs/bright-biology/csqe/openai/gpt-4.1/bm25/e1bfc505.queries.tsv',
}
K, TRUNC, N, SEED = 30, 1500, 16, 20260928


def fetch():
    os.makedirs(D, exist_ok=True)
    for name, url in URLS.items():
        if not os.path.exists(D + name):
            req = urllib.request.Request(url, headers={'User-Agent': 'muninn-raven'})
            with urllib.request.urlopen(req) as r, open(D + name, 'wb') as f:
                f.write(r.read())


def originals():
    csv.field_size_limit(10**9)
    load = lambda p: {r[0]: r[1] for r in csv.reader(open(p, newline=''), delimiter='\t') if len(r) >= 2}
    a, b = load(D + 'genqr.tsv'), load(D + 'csqe.tsv')
    out = {}
    for q in a:
        x, y = a[q], b[q]
        i = 0
        while i < min(len(x), len(y)) and x[i].lower() == y[i].lower():
            i += 1
        out[q] = x[:i].rstrip()
    return out


def main():
    fetch()
    qrels = collections.defaultdict(set)
    for line in open(D + 'qrels.tsv'):
        p = line.split()
        qrels[p[0]].add(p[2])
    cands = collections.defaultdict(list)
    for line in open(D + 'cands.jsonl'):
        r = json.loads(line)
        cands[r['qid']].append((r['docid'], r['passage']))
    orig = originals()
    # q0 excluded: its gold ids were visible to the orchestrator before the briefs were written
    elig = [q for q in cands if q != '0' and any(d in qrels[q] for d, _ in cands[q][:K])]
    random.seed(SEED)
    sel = sorted(random.sample(elig, N), key=int)
    mapping = {q: {f'p{i+1:02d}': d for i, (d, _) in enumerate(cands[q][:K])} for q in sel}
    os.makedirs('inputs', exist_ok=True)
    json.dump({'selected': sel, 'mapping': mapping, 'qrels': {q: sorted(qrels[q]) for q in sel}},
              open('keys.json', 'w'), indent=1)
    json.dump({q: orig[q] for q in sel}, open('queries.json', 'w'), indent=1)

    def block(q, lo, hi):
        out = [f'=== QUERY {q} ===\n{orig[q]}\n']
        for i in range(lo, hi):
            d, p = cands[q][i]
            out.append(f'--- [{q}:p{i+1:02d}] ---\n{p[:TRUNC]}' + (' [...]' if len(p) > TRUNC else '') + '\n')
        return '\n'.join(out)

    for k in range(3):  # batch k = retrieval ranks 10k+1 .. 10k+10, all 16 queries
        blocks = [block(q, 10 * k, 10 * k + 10) for q in sel]
        open(f'inputs/batch{k+1}.txt', 'w').write('\n\n'.join(blocks))
        open(f'inputs/batch{k+1}a.txt', 'w').write('\n\n'.join(blocks[:8]))   # half files: run-2 arms
        open(f'inputs/batch{k+1}b.txt', 'w').write('\n\n'.join(blocks[8:]))
    for g, qs in enumerate([sel[0:6], sel[6:11], sel[11:16]]):  # arm D: all 30 in one context
        open(f'inputs/full{g+1}.txt', 'w').write('\n\n'.join(block(q, 0, K) for q in qs))
    print('selected', sel)


if __name__ == '__main__':
    main()
