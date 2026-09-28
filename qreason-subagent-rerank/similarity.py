"""Cross-batch similarity of arm-C per-batch query analyses (run 1).
Dense cosines come from gemini-embedding-2 (768-d, SEMANTIC_SIMILARITY) over 8 queries x 3 batches,
computed via Muninn's gateway; the vectors are not stored, the pairwise cosines are in results/."""
import itertools
import json
import re
import statistics as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

an = {b: json.load(open(f'outputs/C_{b}.json'))['analysis'] for b in (1, 2, 3)}
docs = [(q, b, an[b][q]) for q in sorted(an[1], key=int) for b in (1, 2, 3)]
S = cosine_similarity(TfidfVectorizer(stop_words='english').fit_transform([t for *_, t in docs]))
tri = lambda t: (lambda w: {tuple(w[i:i + 3]) for i in range(len(w) - 2)})(re.findall(r'[a-z]+', t.lower()))
same, diff, js, jd = [], [], [], []
for i, j in itertools.combinations(range(len(docs)), 2):
    (qi, bi, ti), (qj, bj, tj) = docs[i], docs[j]
    if bi == bj: continue
    a, b = tri(ti), tri(tj); J = len(a & b) / len(a | b)
    (same if qi == qj else diff).append(S[i, j]); (js if qi == qj else jd).append(J)
e = json.load(open('results/c_analysis_embedding_cos.json'))
print(f'analysis length: mean {st.mean(len(t.split()) for *_, t in docs):.0f} words')
print(f'TF-IDF cosine      same query {st.mean(same):.3f} | different query {st.mean(diff):.3f}')
print(f'word-3gram Jaccard same query {st.mean(js):.3f} | different query {st.mean(jd):.3f}')
print(f"gemini-embedding-2 same query {st.mean(e['same']):.3f} (min {min(e['same']):.3f}) | different query {st.mean(e['diff']):.3f} (max {max(e['diff']):.3f})")
