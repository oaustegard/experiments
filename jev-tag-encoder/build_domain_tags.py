"""Round 3: a 256-tag taxonomy fitted to the SciFact corpus, built from the corpus alone.

KMeans (k=256) over the gemini-embedding-2 document vectors, then gemini-3.5-flash-lite names each
cluster from the titles nearest its centroid, shown alongside its 5 nearest neighbour clusters so
names separate siblings. Queries and qrels are never read. Output: tags_scifact.txt (16 x 16, "|"),
clusters_scifact.json (cluster -> name, size, sample titles) for audit.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import KMeans

import jev
from common import DATA, jsonl
from embed import EMB

sys.path.append("/mnt/skills/user/invoking-gemini/scripts")
from gemini_client import invoke_gemini

K = 256
N_TITLES = 12


def name_cluster(own: list[str], neighbours: list[list[str]]) -> str | None:
    nb = "\n".join(f"- neighbour {k + 1}: " + " / ".join(t[:90] for t in ts[:4]) for k, ts in enumerate(neighbours))
    prompt = (
        "These biomedical paper titles form one topic cluster:\n"
        + "\n".join(f"- {t[:160]}" for t in own)
        + "\n\nNearby clusters (for contrast only):\n" + nb
        + "\n\nName the cluster's shared topic as a noun phrase of 2 to 6 words, specific enough to "
        "tell it apart from the nearby clusters (name the disease, organism, pathway, cell type or "
        "intervention where one is shared). Lowercase except proper nouns and gene/protein symbols. "
        "Reply with the phrase only."
    )
    for attempt in range(6):
        jev._pace()  # the AI Gateway's 50/min cap covers Gemini calls too
        try:
            out = invoke_gemini(prompt, model="gemini-3.5-flash-lite", temperature=0.2, max_output_tokens=80)
        except Exception as e:  # noqa: BLE001 — gateway 429 surfaces as a non-retriable client error
            print(f"retry {attempt}: {str(e)[:120]}", file=sys.stderr)
            time.sleep(15)
            continue
        if out and out.strip():
            first = out.strip().splitlines()[0]  # a reply that explains itself keeps only its first line
            name = re.sub(r"\s+", " ", re.sub(r"[|\"*.]", " ", first)).strip()
            if name:
                return " ".join(name.split()[:8])
    return None


def main():
    corpus = jsonl(DATA / "scifact" / "corpus.jsonl")
    titles = [r["title"] or r["text"][:120] for r in corpus]
    E = np.load(EMB / "scifact_docs.npy")
    km = KMeans(K, n_init=4, random_state=98).fit(E)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    sims = E @ C.T
    near = {c: [int(i) for i in np.argsort(-sims[:, c]) if km.labels_[i] == c][:N_TITLES] for c in range(K)}
    nbr = np.argsort(-(C @ C.T), axis=1)[:, 1:6]
    ckpt = EMB / "cluster_names.json"
    done = json.loads(ckpt.read_text()) if ckpt.exists() else {}

    def one(c):
        if not done.get(str(c)):
            done[str(c)] = name_cluster([titles[i] for i in near[c]], [[titles[i] for i in near[int(n)]] for n in nbr[c]])
            ckpt.write_text(json.dumps(done))
        return done[str(c)]
    with ThreadPoolExecutor(2) as ex:
        names = list(ex.map(one, range(K)))
    missing = [c for c, n in enumerate(names) if not n]
    if missing:
        raise SystemExit(f"unnamed clusters: {missing}")
    # duplicate names: keep the larger cluster's, suffix the rest with its most distinctive title word
    seen = {}
    for c in sorted(range(K), key=lambda c: -(km.labels_ == c).sum()):
        key = names[c].lower()
        if key in seen:
            names[c] = f"{names[c]} ({titles[near[c][0]].split(':')[0][:40].strip()})"
        seen.setdefault(names[c].lower(), c)
    (EMB.parent / "tags_scifact.txt").write_text("\n".join("|".join(names[r * 16:(r + 1) * 16]) for r in range(16)) + "\n")
    (EMB.parent / "clusters_scifact.json").write_text(json.dumps(
        [{"cluster": c, "tag": names[c], "size": int((km.labels_ == c).sum()),
          "titles": [titles[i] for i in near[c][:5]]} for c in range(K)], indent=1))
    sizes = np.bincount(km.labels_, minlength=K)
    print(f"cluster sizes: min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}")
    print("sample tags:", names[:24])


if __name__ == "__main__":
    main()
