#!/usr/bin/env bash
# Check 1: positives vs PubMed similar-article negatives (+ easy negatives). Log: results/paper_nbr.log. Completion: 'NBR ARMS DONE'.
set -u; cd "$(dirname "$0")"; LOG=results/paper_nbr.log; mkdir -p results
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
C=data/paper_corpus_nbr.jsonl
[ -s "$C" ] || python3 - <<'EOF' >> "$LOG" 2>&1
import json, os, sys
sys.path.insert(0, '.'); from paper_corpus import has_cue
rows = []; seen = set()
for fn, lab, neg in [("positives.jsonl", "msd", "pos"), ("neighbors.jsonl", "not_msd", "neighbor"), ("easy.jsonl", "not_msd", "easy")]:
    for l in open(os.path.join("data", "papers", fn)):
        if not l.strip(): continue
        r = json.loads(l); pmid = str(r.get("pmid") or "")
        if not pmid or pmid in seen or len((r.get("abstract") or "").split()) < 50: continue
        seen.add(pmid); rows.append({"url": f"pmid:{pmid}", "label": lab, "title": r.get("title") or "", "text": r.get("abstract") or "", "has_cue": has_cue(r.get("title"), r.get("abstract")), "neg_set": neg, "year": r.get("year")})
with open("data/paper_corpus_nbr.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
import collections; print("nbr corpus:", dict(collections.Counter((r["label"], r["neg_set"]) for r in rows)))
EOF
run() { local name=$1; shift; [ -s "results/$name.json" ] && { say "skip $name"; return; }; say "start $name"; "$@" >> "$LOG" 2>&1; say "end $name exit $?"; }
run nbr_ft_ettin_32m python3 train.py --arm ft --model models/ettin-encoder-32m --name nbr_ft_ettin_32m --corpus $C --class-weight --epochs 6
say "start s2 embeddings for neighbours"; python3 data/papers/fetch_s2_embeddings.py neighbors.jsonl >> "$LOG" 2>&1
run nbr_s2probe      python3 paper_s2_probe.py --corpus $C --name nbr_s2probe --knn 10
python3 paper_score.py $C nbr_ >> "$LOG" 2>&1
say "NBR ARMS DONE"
