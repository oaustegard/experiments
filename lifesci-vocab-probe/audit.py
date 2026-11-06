"""Tokenizer fragmentation on PubMed abstracts: stock tokenizers vs domain BPE trained on the train split."""
import json, re, collections, urllib.request
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from wordfreq import top_n_list
D = Path(__file__).resolve().parent
docs = {}
for p in sorted((D / "data").glob("abstracts_*.jsonl")):
    for l in p.open():
        r = json.loads(l); docs.setdefault(r["pmid"], (p.name, r["text"]))
recent = [t for k, (f, t) in docs.items() if "2023" in f]
test = recent[-2000:]
train = [t for k, (f, t) in docs.items() if t not in set(test)]
docs = train + test
print("train", len(train), "test", len(test))
STOCK = {
  "ettin (ModernBERT BPE)": "jhu-clsp/ettin-encoder-17m",
  "BiomedBERT (PubMed WordPiece, uncased)": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}
def load(repo):
    if "BiomedNLP" in repo:
        from tokenizers import BertWordPieceTokenizer
        p = D / "data" / (repo.replace("/", "__") + ".vocab.txt")
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/vocab.txt", timeout=60).read())
        return BertWordPieceTokenizer(str(p), lowercase=True)
    p = D / "data" / (repo.replace("/", "__") + ".json")
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", timeout=60).read())
    return Tokenizer.from_file(str(p))
toks = {}
for k, r in STOCK.items():
    try: toks[k] = load(r)
    except Exception as e: print("skip", k, e)
# domain byte-level BPE, same pre-tokenizer family as ModernBERT
for V in (16384, 32768, 50368):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(train, trainers.BpeTrainer(vocab_size=V, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    toks[f"domain BPE {V//1024}K"] = t
# same, but split on whitespace only so 'IL-6', 'GLP-1', IUPAC names can become one token
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)])
t.decoder = decoders.ByteLevel()
t.train_from_iterator(train, trainers.BpeTrainer(vocab_size=50368, min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
toks["domain BPE 49K, whitespace pre-tok"] = t
common = set(top_n_list("en", 50000))
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/()\[\],'.α-ωΑ-Ω+]*[A-Za-z0-9)\]+]|[A-Za-z]")
words = collections.Counter(w for d in test for w in WORD.findall(d))
def cat(w):
    lw = w.lower()
    if lw in common: return "ordinary"
    if re.fullmatch(r"[A-Z][A-Za-z]*\d+[A-Za-z0-9]*(-\d+)?|[A-Z]{2,}\d*[A-Z]*", w): return "gene/protein-symbol-like"
    if re.search(r"\d", w) and re.search(r"[-,()\[\]]", w) and len(re.findall(r"[A-Za-z]", w)) >= 3: return "chemical/IUPAC-like"
    if len(w) >= 10: return "long technical word"
    return "other rare"
cats = collections.defaultdict(list)
for w, c in words.items():
    cats[cat(w)].append((w, c))
res = {"n_test_docs": len(test), "categories": {k: [len(v), sum(c for _, c in v)] for k, v in cats.items()}, "tokenizers": {}}
for name, t in toks.items():
    row = {"vocab": t.get_vocab_size()}
    ntok = sum(len(e.ids) for e in t.encode_batch(test, add_special_tokens=False))
    row["tokens_per_abstract"] = ntok / len(test)
    for k, v in cats.items():
        enc = t.encode_batch([(" " + w) for w, _ in v], add_special_tokens=False)
        tot = sum(c for _, c in v)
        row[k] = sum(len(e.ids) * c for e, (_, c) in zip(enc, v)) / tot
        row[k + " >=3 pieces"] = sum(c for e, (_, c) in zip(enc, v) if len(e.ids) >= 3) / tot
    res["tokenizers"][name] = row
# how much of Ettin's vocab is used on life-science text at all
et = toks["ettin (ModernBERT BPE)"]
used = collections.Counter(i for e in et.encode_batch(docs, add_special_tokens=False) for i in e.ids)
tot = sum(used.values()); cum = 0; srt = sorted(used.values(), reverse=True)
cov = {}
for n in (4096, 8192, 16384, 24576):
    cov[n] = sum(srt[:n]) / tot
res["ettin_vocab_usage"] = {"distinct_ids_used_in_10k_abstracts": len(used), "of": et.get_vocab_size(), "token_mass_covered_by_top_n": cov}
# examples
ex = sorted(cats["gene/protein-symbol-like"], key=lambda x: -x[1])[:8] + sorted(cats["chemical/IUPAC-like"], key=lambda x: -x[1])[:6] + sorted(cats["long technical word"], key=lambda x: -x[1])[:6]
res["examples"] = {w: {n: t.encode(" " + w, add_special_tokens=False).tokens for n, t in toks.items() if "16K" not in n} for w, _ in ex}
(D / "results").mkdir(exist_ok=True)
(D / "results/audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
for w in list(res["examples"])[:20]:
    print(w, {n[:12]: len(v) for n, v in res["examples"][w].items()})
