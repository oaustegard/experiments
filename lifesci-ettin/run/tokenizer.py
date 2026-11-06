"""Life-sciences tokenizers (32K and 50K) and their mean-of-pieces token tables.

    python3 -I run/tokenizer.py   # .work/tok/ls{32,50}/tokenizer.json, .work/params/{stock,ls32,ls50}.f32, results/tokenizers.json

Pre-tokenizer: split on whitespace and punctuation, but keep letters and digits joined by an internal
- – / . , ' + as one word, so `IL-6`, `GLP-1`, `SARS-CoV-2`, `2,3-dihydro` can become one token; then
byte-level as ModernBERT. Corpus: the training split of PubMed text (titles + passages), wikitext-103
lines for general English (about a fifth of the text), HGNC approved symbols and names, MSD catalog
terms (msd-context-classifier), each repeated count/20 times (5 to 500). Specials and the [CLS] $A [SEP] template follow Ettin's.

Table init: each new token's row is the mean of e17c's rows for its string under Ettin's tokenizer; a
token Ettin also has gets that row exactly (a one-piece mean). Every other weight is e17c's.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import Embedder, load_st, to_flat  # noqa: E402

W = HERE / ".work"
E17C = W / "e17c" / "epoch1"
SPECIALS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
PAT = r" ?[\p{L}\p{N}]+(?:[-–/.,'+][\p{L}\p{N}]+)*| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def corpus():
    rng = random.Random(0)
    pub = [json.loads(l)["t"] for l in (W / "data" / "train.jsonl").open()]
    wt = []
    for f in ["train-00000-of-00002.parquet"]:
        df = pd.read_parquet(hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{f}", repo_type="dataset"))
        wt += [t.strip() for t in df["text"] if len(t.split()) > 20 and not t.strip().startswith("=")]
    pub_words = sum(len(t.split()) for t in pub)
    rng.shuffle(wt)
    gen, n = [], 0
    for t in wt:
        if n > pub_words / 4:
            break
        gen.append(t); n += len(t.split())
    hg = pd.read_csv(W / "hgnc.tsv", sep="\t", usecols=["symbol", "name", "status"], low_memory=False)
    hg = hg[hg["status"] == "Approved"]
    nom = [f"{s} {nm}" for s, nm in zip(hg["symbol"], hg["name"])]
    msd = json.loads((HERE.parent / "msd-context-classifier/vocab/msd_terms.json").read_text())["terms"]
    # each catalog term repeated in proportion to its count on mesoscale.com (count / 20, 5 to 500 times)
    rep = [t for t, v in msd.items() for _ in range(min(500, max(5, v["count"] // 20)))]
    rng.shuffle(rep)
    lines = pub + gen + nom + [" ".join(rep[i:i + 20]) for i in range(0, len(rep), 20)]
    rng.shuffle(lines)
    return lines, {"pubmed_lines": len(pub), "pubmed_words": pub_words, "wikitext_lines": len(gen), "wikitext_words": n,
                   "hgnc_lines": len(nom), "msd_terms": len(msd), "msd_term_occurrences": len(rep)}


def build(lines, V):
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), behavior="isolated"),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(lines, trainers.BpeTrainer(vocab_size=V, min_frequency=2, special_tokens=SPECIALS,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    t.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                                     special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))])
    return t


def init_table(new, old, table):
    V = new.get_vocab_size()
    out = np.zeros((V, table.shape[1]), np.float32)
    dec = decoders.ByteLevel()
    exact = 0
    for tok, i in new.get_vocab().items():
        if tok in SPECIALS:
            out[i] = table[old.token_to_id(tok)]
            continue
        s = dec.decode([tok])
        ids = old.encode(s, add_special_tokens=False).ids or [old.token_to_id("[UNK]")]
        exact += len(ids) == 1
        out[i] = table[ids].mean(0)
    return out, exact


def main():
    lines, cstats = corpus()
    print(cstats, flush=True)
    old = Tokenizer.from_file(str(E17C / "tokenizer.json"))
    old.no_truncation(); old.no_padding()
    p = load_st(E17C)
    (W / "params").mkdir(parents=True, exist_ok=True)
    to_flat(Embedder(p)).tofile(W / "params" / "stock.f32")
    res = {"corpus": cstats, "pattern": PAT, "arms": {}}
    for name, V in (("ls32", 32768), ("ls50", 50368)):
        t = build(lines, V)
        d = W / "tok" / name
        d.mkdir(parents=True, exist_ok=True)
        t.save(str(d / "tokenizer.json"))
        tab, exact = init_table(t, old, p["table"].numpy())
        q = dict(p)
        q["table"] = __import__("torch").from_numpy(tab)
        to_flat(Embedder(q)).tofile(W / "params" / f"{name}.f32")
        res["arms"][name] = {"vocab": t.get_vocab_size(), "rows_copied_exactly": exact}
        print(name, res["arms"][name], flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "tokenizers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
