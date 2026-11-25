"""MSD concept tokens for linkup-sparseup-embed-v2: one new vocabulary row per MSD catalog concept.

The 696 concepts of ../sparseup-v2-msd-probe/tag-vector/data/vocab.json (catalog names + HGNC surface forms)
each get a token id V + k (V = 50,370). Text is tokenized as usual, then every token run that a tagged
surface form covers is replaced by the concept's single token, so `IL-1β`, `IL1B` and `interleukin-1 beta`
are one input token and, through the tied decoder, one output dimension.

    tok = ConceptTokenizer(hf_tokenizer, vocab)        # vocab = vocab.json as a dict
    ids, pool = tok.encode(text, "query")              # model input ids (may contain ids >= V), pooling mask
    extend_model(model, tok)                           # resize the tied embedding/decoder, mean-of-pieces init
"""
from __future__ import annotations

import re

import torch
from tagger import keys, needs_case, upper_ok

WORD = re.compile(r"[^\s,;:()\[\]{}\"'“”‘’/-]+")
MAX_NGRAM = 8


class SpanTagger:
    """tagger.Tagger's greedy longest match, returning character spans."""

    def __init__(self, alias, block, case_free):
        self.alias, self.block, self.case_free = alias, set(block), set(case_free)

    def spans(self, text):
        words = []
        for m in WORD.finditer(text):
            s, e = m.start(), m.end()
            while s < e and text[s] == ".":
                s += 1
            while e > s and text[e - 1] == ".":
                e -= 1
            if s < e:
                words.append((s, e, text[s:e]))
        nk = [keys(w) for _, _, w in words]
        out, i = [], 0
        while i < len(words):
            hit = None
            for n in range(min(MAX_NGRAM, len(words) - i), 0, -1):
                cands = {""}
                for j in range(i, i + n):
                    cands = {c + k for c in cands for k in nk[j]} if nk[j] else cands
                    if len(cands) > 16:
                        cands = set(list(cands)[:16])
                raw = " ".join(w for _, _, w in words[i:i + n])
                ok = [c for c in sorted(cands) if not needs_case(c) or c in self.case_free or upper_ok(raw)]
                cid = next((self.alias[c] for c in ok if c in self.alias), None)
                if cid:
                    hit = (cid, n)
                elif any(c in self.block for c in ok):
                    hit = (None, n)
                if hit:
                    break
            if hit:
                if hit[0]:
                    out.append((words[i][0], words[i + hit[1] - 1][1], hit[0]))
                i += hit[1]
            else:
                i += 1
        return out


class ConceptTokenizer:
    def __init__(self, hf_tok, vocab, prefixes=("[Q]", "[D]"), base_vocab=50370, add_prefix_space=True):
        self.tok, self.V = hf_tok, base_vocab
        self.cids = list(vocab["concepts"])
        self.cid_index = {c: k for k, c in enumerate(self.cids)}
        self.names = [vocab["concepts"][c]["catalog_names"][0] for c in self.cids]
        self.tagger = SpanTagger(vocab["alias"], vocab["block"], vocab.get("case_free", []))
        self.qp, self.dp = prefixes
        self.add_prefix_space = add_prefix_space

    @property
    def n_new(self):
        return len(self.cids)

    def encode(self, text, kind="document", max_length=512, rewrite=True):
        prefix = self.qp if kind == "query" else self.dp if kind == "document" else ""
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        full = prefix + text
        enc = self.tok(full, truncation=False, return_offsets_mapping=True, return_special_tokens_mask=True)
        ids, offs, spec = enc["input_ids"], enc["offset_mapping"], enc["special_tokens_mask"]
        pool = [int(not (prefix and o[0] < len(prefix) and not s)) for o, s in zip(offs, spec)]
        if rewrite:
            spans = [(s + len(prefix), e + len(prefix), c) for s, e, c in self.tagger.spans(text)]
            new_ids, new_pool, j = [], [], 0
            for s, e, c in spans:
                while j < len(ids) and (spec[j] or offs[j][1] <= s):
                    new_ids.append(ids[j]); new_pool.append(pool[j]); j += 1
                covered = False
                while j < len(ids) and not spec[j] and offs[j][0] < e:
                    covered = True
                    j += 1
                if covered:
                    new_ids.append(self.V + self.cid_index[c]); new_pool.append(1)
            new_ids += ids[j:]
            new_pool += pool[j:]
            ids, pool = new_ids, new_pool
        if len(ids) > max_length:
            ids, pool = ids[:max_length - 1] + ids[-1:], pool[:max_length - 1] + pool[-1:]
        return ids, pool

    def piece_ids(self, k):
        """Old-vocabulary token ids of concept k's catalog name, as it appears mid-text."""
        return self.tok(" " + self.names[k], add_special_tokens=False)["input_ids"]


@torch.no_grad()
def extend_model(model, ctok, extra_fold=None):
    """Grow the tied token table and the decoder bias by one row per concept, each the mean of its pieces."""
    emb = model.get_input_embeddings()
    V0, H = emb.weight.shape
    W = emb.weight.data
    rows = torch.stack([W[ctok.piece_ids(k)].mean(0) for k in range(ctok.n_new)])
    new = torch.nn.Embedding(V0 + ctok.n_new, H, device=W.device, dtype=W.dtype)
    new.weight.data[:V0] = W
    new.weight.data[V0:] = rows
    model.set_input_embeddings(new)
    dec = model.decoder
    bias = dec.bias.data
    nb = torch.stack([bias[ctok.piece_ids(k)].mean() for k in range(ctok.n_new)])
    dec.weight = new.weight
    dec.bias = torch.nn.Parameter(torch.cat([bias, nb]))
    dec.out_features = V0 + ctok.n_new
    model.config.vocab_size = V0 + ctok.n_new
    return V0
