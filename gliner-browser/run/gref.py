"""NumPy reference forward for a packed GLiNER-on-Ettin model (the spec the browser runtime follows).

    from gref import GRef
    g = GRef(Path("web/models/<name>"))
    g.predict_entities("John works at Google.", ["person", "organization"], threshold=0.5)

Reads only the packed directory (manifest.json, weights.bin, tokenizer.json) written by gpack.py and
reproduces the upstream gliner 0.2.30 uni-encoder pipeline end to end:

  regex word split (Python `\\w`, code-point offsets) -> words[:max_len]
  prompt = [<<ENT>>, label]* + [<<SEP>>]   (labels deduplicated first-wins, an empty label adds no word)
  per-word tokenization (no leading space unless the checkpoint's prefix_space flag is set), [CLS] ... [SEP]
  words_mask: 1-based text-word index on the first (or last) subtoken, 0 elsewhere
  encoder = browser-decider's `Ref.hidden` (ModernBERT, final-norm states), optional projection
  label vectors at the <<ENT>> positions, word vectors at the marked subtokens (zeros if none)
  optional BiLSTM (PyTorch gate order i,f,g,o; both biases; run over the n real words)
  span head markerV0, naive and factorised (see `span_logits`), or the token-level scorer
  sigmoid(logit / T + b), threshold strictly greater, greedy flat/nested selection with the library's
  tie-breaks (stable sort on -score over candidates in (start, width, class) order), char offsets.

Anything the packer could not represent is refused there; GRef re-checks the manifest and raises
NotImplementedError rather than produce numbers for a configuration it does not implement.

Bi-encoder (`GRefBi`, manifest gliner.encoder_kind == "bi", e.g. knowledgator/gliner-bi-edge-v2.0), the
gliner 0.2.30 BiEncoderSpanModel path:
  text sequence = [CLS] + per-word subtokens of the text words only + [SEP]   (no label prompt, no <<ENT>>)
  words_mask as above with a prompt length of 0
  each label alone -> label tokenizer (BERT WordPiece, lowercasing) -> [CLS] label [SEP] -> BERT (post-LN,
  absolute positions, token type 0) last_hidden_state -> mean over all its tokens ([CLS] and [SEP] included)
  word vectors -> (no BiLSTM: the library builds `rnn` for this checkpoint but its bi-encoder forward never
  calls it) -> markerV0 span head, prompt_rep_layer on the pooled label vectors, logits as above.
Label vectors depend on the label alone, so they are cached per label string.

`load(dir)` returns the right class for a packed directory. Every stage adds its wall time to
`self.timing` (text_encoder, label_encoder, lstm, span_head, decode) for the per-stage timing.
"""
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import experiment  # noqa: E402


def _load_bd_pack():
    """browser-decider/run/pack.py, imported by path under its own module name (not edited)."""
    p = experiment("browser-decider") / "run" / "pack.py"
    spec = importlib.util.spec_from_file_location("bd_pack", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load_bd_pack()

SPAN_MODES = ("markerV0", "token_level")
POOLING = ("first", "last")
STAGES = ("text_encoder", "label_encoder", "lstm", "span_head", "decode")

_DQ = {}   # (id(weights), offset) -> (weights, int8 codes as float32): a speed cache, same numbers


class Ref(bd.Ref):
    """browser-decider's Ref plus an 8-bit token table (groups of 32, fp16 scale, codes in +-127) and a
    cache of the int8 matmul codes converted to float32 (the product is computed exactly as bd.Ref does)."""

    def qf(self, r):
        k = (id(self.d), r["offset"])
        hit = _DQ.get(k)
        if hit is None:
            hit = _DQ[k] = (self.d, self.t(r, np.int8).astype(np.float32))
        return hit[1]

    def lin(self, x, w):
        if "f32_t" in w:
            return x @ self.t(w["f32_t"]).T
        return (x @ self.qf(w["q"])) * self.t(w["s"])

    def table(self, ids):
        T = self.m["table"]
        if T["bits"] != 8:
            return super().table(ids)
        H = T["H"]
        q = self.t(T["codes"], np.int8)[ids].astype(np.float32).reshape(len(ids), H // 32, 32)
        s = self.t(T["scales"], np.float16)[ids].astype(np.float32).reshape(len(ids), H // 32, 1)
        return (q * s).reshape(len(ids), H)


class BertRef(Ref):
    """NumPy BERT (HF BertModel, post-LN) over the `label_encoder` part of a packed manifest.

    embeddings: (word[ids] + token_type[0]) + position[0..S-1] -> LayerNorm; per layer
      a = softmax(q k^T / sqrt(d)) v (fused Wqkv with bias), x = LN(x + Wo a + bo),
      x = LN(x + Wo2 gelu_erf(Wi x + bi) + bo2); output = last_hidden_state (the pooler is not used).
    """

    def __init__(self, m, data):
        super().__init__(m, data)

    def encode(self, ids):
        m = self.m
        ids = np.asarray(ids)
        S, H, nh = len(ids), m["hidden"], m["heads"]
        D = H // nh
        x = (self.table(ids) + self.t(m["type0"])) + self.t(m["pos"])[:S]
        x = self.ln(x, self.t(m["emb_ln_w"]), self.t(m["emb_ln_b"]))
        for L in m["layers"]:
            qkv = (self.lin(x, L["Wqkv"]) + self.t(L["bqkv"])).reshape(S, 3, nh, D).transpose(1, 2, 0, 3)
            sc = np.einsum("hid,hjd->hij", qkv[0], qkv[1]) / np.float32(math.sqrt(D))
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            o = np.einsum("hij,hjd->hid", sc, qkv[2]).transpose(1, 0, 2).reshape(S, H)
            x = self.ln(x + (self.lin(o, L["Wo"]) + self.t(L["bo"])), self.t(L["ln1_w"]), self.t(L["ln1_b"]))
            u = bd.gelu(self.lin(x, L["Wi"]) + self.t(L["bi"])).astype(np.float32)
            x = self.ln(x + (self.lin(u, L["Wo2"]) + self.t(L["bo2"])), self.t(L["ln2_w"]), self.t(L["ln2_b"]))
        return x

    def pooled(self, ids):
        """Mean over every token of the (unpadded) label sequence, [CLS] and [SEP] included."""
        return self.encode(ids).mean(0)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


class Span:
    __slots__ = ("start", "end", "label", "score", "cls")

    def __init__(self, start, end, label, score, cls):
        self.start, self.end, self.label, self.score, self.cls = start, end, label, score, cls

    def tup(self):
        return (self.start, self.end, self.label, self.score)


def has_overlapping(a, b, multi_label=False):
    if (a.start, a.end) == (b.start, b.end):
        return not multi_label
    return not (a.start > b.end or b.start > a.end)


def has_overlapping_nested(a, b, multi_label=False):
    if (a.start, a.end) == (b.start, b.end):
        return not multi_label
    nested = (a.start <= b.start and a.end >= b.end) or (b.start <= a.start and b.end >= a.end)
    return not ((a.start > b.end or b.start > a.end) or nested)


def greedy_search(spans, flat_ner=True, multi_label=False):
    """gliner decoding/decoder.py BaseDecoder.greedy_search, same tie-breaks (two stable sorts)."""
    ov = has_overlapping if flat_ner else has_overlapping_nested
    kept = []
    for sp in sorted(spans, key=lambda s: -s.score):
        if not any(ov(sp, k, multi_label) for k in kept):
            kept.append(sp)
    kept.sort(key=lambda s: s.start)
    return kept


def load(model_dir, **kw):
    """GRef or GRefBi, by the packed manifest's gliner.encoder_kind."""
    m = json.loads((Path(model_dir) / "manifest.json").read_text())
    cls = GRefBi if m["gliner"].get("encoder_kind", "uni") == "bi" else GRef
    return cls(model_dir, m=m, **kw)


class GRef(Ref):
    """Uni-encoder. Components are attributes so a caller can swap one for another pack's:
    `tenc` (text encoder: .hidden), `hsrc` (head weights: .m['ghead'], .lin, .t), and for GRefBi `lenc`."""

    def __init__(self, model_dir, factorised=True, m=None, data=None):
        model_dir = Path(model_dir)
        if m is None:
            m = json.loads((model_dir / "manifest.json").read_text())
        super().__init__(m, data if data is not None else (model_dir / "weights.bin").read_bytes())
        g = m["gliner"]
        self.g = g
        self.bi = g.get("encoder_kind", "uni") == "bi"
        if self.bi != isinstance(self, GRefBi):
            raise TypeError("use gref.load(): the manifest's encoder_kind does not match the class")
        if g["span_mode"] not in SPAN_MODES:
            raise NotImplementedError(f"span_mode {g['span_mode']!r}")
        if g["subtoken_pooling"] not in POOLING:
            raise NotImplementedError(f"subtoken_pooling {g['subtoken_pooling']!r}")
        self.factorised = factorised
        self.tenc = self
        self.hsrc = self
        self.timing = dict.fromkeys(STAGES, 0.0)
        from tokenizers import Tokenizer, pre_tokenizers
        tok = Tokenizer.from_file(str(model_dir / g["tokenizer"]))
        st = json.loads(tok.pre_tokenizer.__getstate__())
        if st.get("type") != "ByteLevel":
            raise NotImplementedError(f"pre-tokenizer {st.get('type')}")
        st.pop("type")
        st["add_prefix_space"] = bool(g["prefix_space"])
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(**st)
        tok.no_truncation()
        tok.no_padding()
        self.tok = tok
        self.word_re = re.compile(g["word_regex"])
        sp = g["special"]
        self.cls_id, self.sep_id = sp["cls"], sp["sep"]
        self.ent_id = g["class_token_index"]
        self._wcache = {}
        if not self.bi and (self.word_ids(g["ent_token"]) != [self.ent_id]
                            or self.word_ids(g["sep_token"]) != [sp["prompt_sep"]]):
            raise ValueError("tokenizer does not map the prompt tokens to the manifest's ids")
        self.T = float(g["logit_temperature"])
        self.b = float(g["logit_bias"])

    def clock(self, stage, t0):
        t1 = time.perf_counter()
        self.timing[stage] += t1 - t0
        return t1

    # ---------------------------------------------------------------- input construction
    def split(self, text):
        return [(mt.group(), mt.start(), mt.end()) for mt in self.word_re.finditer(text)]

    def word_ids(self, w):
        ids = self._wcache.get(w)
        if ids is None:
            ids = self._wcache[w] = self.tok.encode(w, add_special_tokens=False).ids
        return ids

    @staticmethod
    def norm_labels(labels):
        if isinstance(labels, str):
            labels = [labels]
        return list(dict.fromkeys(labels))

    def prompt_words(self, labels):
        g = self.g
        prompt = []
        for lab in labels:
            prompt.append(g["ent_token"])
            if str(lab):
                prompt.append(str(lab))
        prompt.append(g["sep_token"])
        return prompt

    def build(self, words, labels):
        """words: already truncated text words. Returns dict with ids, words_mask, prompt_len."""
        g = self.g
        prompt = self.prompt_words(labels)
        ids, wids = [self.cls_id], [None]
        for i, w in enumerate(prompt + list(words)):
            sub = self.word_ids(w)
            ids += sub
            wids += [i] * len(sub)
        room = g["model_max_length"] - 2      # HF right-truncation of the content, specials kept
        ids, wids = ids[: 1 + room], wids[: 1 + room]
        ids.append(self.sep_id)
        wids.append(None)
        P = len(prompt)
        last = g["subtoken_pooling"] == "last"
        wm = []
        for t, w in enumerate(wids):
            if w is None or w < P:
                wm.append(0)
                continue
            first = wids[t - 1] != w
            nxt = wids[t + 1] if t + 1 < len(wids) else None
            sel = (nxt != w) if last else first
            wm.append(w - P + 1 if sel else 0)
        return {"ids": ids, "words_mask": wm, "prompt_len": P}

    # ---------------------------------------------------------------- representations
    def words_from_hidden(self, Hs, b, n):
        W = np.zeros((n, Hs.shape[1]), np.float32)
        wm = np.array(b["words_mask"])
        tk = np.nonzero(wm > 0)[0]
        W[wm[tk] - 1] = Hs[tk]
        return W

    def mlp(self, x, r):
        """create_projection_layer: Linear -> ReLU -> (Dropout) -> Linear."""
        h = np.maximum(self.lin(x, r["l0"]) + self.t(r["b0"]), 0)
        return self.lin(h, r["l1"]) + self.t(r["b1"])

    def deq(self, w):
        """Weight of a lin record in torch layout [N, K], float32."""
        if "f32_t" in w:
            return self.t(w["f32_t"])
        k = (id(self.d), w["q"]["offset"], "deq")
        hit = _DQ.get(k)
        if hit is None:
            hit = _DQ[k] = (self.d, (self.qf(w["q"]) * self.t(w["s"])).T)
        return hit[1]

    def lstm(self, x):
        for layer in self.m["lstm"]:
            outs = []
            for d, rev in (("fwd", False), ("bwd", True)):
                r = layer[d]
                hh = r["hidden"]
                gx = self.lin(x, r["w_ih"]) + self.t(r["b_ih"])          # [n, 4h], gates i f g o
                Whh = self.deq(r["w_hh"]).T                              # [h, 4h]
                bhh = self.t(r["b_hh"])
                h = np.zeros(hh, np.float32)
                c = np.zeros(hh, np.float32)
                out = np.zeros((len(x), hh), np.float32)
                order = range(len(x) - 1, -1, -1) if rev else range(len(x))
                for t in order:
                    gt = gx[t] + (h @ Whh + bhh)
                    i, f = sigmoid(gt[:hh]), sigmoid(gt[hh:2 * hh])
                    gg, o = np.tanh(gt[2 * hh:3 * hh]), sigmoid(gt[3 * hh:])
                    c = f * c + i * gg
                    h = o * np.tanh(c)
                    out[t] = h
                outs.append(out)
            x = np.concatenate(outs, -1)
        return x

    def reps(self, words, labels):
        """Encoder + gathers (+ BiLSTM). Returns (word_vecs [n, d], label_vecs [C, d], built inputs)."""
        g = self.g
        n = len(words)
        b = self.build(words, labels)
        t0 = time.perf_counter()
        Hs = self.tenc.hidden(np.array(b["ids"]))
        if self.m.get("projection"):
            pr = self.m["projection"]
            Hs = self.lin(Hs, pr["l"]) + self.t(pr["b"])
        ids = np.array(b["ids"])
        pos = np.nonzero(ids == self.ent_id)[0]
        if not g["embed_ent_token"]:
            pos = pos + 1
        C = len(labels)
        if len(pos) != C:
            raise ValueError(f"found {len(pos)} <<ENT>> markers for {C} labels (subtoken truncation?)")
        lab = Hs[pos]
        W = self.words_from_hidden(Hs, b, n)
        b["label_pos"] = pos.tolist()
        t0 = self.clock("text_encoder", t0)
        if self.m.get("lstm"):
            W = self.lstm(W)
            self.clock("lstm", t0)
        return W, lab, b

    # ---------------------------------------------------------------- heads
    def span_logits(self, W, lab, factorised=None):
        """markerV0 logits [n, K, C]; spans past the last word are -inf.

        Naive (library order): S = mlp_start(W), E = mlp_end(W);
          rep(s, k) = out_project(relu(concat(S[s], E[s+k]))); logit = rep . mlp_prompt(label).
        Factorised: out_project.0 on concat(x, y) is W1[:, :H] x + W1[:, H:] y + b1, and the ReLU before
        it acts elementwise, so a_i = W1a relu(S_i) + b1 and b_j = W1b relu(E_j) are per-word vectors;
        out_project.3 (W2, b2) folds into the labels: q_c = W2^T p_c, r_c = b2 . p_c. Then
          logit(s, k, c) = relu(a_s + b_{s+k}) . q_c + r_c.
        """
        hs = self.hsrc
        hd = hs.m["ghead"]
        n, K = len(W), self.g["max_width"]
        S = hs.mlp(W, hd["start"])
        E = hs.mlp(W, hd["end"])
        Pm = hs.mlp(lab, hd["prompt"])
        C = len(Pm)
        out = np.full((n, K, C), -np.inf, np.float32)
        if factorised is None:
            factorised = self.factorised
        o = hd["out"]
        if not factorised:
            for k in range(min(K, n)):
                cat = np.maximum(np.concatenate([S[: n - k], E[k:]], -1), 0)
                out[: n - k, k] = hs.mlp(cat, o) @ Pm.T
            return out
        W1 = hs.deq(o["l0"])
        Hh = S.shape[1]
        a = np.maximum(S, 0) @ W1[:, :Hh].T + hs.t(o["b0"])
        bb = np.maximum(E, 0) @ W1[:, Hh:].T
        q = Pm @ hs.deq(o["l1"])                   # [C, 4H]
        r = Pm @ hs.t(o["b1"])                     # [C]
        for k in range(min(K, n)):
            out[: n - k, k] = np.maximum(a[: n - k] + bb[k:], 0) @ q.T + r
        return out

    def token_logits(self, W, lab):
        """Token-level Scorer: logits [n, C, 3] (start, end, inside)."""
        hs = self.hsrc
        hd = hs.m["ghead"]
        Hh = W.shape[1]
        tk = hs.lin(W, hd["proj_token"]["l"]) + hs.t(hd["proj_token"]["b"])     # [n, 2H]
        lb = hs.lin(lab, hd["proj_label"]["l"]) + hs.t(hd["proj_label"]["b"])   # [C, 2H]
        n, C = len(W), len(lab)
        cat = np.concatenate([np.broadcast_to(tk[:, None, :Hh], (n, C, Hh)),
                              np.broadcast_to(lb[None, :, :Hh], (n, C, Hh)),
                              tk[:, None, Hh:] * lb[None, :, Hh:]], -1).reshape(n * C, 3 * Hh)
        return hs.mlp(cat, hd["out_mlp"]).reshape(n, C, 3)

    def logits(self, words, labels, factorised=None):
        """Raw logits for pre-split words (already truncated to max_len) and normalised labels."""
        W, lab, b = self.reps(words, labels)
        t0 = time.perf_counter()
        if self.g["span_mode"] == "markerV0":
            z = self.span_logits(W, lab, factorised)
        else:
            z = self.token_logits(W, lab)
        self.clock("span_head", t0)
        return z, b

    # ---------------------------------------------------------------- decoding
    def probs(self, z):
        x = z if (self.T == 1.0 and self.b == 0.0) else z / np.float32(self.T) + np.float32(self.b)
        return sigmoid(x)

    def decode(self, z, labels, threshold=0.5, flat_ner=True, multi_label=False):
        t0 = time.perf_counter()
        out = self._decode(z, labels, threshold, flat_ner, multi_label)
        self.clock("decode", t0)
        return out

    def _decode(self, z, labels, threshold, flat_ner, multi_label):
        n = z.shape[0]
        p = self.probs(z)
        spans = []
        if self.g["span_mode"] == "markerV0":
            s_i, k_i, c_i = np.nonzero(p > np.float32(threshold))   # C order: start, width, class
            for s, k, c in zip(s_i.tolist(), k_i.tolist(), c_i.tolist()):
                if s + k + 1 <= n:
                    spans.append(Span(s, s + k, labels[c], float(p[s, k, c]), c))
        else:
            ps, pe, pi = p[..., 0], p[..., 1], p[..., 2]
            st = list(zip(*[a.tolist() for a in np.nonzero(ps > np.float32(threshold))]))
            en = list(zip(*[a.tolist() for a in np.nonzero(pe > np.float32(threshold))]))
            pil = pi.tolist()
            psl, pel = ps.tolist(), pe.tolist()
            for s, cs in st:
                for e, ce in en:
                    if e >= s and cs == ce:
                        ins = [pil[q][cs] for q in range(s, e + 1)]
                        if any(v < threshold for v in ins):
                            continue
                        spans.append(Span(s, e, labels[cs], min(*ins, psl[s][cs], pel[e][ce]), cs))
        return greedy_search(spans, flat_ner, multi_label)

    def predict_words(self, words, labels, thresholds=(0.5,), flat_ner=True, multi_label=False):
        """Pre-split words (the benchmark path). Returns {threshold: [(start, end, label, score)]}."""
        labels = self.norm_labels(labels)
        words = list(words)[: self.g["max_len"]] or ["[PAD]"]
        z, _ = self.logits(words, labels)
        return {t: [s.tup() for s in self.decode(z, labels, t, flat_ner, multi_label)] for t in thresholds}

    def predict_entities(self, text, labels, threshold=0.5, flat_ner=True, multi_label=False, return_logits=False):
        """model.predict_entities equivalent: [{start, end, text, label, score}] with char offsets."""
        if not isinstance(text, str) or not text.strip():
            return ([], None) if return_logits else []
        labels = self.norm_labels(labels)
        sp = self.split(text)
        words = [w for w, _, _ in sp][: self.g["max_len"]] or ["[PAD]"]
        z, b = self.logits(words, labels)
        ents = []
        for s in self.decode(z, labels, threshold, flat_ner, multi_label):
            a, e = sp[s.start][1], sp[s.end][2]
            ents.append({"start": a, "end": e, "text": text[a:e], "label": s.label, "score": s.score})
        if return_logits:
            return ents, {"z": z, "words": words, **b}
        return ents


class GRefBi(GRef):
    """Bi-encoder span model (gliner BiEncoderSpanModel). `lenc` is the label encoder (a BertRef)."""

    def __init__(self, model_dir, factorised=True, m=None, data=None):
        super().__init__(model_dir, factorised, m, data)
        g = self.g
        if g["span_mode"] != "markerV0":
            raise NotImplementedError("bi-encoder with span_mode " + repr(g["span_mode"]))
        if self.m.get("lstm"):
            raise NotImplementedError("the bi-encoder forward applies no BiLSTM; a packed one is a packer error")
        self.lenc = BertRef(self.m["label_encoder"], self.d)
        from tokenizers import Tokenizer
        lt = Tokenizer.from_file(str(Path(model_dir) / g["label_tokenizer"]))
        lt.no_padding()
        lt.enable_truncation(int(g["label_max_length"]))
        self.ltok = lt
        self._lcache = {}

    def prompt_words(self, labels):
        return []

    def label_ids(self, label):
        """The label tokenizer's [CLS] ... [SEP] ids (HF call: truncation=True, padding only between labels)."""
        return self.ltok.encode(str(label)).ids

    def label_vecs(self, labels):
        """[C, H] pooled label-encoder outputs, cached per (label encoder, label string)."""
        t0 = time.perf_counter()
        out = []
        for lab in labels:
            k = (id(self.lenc), lab)
            v = self._lcache.get(k)
            if v is None:
                v = self._lcache[k] = self.lenc.pooled(self.label_ids(lab))
            out.append(v)
        self.clock("label_encoder", t0)
        return np.stack(out).astype(np.float32) if out else np.zeros((0, self.g["hidden_size"]), np.float32)

    def reps(self, words, labels):
        n = len(words)
        b = self.build(words, labels)
        t0 = time.perf_counter()
        Hs = self.tenc.hidden(np.array(b["ids"]))
        W = self.words_from_hidden(Hs, b, n)
        self.clock("text_encoder", t0)
        lab = self.label_vecs(labels)
        b["label_ids"] = [self.label_ids(lab_) for lab_ in labels]
        b["label_pos"] = None
        return W, lab, b
