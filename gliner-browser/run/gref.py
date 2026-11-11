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
"""
import importlib.util
import json
import re
import sys
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
Ref = bd.Ref

SPAN_MODES = ("markerV0", "token_level")
POOLING = ("first", "last")


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


class GRef(Ref):
    def __init__(self, model_dir, factorised=True):
        model_dir = Path(model_dir)
        m = json.loads((model_dir / "manifest.json").read_text())
        super().__init__(m, (model_dir / "weights.bin").read_bytes())
        g = m["gliner"]
        self.g = g
        if g["span_mode"] not in SPAN_MODES:
            raise NotImplementedError(f"span_mode {g['span_mode']!r}")
        if g["subtoken_pooling"] not in POOLING:
            raise NotImplementedError(f"subtoken_pooling {g['subtoken_pooling']!r}")
        self.factorised = factorised
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
        if self.word_ids(g["ent_token"]) != [self.ent_id] or self.word_ids(g["sep_token"]) != [sp["prompt_sep"]]:
            raise ValueError("tokenizer does not map the prompt tokens to the manifest's ids")
        self.T = float(g["logit_temperature"])
        self.b = float(g["logit_bias"])

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

    def build(self, words, labels):
        """words: already truncated text words. Returns dict with ids, words_mask, prompt_len."""
        g = self.g
        prompt = []
        for lab in labels:
            prompt.append(g["ent_token"])
            if str(lab):
                prompt.append(str(lab))
        prompt.append(g["sep_token"])
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
    def mlp(self, x, r):
        """create_projection_layer: Linear -> ReLU -> (Dropout) -> Linear."""
        h = np.maximum(self.lin(x, r["l0"]) + self.t(r["b0"]), 0)
        return self.lin(h, r["l1"]) + self.t(r["b1"])

    def deq(self, w):
        """Weight of a lin record in torch layout [N, K], float32."""
        if "f32_t" in w:
            return self.t(w["f32_t"])
        return (self.t(w["q"], np.int8).astype(np.float32) * self.t(w["s"])).T

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
        Hs = self.hidden(np.array(b["ids"]))
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
        W = np.zeros((n, Hs.shape[1]), np.float32)
        wm = np.array(b["words_mask"])
        tk = np.nonzero(wm > 0)[0]
        W[wm[tk] - 1] = Hs[tk]
        b["label_pos"] = pos.tolist()
        if self.m.get("lstm"):
            W = self.lstm(W)
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
        hd = self.m["ghead"]
        n, K = len(W), self.g["max_width"]
        S = self.mlp(W, hd["start"])
        E = self.mlp(W, hd["end"])
        Pm = self.mlp(lab, hd["prompt"])
        C = len(Pm)
        out = np.full((n, K, C), -np.inf, np.float32)
        if factorised is None:
            factorised = self.factorised
        o = hd["out"]
        if not factorised:
            for k in range(min(K, n)):
                cat = np.maximum(np.concatenate([S[: n - k], E[k:]], -1), 0)
                out[: n - k, k] = self.mlp(cat, o) @ Pm.T
            return out
        W1 = self.deq(o["l0"])
        Hh = S.shape[1]
        a = np.maximum(S, 0) @ W1[:, :Hh].T + self.t(o["b0"])
        bb = np.maximum(E, 0) @ W1[:, Hh:].T
        q = Pm @ self.deq(o["l1"])                 # [C, 4H]
        r = Pm @ self.t(o["b1"])                   # [C]
        for k in range(min(K, n)):
            out[: n - k, k] = np.maximum(a[: n - k] + bb[k:], 0) @ q.T + r
        return out

    def token_logits(self, W, lab):
        """Token-level Scorer: logits [n, C, 3] (start, end, inside)."""
        hd = self.m["ghead"]
        Hh = W.shape[1]
        tk = self.lin(W, hd["proj_token"]["l"]) + self.t(hd["proj_token"]["b"])     # [n, 2H]
        lb = self.lin(lab, hd["proj_label"]["l"]) + self.t(hd["proj_label"]["b"])   # [C, 2H]
        n, C = len(W), len(lab)
        cat = np.concatenate([np.broadcast_to(tk[:, None, :Hh], (n, C, Hh)),
                              np.broadcast_to(lb[None, :, :Hh], (n, C, Hh)),
                              tk[:, None, Hh:] * lb[None, :, Hh:]], -1).reshape(n * C, 3 * Hh)
        return self.mlp(cat, hd["out_mlp"]).reshape(n, C, 3)

    def logits(self, words, labels, factorised=None):
        """Raw logits for pre-split words (already truncated to max_len) and normalised labels."""
        W, lab, b = self.reps(words, labels)
        if self.g["span_mode"] == "markerV0":
            z = self.span_logits(W, lab, factorised)
        else:
            z = self.token_logits(W, lab)
        return z, b

    # ---------------------------------------------------------------- decoding
    def probs(self, z):
        x = z if (self.T == 1.0 and self.b == 0.0) else z / np.float32(self.T) + np.float32(self.b)
        return sigmoid(x)

    def decode(self, z, labels, threshold=0.5, flat_ner=True, multi_label=False):
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
