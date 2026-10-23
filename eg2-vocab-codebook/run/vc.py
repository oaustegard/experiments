"""Shared pieces for eg2-vocab-codebook: the k=4 student with a full 262,144-row token table,
table codecs, and the three benchmarks (Bluesky real/synth from eg2-layer-student/run/bench.py,
text->image from eg2-layer-student/run/img_common.py).

The student's own run/model.py builds the cut model; here the full table is EG2's q4 table
(dequantized, sqrt(512) scale divided out, as model.q4_weights does) with the student's trained
rows written over the ids the 32K cut kept.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
LS = HERE.parent / "eg2-layer-student"
sys.path.insert(0, str(LS / "run"))
from model import PREV, Q4, HID, Student, dequant, embed, tokenize  # noqa: E402,F401
sys.path.insert(0, str(PREV / "run"))
from common import QUERY, Teacher, boot_ci, overlap  # noqa: E402,F401

WORK = HERE / ".work"
CUT_ROWS = LS / "out" / "vocab" / "k32000" / "rows.json"


def teacher_table():
    """EG2 q4 token table as the HF module stores it (scale divided out), fp32 262,144 x 512."""
    p = WORK / "eg2_table.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(Q4))
    I = {i.name: i for i in m.graph.initializer}
    g = lambda s: numpy_helper.to_array(I["model_embed_tokens_weight" + s])  # noqa: E731
    q, sc, zp = g("_quant"), g("_scales"), g("_zp")
    T = np.empty((len(q), HID), np.float32)
    for s in range(0, len(q), 16384):
        T[s:s + 16384] = dequant(q[s:s + 16384], sc[s:s + 16384], zp[s:s + 16384], len(q[s:s + 16384]), HID)
    T /= np.float32(HID ** 0.5)
    WORK.mkdir(exist_ok=True)
    np.save(p, T)
    return np.load(p, mmap_mode="r")


def cut_rows():
    return np.array(json.loads(CUT_ROWS.read_text()))


def full_student(sd, table=None):
    """k=4 Student with all 262,144 rows. sd: a cut-student state dict (41,741-row table).
    table: the full table to install (default: EG2's rows with the trained cut rows written over)."""
    if table is None:
        table = np.array(teacher_table())
        table[cut_rows()] = sd["tm.embed_tokens.weight"].numpy()
    s = Student(4, rows=cut_rows())  # cheap path for the layer weights; table replaced below
    s.load_state_dict(sd, strict=True)
    emb = torch.nn.Embedding(len(table), HID)
    emb.weight.data = torch.from_numpy(np.ascontiguousarray(table, np.float32))
    old = s.tm.embed_tokens
    old.weight = emb.weight
    old.num_embeddings = len(table)
    s.tm.config.vocab_size = len(table)
    return s


def rows_student(sd, rows, table):
    """k=4 Student over an arbitrary sorted row set (a pruned vocabulary), table = its rows."""
    s = Student(4, rows=cut_rows())
    s.load_state_dict(sd, strict=True)
    emb = torch.nn.Embedding(len(rows), HID)
    emb.weight.data = torch.from_numpy(np.ascontiguousarray(table, np.float32))
    s.tm.embed_tokens.weight = emb.weight
    s.tm.embed_tokens.num_embeddings = len(rows)
    s.tm.config.vocab_size = len(rows)
    return s


def cut_student(sd, table=None):
    s = Student(4, rows=cut_rows())
    s.load_state_dict(sd, strict=True)
    if table is not None:
        s.tm.embed_tokens.weight.data = torch.from_numpy(np.ascontiguousarray(table, np.float32))
    return s


# ---------------- benchmarks ----------------
class Benches:
    def __init__(self, image=True):
        from bench import Bench
        self.T = Teacher()
        self.b = Bench()
        from tokenizers import Tokenizer
        self.cut_tok = Tokenizer.from_file(str(LS / "out" / "vocab" / "k32000" / "tokenizer.json"))
        self.texts = {t: self.b.texts(t) for t in ("real_test", "synth_test")}
        self.image = image
        if image:
            from img_common import IMG, ImageIndex, eval_queries, is_test, rows
            self.I = ImageIndex()
            self.itexts, self.itarget = eval_queries()
            self.E = np.load(IMG / "teacher_eval_all.npy")
            assert json.loads((IMG / "teacher_eval_all.json").read_text()) == self.itexts
            cid = {r["i"]: r["cid"] for r in rows()}
            self.itest = np.array([is_test(cid[t]) for t in self.itarget])
            _, self.top_e = self.I.rank_and_top(self.E, self.itarget)
        self._enc = {}

    def enc(self, which, vocab):
        """vocab: 'cut', 'full', or a (name, Tokenizer) pair for another pruned vocabulary."""
        key = (which, vocab if isinstance(vocab, str) else vocab[0])
        if key not in self._enc:
            texts = self.itexts if which == "image" else self.texts[which]
            tok = self.cut_tok if vocab == "cut" else None if vocab == "full" else vocab[1]
            self._enc[key] = tokenize(self.T, texts, remap=tok)
        return self._enc[key]

    def run(self, model, vocab, per=False):
        """vocab: 'cut' (pruned tokenizer, model has 41,741 rows) or 'full'."""
        P = {t: embed(model, self.enc(t, vocab)) for t in ("real_test", "synth_test")}
        r, pq = self.b.score(P)
        out = {"real_overlap": r["real_test"]["overlap@10"][0], "synth_overlap": r["synth_test"]["overlap@10"][0],
               "synth_hit10": r["synth_test"]["hit@10"][0]}
        perq = {"real_overlap": pq["real_test"], "synth_overlap": pq["synth_test"]}
        top = self.b.ix.topk(P["synth_test"])
        perq["synth_hit10"] = (top == self.b.src[:, None]).any(1).astype(float)
        if self.image:
            Q = embed(model, self.enc("image", vocab))
            rank, top = self.I.rank_and_top(Q, self.itarget)
            ov = overlap(top, self.top_e)
            h10 = (rank < 10).astype(float)
            m = self.itest
            out.update({"img_overlap": float(ov.mean()), "img_hit10": float(h10.mean()),
                        "img_test_overlap": float(ov[m].mean()), "img_test_hit10": float(h10[m].mean())})
            perq.update({"img_overlap": ov, "img_hit10": h10})
        out = {k: round(float(v), 4) for k, v in out.items()}
        return (out, perq) if per else out
