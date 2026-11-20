"""Port of the dictionary path of Tesseract 5.3.4's LSTM beam search (RecodeBeamSearch + Dict).

Sources, read from the 5.3.4 tag:
  src/lstm/recodebeam.{h,cpp}       every dawg branch of the search
  src/lstm/lstmrecognizer.cpp       LoadDictionary, kDictRatio 2.25, kCertOffset -0.085
  src/ccmain/linerec.cpp            worst_dict_cert = kWorstDictCertainty / kCertaintyScale
                                    = -25.0f / 7.0f
  src/dict/dict.cpp, dict.h         LoadLSTM, FinishLoad, default_dawgs, def_letter_is_okay,
                                    GetStartingNode, char_for_dawg, IsSpaceDelimitedLang
  src/dict/dawg.{h,cpp}             SquishedDawg serialization, edge_char_of, kDawgSuccessors
  src/dict/dawg_cache.cpp           dawg type and permuter of each lstm-*-dawg component
  src/ccutil/unicharset.{h,cpp}     the properties the search reads (isdigit, script ids)

What LSTMRecognizer::LoadDictionary builds for a .traineddata with LSTM dawgs:
  dawgs_ = [lstm-punc-dawg   (DAWG_TYPE_PUNCTUATION, PUNC_PERM),
            lstm-word-dawg   (DAWG_TYPE_WORD,        SYSTEM_DAWG_PERM),
            lstm-number-dawg (DAWG_TYPE_NUMBER,      NUMBER_PERM)]
in that order, each only if present. No document-words trie (only Dict::Load adds one) and no
user words or patterns (user_words_file etc. are empty unless set by config). The Dict
parameters keep their defaults: the LSTM's Dict registers them on the recognizer's own CCUtil,
which no config file reaches.

RecodeBeamSearch then doubles every beam into a dawg half (indices 30..59) and a non-dawg half
(0..29), keeps one best "initial dawg" node per continuation per step, and at the end takes the
best node from either half, a dawg node only if its last unichar is a valid word end or a space.
Word scores are float sums of float certainties exactly as in beam.py; the dawg half does not
multiply by dict_ratio, the non-dawg half does, which is what makes dictionary words win.

The no-dictionary search lives in beam.py and is not touched by this module; with
dictionary=None this module's search reduces to the same arithmetic (checked against beam.py
by run/check_beam_dict.py).

Public:
  load_dict(path_or_model) -> LSTMDict or None    (None when the model has no LSTM dawgs)
  decode(probs, model, dictionary) -> list of unichar ids
  decode_full(probs, model, dictionary) -> dict with ids, xcoords, per-id node flags, words
"""

import math
import struct

import numpy as np

from . import traineddata
from .beam import (BEAM_WIDTHS, DICT_RATIO, INVALID, MASK64, MAX_CODE_LEN, MIN_CERT, NC_ANY,
                   NC_NO_DUP, NC_ONLY_DUP, NUM_LENGTHS, TN_ALSO_RAN, TN_TOP2, TN_TOPN, _Heap,
                   _Recoder, cert_of, f32)

# ---------------------------------------------------------------- constants

UNICHAR_SPACE = 0                  # unicharset.h SpecialUnicharCodes
NO_EDGE = -1                       # dawg.h NO_EDGE (int64 all ones)
PATTERN_UNICHAR_ID = 0             # Dawg::kPatternUnicharID
DAWG_MAGIC = 42                    # Dawg::kDawgMagicNumber
# ratngs.h PermuterType
NO_PERM, PUNC_PERM, TOP_CHOICE_PERM = 0, 1, 2
NUMBER_PERM, SYSTEM_DAWG_PERM, COMPOUND_PERM = 6, 8, 12
PERM_NAMES = {0: "NO_PERM", 1: "PUNC_PERM", 2: "TOP_CHOICE_PERM", 3: "LOWER_CASE_PERM",
              4: "UPPER_CASE_PERM", 5: "NGRAM_PERM", 6: "NUMBER_PERM", 7: "USER_PATTERN_PERM",
              8: "SYSTEM_DAWG_PERM", 9: "DOC_DAWG_PERM", 10: "USER_DAWG_PERM",
              11: "FREQ_DAWG_PERM", 12: "COMPOUND_PERM"}
# dawg.h DawgType and kDawgSuccessors[type][other_type]
DAWG_PUNC, DAWG_WORD, DAWG_NUMBER, DAWG_PATTERN = 0, 1, 2, 3
DAWG_SUCCESSORS = (
    (False, True, True, False),    # punctuation -> word, number
    (True, False, False, False),   # word -> punctuation
    (True, False, False, False),   # number -> punctuation
    (False, False, False, False),  # pattern
)
# traineddata component index -> (dawg type, permuter): dawg_cache.cpp DawgLoader::Load
LSTM_DAWGS = ((18, DAWG_PUNC, PUNC_PERM), (19, DAWG_WORD, SYSTEM_DAWG_PERM),
              (20, DAWG_NUMBER, NUMBER_PERM))
# worst_dict_cert: float(-25.0f / 7.0f), passed as double and narrowed back to float
WORST_DICT_CERT = float(np.float32(-25.0) / np.float32(7.0))
NUM_BEAMS = 2 * 3 * NUM_LENGTHS    # kNumBeams = 2 * NC_COUNT * kNumLengths
BW0 = BEAM_WIDTHS[0]


# ---------------------------------------------------------------- SquishedDawg

class SquishedDawg:
    """SquishedDawg::read_squished_dawg plus the lookups def_letter_is_okay makes.

    Serialized form (little-endian): int16 magic (42), int32 unicharset_size, int32 num_edges,
    uint64 edges[num_edges]. Each edge record packs, low bits first: the unichar id
    (flag_start_bit bits), three flags (marker = last edge of its node, direction, word end) and
    the next node (the index of that node's first edge; 0 means no continuation).
    """

    def __init__(self, buf, dawg_type, permuter, lang=""):
        magic, usize, n = struct.unpack_from("<hii", buf, 0)
        if magic != DAWG_MAGIC:
            raise ValueError(f"bad dawg magic {magic}")
        if n <= 0:
            raise ValueError("empty dawg")
        if len(buf) < 10 + 8 * n:
            raise ValueError("truncated dawg")
        self.type = dawg_type
        self.perm = permuter
        self.lang = lang
        self.unicharset_size = usize
        # Dawg::init: ceil(log(size + 1) / log(2)) in double, as the C++ computes it
        fsb = math.ceil(math.log(usize + 1.0) / math.log(2.0))
        self.flag_start_bit = fsb
        nnsb = fsb + 3
        letter_mask = (1 << fsb) - 1
        next_node_mask = (MASK64 << nnsb) & MASK64
        e = np.frombuffer(buf, dtype="<u8", count=n, offset=10)
        self.num_edges = n
        self.raw = e
        self.letter = (e & np.uint64(letter_mask)).astype(np.int64).tolist()
        self.last = ((e >> np.uint64(fsb)) & np.uint64(1)).astype(bool).tolist()
        self.backward = ((e >> np.uint64(fsb + 1)) & np.uint64(1)).astype(bool).tolist()
        self.eow = ((e >> np.uint64(fsb + 2)) & np.uint64(1)).astype(bool).tolist()
        self.next = (e >> np.uint64(nnsb)).astype(np.int64).tolist()
        self.occupied = (e != np.uint64(next_node_mask)).tolist()
        self.n0 = self._num_forward_edges(0)
        self._cache = {}

    def _num_forward_edges(self, node):
        edge = node
        num = 0
        if self.occupied[edge] and not self.backward[edge]:
            while True:
                num += 1
                if self.last[edge]:
                    break
                edge += 1
        return num

    def _given_greater(self, word_end, uid, edge):
        """given_greater_than_edge_rec(NO_EDGE, word_end, uid, edges_[edge])."""
        cur = self.letter[edge]
        if uid == cur and (not word_end or self.eow[edge]):
            return 0                # edge_rec_match with next_node == NO_EDGE
        if uid > cur:
            return 1
        if uid == cur:
            # next_node (NO_EDGE = -1) is never > or == the record's next node (>= 0)
            pass
        return -1

    def edge_char_of(self, node, uid, word_end):
        key = (node, uid, word_end)
        r = self._cache.get(key)
        if r is None:
            r = self._edge_char_of(node, uid, word_end)
            self._cache[key] = r
        return r

    def _edge_char_of(self, node, uid, word_end):
        if node == 0:              # binary search over the root node's forward edges
            start = 0
            end = self.n0 - 1
            while start <= end:
                edge = (start + end) >> 1
                c = self._given_greater(word_end, uid, edge)
                if c == 0:
                    return edge
                if c == 1:
                    start = edge + 1
                else:
                    end = edge - 1
        else:                      # linear search through the node's edges
            edge = node
            if edge != NO_EDGE and self.occupied[edge]:
                letter, eow, last = self.letter, self.eow, self.last
                while True:
                    if letter[edge] == uid and (not word_end or eow[edge]):
                        return edge
                    if last[edge]:
                        break
                    edge += 1
        return NO_EDGE

    def next_node(self, edge):
        if edge < 0:
            raise IndexError("next_node(NO_EDGE): undefined in the C++")
        return self.next[edge]

    def end_of_word(self, edge):
        if edge < 0:
            raise IndexError("end_of_word(NO_EDGE): undefined in the C++")
        return self.eow[edge]

    def words(self, unicharset, limit=None):
        """Iterate the words of the dawg as strings (debug helper, Dawg::iterate_words)."""
        out = []

        def rec(node, prefix):
            if limit is not None and len(out) >= limit:
                return
            edge = node
            if not self.occupied[edge] or self.backward[edge]:
                return
            while True:
                s = prefix + unicharset[self.letter[edge]] if self.letter[edge] < len(
                    unicharset) else prefix + f"<{self.letter[edge]}>"
                if self.eow[edge]:
                    out.append(s)
                if self.next[edge] != 0:
                    rec(self.next[edge], s)
                if self.last[edge]:
                    break
                edge += 1

        rec(0, "")
        return out


def starting_node(dawg, edge_ref):
    """Dict::GetStartingNode."""
    if edge_ref == NO_EDGE:
        return 0
    node = dawg.next_node(edge_ref)
    return NO_EDGE if node == 0 else node


# ---------------------------------------------------------------- unicharset properties

def parse_unicharset(text):
    """The UNICHARSET fields the dictionary search reads: per id (isdigit, script id), plus the
    script table (index 0 is always "NULL"). Mirrors load_via_fgets: line 0 is the count, then
    one line per id, `unichar hexprops [metrics] script other_case ...`, with the metrics field
    optional; the script defaults to "NULL"."""
    lines = text.split("\n")
    n = int(lines[0].split()[0])
    scripts = ["NULL"]
    isdigit = []
    script_id = []
    for line in lines[1:1 + n]:
        tok = line.split()
        props = int(tok[1], 16) if len(tok) > 1 else 0
        script = "NULL"
        if len(tok) > 2:
            script = tok[3] if "," in tok[2] and len(tok) > 3 else tok[2]
            if "," in script:
                script = "NULL"
        if script not in scripts:
            scripts.append(script)
        isdigit.append(bool(props & 0x8))
        script_id.append(scripts.index(script))
    if len(isdigit) != n:
        raise ValueError(f"unicharset declares {n} entries, has {len(isdigit)}")

    def sid(name):
        return scripts.index(name) if name in scripts else 0

    return {"isdigit": isdigit, "script_id": script_id, "scripts": scripts,
            "han": sid("Han"), "hiragana": sid("Hiragana"), "katakana": sid("Katakana"),
            "thai": sid("Thai"), "hangul": sid("Hangul")}


# ---------------------------------------------------------------- Dict

class LSTMDict:
    """The parts of tesseract::Dict that RecodeBeamSearch calls, after LoadLSTM + FinishLoad."""

    def __init__(self, comps, lang=""):
        uni = parse_unicharset(comps[21].decode("utf-8"))
        self.isdigit = uni["isdigit"]
        non_sd = {uni[k] for k in ("han", "thai", "hangul", "hiragana", "katakana")}
        # UNICHARSET::IsSpaceDelimited: script not Han/Thai/Hangul/Hiragana/Katakana. A missing
        # script has id 0, the "NULL" script, so an id whose script is "NULL" is not
        # space-delimited (no entry of eng/fra/grc has the NULL script).
        self.space_delimited = [s not in non_sd for s in uni["script_id"]]
        # Dict::IsSpaceDelimitedLang
        self.space_delimited_lang = not (uni["han"] > 0 or uni["katakana"] > 0 or uni["thai"] > 0)
        self.dawgs = []
        self.punc = None
        for idx, dtype, perm in LSTM_DAWGS:
            if idx in comps and len(comps[idx]):
                d = SquishedDawg(comps[idx], dtype, perm, lang)
                if d.unicharset_size != len(self.isdigit):
                    raise ValueError(f"dawg {idx} unicharset size {d.unicharset_size} != "
                                     f"{len(self.isdigit)}")
                self.dawgs.append(d)
                if dtype == DAWG_PUNC:
                    self.punc = d
        # FinishLoad: successors_[i] = indices j whose type may follow dawg i (same lang)
        self.successors = [[j for j, o in enumerate(self.dawgs)
                            if d.lang == o.lang and DAWG_SUCCESSORS[d.type][o.type]]
                           for d in self.dawgs]
        self._default = self._default_dawgs()

    def is_space_delimited(self, uid):
        if uid == INVALID:
            return True
        return self.space_delimited[uid]

    def _default_dawgs(self):
        """Dict::default_dawgs(vec, suppress_patterns=false). DawgPosition tuples are
        (dawg_index, dawg_ref, punc_index, punc_ref, back_to_punc)."""
        punc_ok = (self.punc is not None and
                   self.punc.edge_char_of(0, PATTERN_UNICHAR_ID, True) != NO_EDGE)
        out = []
        for i, d in enumerate(self.dawgs):
            if d.type == DAWG_PUNC:
                out.append((-1, NO_EDGE, i, NO_EDGE, False))
            elif not punc_ok or not DAWG_SUCCESSORS[DAWG_PUNC][d.type]:
                out.append((i, NO_EDGE, -1, NO_EDGE, False))
        return tuple(out)

    def default_dawgs(self):
        return self._default

    def char_for_dawg(self, uid, dawg):
        if dawg.type == DAWG_NUMBER and self.isdigit[uid]:
            return PATTERN_UNICHAR_ID
        return uid

    def letter_is_okay(self, active, uid):
        """Dict::def_letter_is_okay(dawg_args, unicharset, uid, word_end=false) with
        dawg_args.permuter starting at NO_PERM, as ContinueDawg constructs it.
        Returns (permuter, updated_dawgs tuple, valid_end)."""
        if uid == PATTERN_UNICHAR_ID or uid == INVALID:
            return NO_PERM, (), False
        word_end = False
        curr = NO_PERM
        upd = []
        valid_end = False
        dawgs = self.dawgs
        for pos in active:
            dawg_index, dawg_ref, punc_index, punc_ref, back = pos
            punc_dawg = dawgs[punc_index] if punc_index >= 0 else None
            dawg = dawgs[dawg_index] if dawg_index >= 0 else None
            if dawg is None and punc_dawg is None:
                continue
            if dawg is None:
                # In the punctuation dawg; no core dawg chosen yet.
                punc_node = starting_node(punc_dawg, punc_ref)
                pte = punc_dawg.edge_char_of(punc_node, PATTERN_UNICHAR_ID, word_end)
                if pte != NO_EDGE:
                    for s in self.successors[punc_index]:
                        sd = dawgs[s]
                        de = sd.edge_char_of(0, self.char_for_dawg(uid, sd), word_end)
                        if de != NO_EDGE:
                            p = (s, de, punc_index, pte, False)
                            if p not in upd:
                                upd.append(p)
                            if sd.perm > curr:
                                curr = sd.perm
                            if sd.end_of_word(de) and punc_dawg.end_of_word(pte):
                                valid_end = True
                pe = punc_dawg.edge_char_of(punc_node, uid, word_end)
                if pe != NO_EDGE:
                    p = (-1, NO_EDGE, punc_index, pe, False)
                    if p not in upd:
                        upd.append(p)
                    if PUNC_PERM > curr:
                        curr = PUNC_PERM
                    if punc_dawg.end_of_word(pe):
                        valid_end = True
                continue
            if punc_dawg is not None and dawg.end_of_word(dawg_ref):
                # The core word may end here: try to continue in the punctuation dawg.
                punc_node = starting_node(punc_dawg, punc_ref)
                pe = (NO_EDGE if punc_node == NO_EDGE
                      else punc_dawg.edge_char_of(punc_node, uid, word_end))
                if pe != NO_EDGE:
                    p = (dawg_index, dawg_ref, punc_index, pe, True)
                    if p not in upd:
                        upd.append(p)
                    if dawg.perm > curr:
                        curr = dawg.perm
                    if punc_dawg.end_of_word(pe):
                        valid_end = True
            if back:
                continue
            if dawg.type == DAWG_PATTERN:
                raise NotImplementedError("pattern dawgs are not loaded by LoadLSTM")
            node = starting_node(dawg, dawg_ref)
            edge = (NO_EDGE if node == NO_EDGE
                    else dawg.edge_char_of(node, self.char_for_dawg(uid, dawg), word_end))
            if edge != NO_EDGE:
                if word_end and punc_dawg is not None and not punc_dawg.end_of_word(punc_ref):
                    continue
                if dawg.perm > curr:
                    curr = dawg.perm
                if dawg.end_of_word(edge) and (punc_dawg is None or
                                               punc_dawg.end_of_word(punc_ref)):
                    valid_end = True
                p = (dawg_index, edge, punc_index, punc_ref, False)
                if p not in upd:
                    upd.append(p)
        # dawg_args->permuter was NO_PERM, so it always takes curr_perm
        return curr, tuple(upd), valid_end


_DICT_CACHE = {}


def load_dict(model_or_path, lang=None):
    """LSTMRecognizer::LoadDictionary for a .traineddata: an LSTMDict, or None when the file
    carries no LSTM dawg (FinishLoad fails and Tesseract runs without a dictionary). Cached per
    path. `lang` only feeds the same-language test of FinishLoad, which every dawg of one file
    passes."""
    path = model_or_path if isinstance(model_or_path, str) else model_or_path.path
    key = (path, lang)
    if key not in _DICT_CACHE:
        buf = open(path, "rb").read()
        comps = traineddata.components(buf)
        d = None
        if any(idx in comps and len(comps[idx]) for idx, _, _ in LSTM_DAWGS):
            d = LSTMDict(comps, lang or "")
        _DICT_CACHE[key] = d
    return _DICT_CACHE[key]


# ---------------------------------------------------------------- beam search

class _Node:
    """RecodeNode minus certainty (it never steers the search or the ids)."""

    __slots__ = ("code", "uid", "perm", "sod", "sow", "eow", "dup", "score", "prev", "dawgs",
                 "hash")

    def __init__(self, code, uid, perm, sod, sow, eow, dup, score, prev, dawgs, hsh):
        self.code = code
        self.uid = uid
        self.perm = perm
        self.sod = sod             # start_of_dawg
        self.sow = sow             # start_of_word
        self.eow = eow             # end_of_word
        self.dup = dup
        self.score = score
        self.prev = prev
        self.dawgs = dawgs         # tuple of DawgPosition tuples, or None (nullptr)
        self.hash = hsh


class _DictSearch:
    def __init__(self, recoder, null_char, simple_text, dictionary):
        self.rc = recoder
        self.null = null_char
        self.simple = simple_text
        self.dict = dictionary
        self.space_delimited = dictionary is None or dictionary.space_delimited_lang
        self.nbeams = NUM_BEAMS if dictionary is not None else NUM_BEAMS // 2
        self.beams = []
        self.outs = None
        self.flags = None
        self.top_code = -1
        self.second_code = -1
        self.best_init = None

    # ---- ComputeTopN, identical to beam._Search.compute_top_n
    def compute_top_n(self, outs, top_n):
        n = len(outs)
        flags = [TN_ALSO_RAN] * n
        self.top_code = -1
        self.second_code = -1
        heap = _Heap()
        for i in range(n):
            if len(heap.h) < top_n or outs[i] > heap.h[0][0]:
                heap.push([outs[i], i])
                if len(heap.h) > top_n:
                    heap.pop()
        while heap.h:
            e = heap.pop()
            if len(heap.h) > 1:
                flags[e[1]] = TN_TOPN
            else:
                flags[e[1]] = TN_TOP2
                if not heap.h:
                    self.top_code = e[1]
                else:
                    self.second_code = e[1]
        flags[self.null] = TN_TOP2
        self.flags = flags
        self.outs = outs

    def code_hash(self, code, dup, prev):
        h = 0 if prev is None else prev.hash
        if not dup and code != self.null:
            nc = self.rc.code_range
            carry = (((h >> 32) * nc) >> 32) & MASK64
            h = (h * nc) & MASK64
            h = (h + carry) & MASK64
            h = (h + code) & MASK64
        return h

    # ---- UpdateHeapIfMatched: code, code_hash, permuter and start_of_dawg must all match
    @staticmethod
    def update_if_matched(node, heap):
        h = heap.h
        for idx in range(len(h)):
            pair = h[idx]
            old = pair[1]
            if (old.code == node.code and old.hash == node.hash and old.perm == node.perm
                    and old.sod == node.sod):
                if node.score > old.score:
                    pair[1] = node
                    pair[0] = node.score
                    heap.reshuffle(idx)
                return True
        return False

    # ---- PushHeapIfBetter (the field-wise overload)
    def push_heap(self, max_size, code, uid, perm, sod, sow, eow, dup, cert, prev, dawgs, heap):
        score = cert if prev is None else f32(cert + prev.score)
        h = heap.h
        if len(h) < max_size or score > h[0][0]:
            node = _Node(code, uid, perm, sod, sow, eow, dup, score, prev, dawgs,
                         self.code_hash(code, dup, prev))
            if self.update_if_matched(node, heap):
                return
            heap.push([score, node])
            if len(h) > max_size:
                heap.pop()

    # ---- PushHeapIfBetter (the node overload, for best_initial_dawgs_)
    def push_node(self, max_size, node, heap):
        h = heap.h
        if len(h) < max_size or node.score > h[0][0]:
            if self.update_if_matched(node, heap):
                return
            heap.push([node.score, node])
            if len(h) > max_size:
                heap.pop()

    # ---- PushInitialDawgIfBetter
    def push_initial_dawg(self, code, uid, perm, start, end, cert, cont, prev):
        score = cert if prev is None else f32(cert + prev.score)
        best = self.best_init[cont]
        if best is None or score > best.score:
            self.best_init[cont] = _Node(code, uid, perm, True, start, end, False, score, prev,
                                         self.dict.default_dawgs(),
                                         self.code_hash(code, False, prev))

    # ---- PushDupOrNoDawgIfBetter
    def push_dup(self, length, dup, code, uid, cert, use_dawgs, cont, prev, step):
        if use_dawgs:
            if cert > WORST_DICT_CERT:
                self.push_heap(BEAM_WIDTHS[length], code, uid,
                               prev.perm if prev is not None else NO_PERM, False, False, False,
                               dup, cert, prev, None, step[(3 + cont) * NUM_LENGTHS + length])
        else:
            cert = f32(cert * DICT_RATIO)
            if cert >= MIN_CERT or code == self.null:
                self.push_heap(BEAM_WIDTHS[length], code, uid,
                               prev.perm if prev is not None else TOP_CHOICE_PERM, False, False,
                               False, dup, cert, prev, None, step[cont * NUM_LENGTHS + length])

    # ---- ContinueUnichar
    def continue_unichar(self, code, uid, cert, use_dawgs, cont, prev, step):
        if use_dawgs:
            if cert > WORST_DICT_CERT:
                self.continue_dawg(code, uid, cert, cont, prev, step)
            return
        self.push_heap(BW0, code, uid, TOP_CHOICE_PERM, False, False, False, False,
                       f32(cert * DICT_RATIO), prev, None, step[cont * NUM_LENGTHS])
        d = self.dict
        if d is not None and ((uid == UNICHAR_SPACE and cert > WORST_DICT_CERT) or
                              not d.is_space_delimited(uid)):
            # A top-choice position that can start a word also seeds the dawg search. A space
            # keeps its certainty unscaled and is flagged NO_PERM (see recodebeam.cpp).
            if uid == UNICHAR_SPACE:
                perm = NO_PERM
                dawg_cert = cert
            else:
                perm = TOP_CHOICE_PERM
                dawg_cert = f32(cert * DICT_RATIO)
            self.push_initial_dawg(code, uid, perm, False, False, dawg_cert, cont, prev)

    # ---- ContinueDawg
    def continue_dawg(self, code, uid, cert, cont, prev, step):
        dawg_heap = step[(3 + cont) * NUM_LENGTHS]
        nodawg_heap = step[cont * NUM_LENGTHS]
        if uid == INVALID:
            self.push_heap(BW0, code, uid, NO_PERM, False, False, False, False, cert, prev,
                           None, dawg_heap)
            return
        # Avoid the dictionary probe if the score is a total loss.
        score = cert if prev is None else f32(cert + prev.score)
        dh = dawg_heap.h
        nh = nodawg_heap.h
        if (len(dh) >= BW0 and score <= dh[0][1].score and
                len(nh) >= BW0 and score <= nh[0][1].score):
            return
        uni_prev = prev
        while uni_prev is not None and (uni_prev.uid == INVALID or uni_prev.dup):
            uni_prev = uni_prev.prev
        d = self.dict
        if uid == UNICHAR_SPACE:
            if uni_prev is not None and uni_prev.eow:
                self.push_initial_dawg(code, uid, uni_prev.perm, False, False, cert, cont, prev)
                self.push_heap(BW0, code, uid, uni_prev.perm, False, False, False, False, cert,
                               prev, None, nodawg_heap)
            return
        elif (uni_prev is not None and uni_prev.sod and uni_prev.uid != UNICHAR_SPACE and
              d.is_space_delimited(uni_prev.uid) and d.is_space_delimited(uid)):
            return                 # can't break words between space-delimited chars
        if uni_prev is None:
            active = d.default_dawgs()
            word_start = True
        elif uni_prev.dawgs is not None:
            active = uni_prev.dawgs
            word_start = uni_prev.sod
        else:
            return                 # can't continue if not a dict word
        perm, updated, valid_end = d.letter_is_okay(active, uid)
        if perm != NO_PERM:
            self.push_heap(BW0, code, uid, perm, False, word_start, valid_end, False, cert, prev,
                           updated, dawg_heap)
            if valid_end and not self.space_delimited:
                self.push_initial_dawg(code, uid, perm, word_start, True, cert, cont, prev)
                self.push_heap(BW0, code, uid, perm, False, word_start, True, False, cert, prev,
                               None, nodawg_heap)

    # ---- ContinueContext
    def continue_context(self, prev, index, flag, step):
        null = self.null
        outs = self.outs
        flags = self.flags
        length = index % NUM_LENGTHS
        cont = (index // NUM_LENGTHS) % 3
        use_dawgs = index >= 3 * NUM_LENGTHS
        prefix = [null] * length
        p_node = prev
        for p in range(length - 1, -1, -1):
            while p_node is not None and (p_node.dup or p_node.code == null):
                p_node = p_node.prev
            if p_node is not None:
                prefix[p] = p_node.code
            p_node = None if p_node is None else p_node.prev
        pkey = tuple(prefix)

        if prev is not None and not self.simple:
            if flags[prev.code] == flag:
                if cont != NC_NO_DUP:
                    self.push_dup(length, True, prev.code, prev.uid, cert_of(outs[prev.code]),
                                  use_dawgs, NC_ANY, prev, step)
                if cont == NC_ANY and flag == TN_TOP2 and prev.code != null:
                    cert = cert_of(f32(outs[prev.code] + outs[null]))
                    self.push_dup(length, True, prev.code, prev.uid, cert, use_dawgs, NC_NO_DUP,
                                  prev, step)
            if cont == NC_ONLY_DUP:
                return
            if prev.code != null and length > 0 and flags[null] == flag:
                self.push_dup(length, False, null, INVALID, cert_of(outs[null]), use_dawgs,
                              NC_ANY, prev, step)

        final_codes = self.rc.final.get(pkey)
        if final_codes:
            for code in final_codes:
                if flags[code] != flag:
                    continue
                if prev is not None and prev.code == code and not self.simple:
                    continue
                cert = cert_of(outs[code])
                if cert < MIN_CERT and code != null:
                    continue
                if length + 1 > MAX_CODE_LEN:
                    uid = INVALID
                else:
                    uid = self.rc.decoder.get(pkey + (code,), INVALID)
                if length == 0 and code == null:
                    uid = INVALID
                self.continue_unichar(code, uid, cert, use_dawgs, NC_ANY, prev, step)
                if flag == TN_TOP2 and code != null:
                    prob = f32(outs[code] + outs[null])
                    if (prev is not None and cont == NC_ANY and prev.code != null and
                            ((prev.code == self.top_code and code == self.second_code) or
                             (code == self.top_code and prev.code == self.second_code))):
                        prob = f32(prob + outs[prev.code])
                    self.continue_unichar(code, uid, cert_of(prob), use_dawgs, NC_ONLY_DUP,
                                          prev, step)

        next_codes = self.rc.next.get(pkey)
        if next_codes:
            for code in next_codes:
                if flags[code] != flag:
                    continue
                if prev is not None and prev.code == code and not self.simple:
                    continue
                cert = cert_of(outs[code])
                self.push_dup(length + 1, False, code, INVALID, cert, use_dawgs, NC_ANY, prev,
                              step)
                if flag == TN_TOP2 and code != null:
                    prob = f32(outs[code] + outs[null])
                    if (prev is not None and cont == NC_ANY and prev.code != null and
                            ((prev.code == self.top_code and code == self.second_code) or
                             (code == self.top_code and prev.code == self.second_code))):
                        prob = f32(prob + outs[prev.code])
                    self.push_dup(length + 1, False, code, INVALID, cert_of(prob), use_dawgs,
                                  NC_ONLY_DUP, prev, step)

    # ---- DecodeStep
    def decode_step(self, t):
        step = [_Heap() for _ in range(self.nbeams)]
        self.beams.append(step)
        self.best_init = [None, None, None]
        if t == 0:
            # The first step only uses singles and initials; best_initial_dawgs_ set here are
            # never pushed (the C++ pushes them only for t > 0).
            self.continue_context(None, 0, TN_TOP2, step)
            if self.dict is not None:
                self.continue_context(None, 3 * NUM_LENGTHS, TN_TOP2, step)
            return
        prev_step = self.beams[t - 1]
        any_idx = list(range(NUM_LENGTHS))
        if self.dict is not None:
            any_idx += list(range(3 * NUM_LENGTHS, 4 * NUM_LENGTHS))
        total = 0
        for tn in range(3):
            if total != 0:
                break
            for index in range(self.nbeams):
                h = prev_step[index].h
                for i in range(len(h) - 1, -1, -1):
                    self.continue_context(h[i][1], index, tn, step)
            total = sum(len(step[i].h) for i in any_idx)
        if self.dict is not None:
            for c in range(3):
                node = self.best_init[c]
                if node is not None:
                    self.push_node(BW0, node, step[(3 + c) * NUM_LENGTHS])

    # ---- ExtractBestPaths
    def best_node(self):
        last = self.beams[-1]
        best = None
        second = None
        halves = (0, 1) if self.dict is not None else (0,)
        for cont in (NC_ANY, NC_NO_DUP):
            for is_dawg in halves:
                for pair in last[(is_dawg * 3 + cont) * NUM_LENGTHS].h:
                    node = pair[1]
                    if is_dawg:
                        dn = node
                        while dn is not None and (dn.uid == INVALID or dn.dup):
                            dn = dn.prev
                        if dn is None or (not dn.eow and dn.uid != UNICHAR_SPACE):
                            continue
                    if best is None or node.score > best.score:
                        second = best
                        best = node
                    elif second is None or node.score > second.score:
                        second = node
        return best


def _path(node):
    out = []
    while node is not None:
        out.append(node)
        node = node.prev
    out.reverse()
    return out


def _path_ids_x(path):
    """ExtractPathAsUnicharIds: unichar ids and the path index of each (xcoords)."""
    ids = []
    xs = []
    t = 0
    width = len(path)
    while t < width:
        while t < width and path[t].uid == INVALID:
            t += 1
        if t < width:
            ids.append(path[t].uid)
            xs.append(t)
            t += 1
            while t < width and path[t].dup:
                t += 1
    return ids, xs


def _words(path, ids, xs, dictionary):
    """The word split of ExtractBestPathAsWords: [(start, end, permuter)] over ids."""
    def sd(uid):
        return True if dictionary is None else dictionary.is_space_delimited(uid)

    out = []
    n = len(ids)
    start = 0
    while start < n:
        end = start + 1
        while end < n:
            if ids[end] == UNICHAR_SPACE:
                break
            node = path[xs[end]]
            if node.sow:
                break
            if node.perm == TOP_CHOICE_PERM and (not sd(ids[end]) or not sd(ids[end - 1])):
                break
            end += 1
        out.append((start, end, path[xs[end - 1]].perm))
        if end < n and ids[end] == UNICHAR_SPACE:
            end += 1
        start = end
    return out


def _run(probs, model, dictionary):
    rows = np.asarray(probs, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return None
    rows = rows.tolist()
    s = _DictSearch(_Recoder(model.recoder), model.null_char, bool(model.simple_text),
                    dictionary)
    for t, outs in enumerate(rows):
        s.compute_top_n(outs, BW0)
        s.decode_step(t)
    return s.best_node()


def decode(probs, model, dictionary):
    """RecodeBeamSearch with the LSTM dictionary over one line: [T, C] float32 -> unichar ids
    (ExtractBestPathAsUnicharIds). dictionary=None runs the same code without dawgs."""
    best = _run(probs, model, dictionary)
    if best is None:
        return []
    return _path_ids_x(_path(best))[0]


def decode_full(probs, model, dictionary):
    """decode() plus what ExtractBestPathAsWords sees: per id its path index (xcoords), the
    node's permuter / start_of_word / end_of_word, and the words as (start, end, permuter)."""
    best = _run(probs, model, dictionary)
    if best is None:
        return {"ids": [], "xcoords": [], "perm": [], "sow": [], "eow": [], "words": []}
    path = _path(best)
    ids, xs = _path_ids_x(path)
    return {"ids": ids, "xcoords": xs,
            "perm": [path[x].perm for x in xs],
            "sow": [path[x].sow for x in xs],
            "eow": [path[x].eow for x in xs],
            "words": _words(path, ids, xs, dictionary)}
