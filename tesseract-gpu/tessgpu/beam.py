"""Port of Tesseract 5.3.4 RecodeBeamSearch, no-dictionary path only.

Source: src/lstm/recodebeam.{h,cpp} and src/ccutil/unicharcompress.{h,cpp} (SetupDecoder,
DecodeUnichar, code range), read from the 5.3.4 tag. Only what the call

    Decode(outputs, dict_ratio=2.25, cert_offset=-0.085, worst_dict_cert=-25/7, charset,
           lstm_choice_mode=0)  ->  ExtractBestPathAsUnicharIds

executes when dict_ == nullptr is ported. Every dawg branch is dead there, so it is omitted,
and worst_dict_cert is never read (it only gates dawg pushes). The certainty/score arithmetic
mirrors the C++ types: ProbToCertainty returns float, `+ cert_offset` widens to double and is
rounded back to float at the assignment, and scores are float sums of float certainties.
Every float32 rounding point is kept with f32(). Heaps are a faithful GenericHeap (same
sift order, same storage order), because tie-breaks and the ExtractBestPaths scan follow
storage order.

decode(probs, model) -> list[int] of unichar ids, the same as ExtractBestPathAsUnicharIds.
"""

import math

import numpy as np

INVALID = -1                       # INVALID_UNICHAR_ID
MIN_CERT = -20.0                   # RecodeBeamSearch::kMinCertainty (float)
KMIN_PROB = float(np.float32(math.exp(-20.0)))  # networkio.cpp: kMinProb = exp(kMinCertainty)
DICT_RATIO = 2.25                  # the dict_ratio argument (exact as float)
CERT_OFFSET = -0.085               # the cert_offset argument (double)
BEAM_WIDTHS = (5, 10, 16, 16, 16, 16, 16, 16, 16, 16)  # kBeamWidths, indexed by code length
NUM_LENGTHS = 10                   # kMaxCodeLen + 1
MAX_CODE_LEN = 9                   # RecodedCharID::kMaxCodeLen
NC_ANY, NC_ONLY_DUP, NC_NO_DUP = 0, 1, 2
TN_TOP2, TN_TOPN, TN_ALSO_RAN = 0, 1, 2
NUM_BEAMS = 3 * NUM_LENGTHS        # non-dawg part of kNumBeams; the dawg half is always empty
MASK64 = (1 << 64) - 1

_f32 = np.float32


def f32(x):
    """Round a Python float to the nearest float32, returned as a float."""
    return float(_f32(x))


def prob_to_certainty(p):
    """NetworkIO::ProbToCertainty: std::log of a float, a float result."""
    return f32(math.log(p)) if p > KMIN_PROB else MIN_CERT


def cert_of(p):
    """float cert = ProbToCertainty(p) + cert_offset: double sum, stored as float."""
    return f32(prob_to_certainty(p) + CERT_OFFSET)


# ---------------------------------------------------------------- GenericHeap (min-heap)

def _sift_up(h, hole, pair):
    key = pair[0]
    while hole > 0:
        parent = (hole + 1) // 2 - 1
        if key < h[parent][0]:
            h[hole] = h[parent]
            hole = parent
        else:
            break
    return hole


def _sift_down(h, hole, pair):
    n = len(h)
    key = pair[0]
    while True:
        child = 2 * hole + 1
        if child >= n:
            break
        if child + 1 < n and h[child + 1][0] < h[child][0]:
            child += 1
        if h[child][0] < key:
            h[hole] = h[child]
            hole = child
        else:
            break
    return hole


class _Heap:
    """GenericHeap<KDPairInc>: pairs are [key, data]; the top is the smallest key (the worst)."""

    __slots__ = ("h",)

    def __init__(self):
        self.h = []

    def push(self, pair):
        h = self.h
        h.append(pair)
        hole = _sift_up(h, len(h) - 1, pair)
        h[hole] = pair

    def pop(self):
        h = self.h
        top = h[0]
        last = h.pop()
        if h:
            hole = _sift_down(h, 0, last)
            h[hole] = last
        return top

    def reshuffle(self, index):
        h = self.h
        pair = h[index]
        hole = _sift_down(h, index, pair)
        hole = _sift_up(h, hole, pair)
        h[hole] = pair


# ---------------------------------------------------------------- recoder (SetupDecoder)

class _Recoder:
    """UnicharCompress lookups: full code -> unichar id, prefix -> next / final codes."""

    def __init__(self, recoder):
        self.code_range = max(c for code in recoder for c in code) + 1
        self.decoder = {}
        self.final = {}
        self.next = {}
        for uid, code in enumerate(recoder):
            self.decoder[code] = uid
            n = len(code)
            ln = n - 1
            prefix = code[:ln]
            if prefix not in self.final:
                self.final[prefix] = [code[ln]]
                ln -= 1
                while ln >= 0:
                    p = code[:ln]
                    if p not in self.next:
                        self.next[p] = [code[ln]]
                    else:
                        if code[ln] not in self.next[p]:
                            self.next[p].append(code[ln])
                        break
                    ln -= 1
            else:
                if code[ln] not in self.final[prefix]:
                    self.final[prefix].append(code[ln])


# ---------------------------------------------------------------- beam search

class _Node:
    """RecodeNode, reduced to the fields the no-dictionary path reads."""

    __slots__ = ("code", "uid", "dup", "score", "prev", "hash")

    def __init__(self, code, uid, dup, score, prev, hsh):
        self.code = code
        self.uid = uid
        self.dup = dup
        self.score = score
        self.prev = prev
        self.hash = hsh


class _Search:
    def __init__(self, recoder, null_char, simple_text):
        self.rc = recoder
        self.null = null_char
        self.simple = simple_text
        self.beams = []
        self.outs = None
        self.flags = None
        self.top_code = -1
        self.second_code = -1

    # ---- ComputeTopN: top-5 by a min-heap of (value, index); flags TOP2 / TOPN / ALSO_RAN
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

    # ---- ComputeCodeHash
    def code_hash(self, code, dup, prev):
        h = 0 if prev is None else prev.hash
        if not dup and code != self.null:
            nc = self.rc.code_range
            carry = (((h >> 32) * nc) >> 32) & MASK64
            h = (h * nc) & MASK64
            h = (h + carry) & MASK64
            h = (h + code) & MASK64
        return h

    # ---- UpdateHeapIfMatched
    @staticmethod
    def update_if_matched(node, heap):
        h = heap.h
        for idx in range(len(h)):
            pair = h[idx]
            old = pair[1]
            if old.code == node.code and old.hash == node.hash:
                if node.score > old.score:
                    pair[1] = node
                    pair[0] = node.score
                    heap.reshuffle(idx)
                return True
        return False

    # ---- PushHeapIfBetter(max_size, code, uid, perm, ..., dup, cert, prev, heap)
    def push_heap(self, max_size, code, uid, dup, cert, prev, heap):
        score = cert if prev is None else f32(cert + prev.score)
        h = heap.h
        if len(h) < max_size or score > h[0][0]:
            node = _Node(code, uid, dup, score, prev, self.code_hash(code, dup, prev))
            if self.update_if_matched(node, heap):
                return
            heap.push([score, node])
            if len(h) > max_size:
                heap.pop()

    # ---- PushDupOrNoDawgIfBetter, no-dawg branch
    def push_dup(self, length, dup, code, uid, cert, cont, prev, step):
        cert = f32(cert * DICT_RATIO)
        if cert >= MIN_CERT or code == self.null:
            self.push_heap(BEAM_WIDTHS[length], code, uid, dup, cert, prev,
                           step[cont * NUM_LENGTHS + length])

    # ---- ContinueUnichar, no-dawg branch
    def continue_unichar(self, code, uid, cert, cont, prev, step):
        cert = f32(cert * DICT_RATIO)
        self.push_heap(BEAM_WIDTHS[0], code, uid, False, cert, prev,
                       step[cont * NUM_LENGTHS])

    # ---- ContinueContext
    def continue_context(self, prev, index, flag, step):
        null = self.null
        outs = self.outs
        flags = self.flags
        length = index % NUM_LENGTHS
        cont = index // NUM_LENGTHS
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
                                  NC_ANY, prev, step)
                if cont == NC_ANY and flag == TN_TOP2 and prev.code != null:
                    cert = cert_of(f32(outs[prev.code] + outs[null]))
                    self.push_dup(length, True, prev.code, prev.uid, cert, NC_NO_DUP, prev, step)
            if cont == NC_ONLY_DUP:
                return
            if prev.code != null and length > 0 and flags[null] == flag:
                self.push_dup(length, False, null, INVALID, cert_of(outs[null]), NC_ANY,
                              prev, step)

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
                self.continue_unichar(code, uid, cert, NC_ANY, prev, step)
                if flag == TN_TOP2 and code != null:
                    prob = f32(outs[code] + outs[null])
                    if (prev is not None and cont == NC_ANY and prev.code != null and
                            ((prev.code == self.top_code and code == self.second_code) or
                             (code == self.top_code and prev.code == self.second_code))):
                        prob = f32(prob + outs[prev.code])
                    cert = cert_of(prob)
                    self.continue_unichar(code, uid, cert, NC_ONLY_DUP, prev, step)

        next_codes = self.rc.next.get(pkey)
        if next_codes:
            for code in next_codes:
                if flags[code] != flag:
                    continue
                if prev is not None and prev.code == code and not self.simple:
                    continue
                cert = cert_of(outs[code])
                self.push_dup(length + 1, False, code, INVALID, cert, NC_ANY, prev, step)
                if flag == TN_TOP2 and code != null:
                    prob = f32(outs[code] + outs[null])
                    if (prev is not None and cont == NC_ANY and prev.code != null and
                            ((prev.code == self.top_code and code == self.second_code) or
                             (code == self.top_code and prev.code == self.second_code))):
                        prob = f32(prob + outs[prev.code])
                    cert = cert_of(prob)
                    self.push_dup(length + 1, False, code, INVALID, cert, NC_ONLY_DUP, prev,
                                  step)

    # ---- DecodeStep (no-dawg: index = cont * NUM_LENGTHS + length)
    def decode_step(self, t):
        step = [_Heap() for _ in range(NUM_BEAMS)]
        self.beams.append(step)
        if t == 0:
            self.continue_context(None, 0, TN_TOP2, step)
            return
        prev_step = self.beams[t - 1]
        total = 0
        for tn in range(3):
            if total != 0:
                break
            for index in range(NUM_BEAMS):
                h = prev_step[index].h
                for i in range(len(h) - 1, -1, -1):
                    self.continue_context(h[i][1], index, tn, step)
            total = sum(len(step[i].h) for i in range(NUM_LENGTHS))

    # ---- ExtractBestPaths (length-0 heaps, ANYTHING and NO_DUP continuations)
    def best_node(self):
        last = self.beams[-1]
        best = None
        second = None
        for cont in (NC_ANY, NC_NO_DUP):
            for node in (pair[1] for pair in last[cont * NUM_LENGTHS].h):
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


def _path_ids(path):
    """ExtractPathAsUnicharIds, the unichar-id part only (certainties do not affect ids)."""
    ids = []
    t = 0
    width = len(path)
    while t < width:
        while t < width and path[t].uid == INVALID:
            t += 1
        if t < width:
            ids.append(path[t].uid)
            t += 1
            while t < width and path[t].dup:
                t += 1
    return ids


def decode(probs, model):
    """Tesseract's RecodeBeamSearch (no dictionary) over one line: [T, C] float32 -> unichar ids."""
    rows = np.asarray(probs, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return []
    rows = rows.tolist()           # exact float32 values as Python floats
    rc = _Recoder(model.recoder)
    s = _Search(rc, model.null_char, bool(model.simple_text))
    for t, outs in enumerate(rows):
        s.compute_top_n(outs, BEAM_WIDTHS[0])
        s.decode_step(t)
    best = s.best_node()
    if best is None:
        return []
    return _path_ids(_path(best))
