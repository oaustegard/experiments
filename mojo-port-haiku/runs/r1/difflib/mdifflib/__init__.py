"""SequenceMatcher with its b-side index, find_longest_match and
get_matching_blocks running in Mojo (_kernel.so, built by build.sh).

The Python side maps elements to integer ids and hands the kernel buffers
of int32/int64 through one int64 context block (see kernel.mojo, S_*).

Two ways to get ids, with the same results:

* codepoint mode, when b is a str: its code points (UTF-32, one int per
  character) go to Mojo, which interns them. Two characters are equal
  exactly when their code points are, so no Python loop touches the
  elements.
* dict mode, for any other hashable sequence: a dict assigns ids, so equality
  and hashing are exactly those of the elements.

Everything else mirrors difflib.SequenceMatcher.
"""
from __future__ import annotations

import sys
from array import array
from _collections import _count_elements  # the C helper collections.Counter uses
from collections import namedtuple
from itertools import starmap
from types import GenericAlias

from . import _kernel

__all__ = ["SequenceMatcher", "Match"]

Match = namedtuple('Match', 'a b size')

assert array('i').itemsize == 4 and array('q').itemsize == 8

# Context-block slots; keep in step with kernel.mojo.
_S_N, _S_K, _S_BIDS, _S_JUNK, _S_COUNT, _S_POP, _S_START, _S_POS = range(8)
_S_AUTOJUNK, _S_LENS, _S_STAMPS, _S_GEN, _S_AIDS, _S_AOFF, _S_LA, _S_OUT = range(8, 16)
_S_BCP, _S_HKEYS, _S_HVALS, _S_HMASK, _S_KCP, _S_ACP = range(16, 22)
_NSLOTS = 24


def _calculate_ratio(matches, length):
    if length:
        return 2.0 * matches / length
    return 1.0


def _addr(arr):
    return arr.buffer_info()[0]


def _codepoints(s):
    """One int32 per character of s: its code point. None unless s is exactly
    a str that encodes (a lone surrogate does not). The buffer is never empty,
    so its address is never 0."""
    if type(s) is not str:
        return None
    try:
        data = s.encode('utf-32-le' if sys.byteorder == 'little' else 'utf-32-be')
    except UnicodeEncodeError:
        return None
    cps = array('i')
    cps.frombytes(data)
    if not cps:
        cps.append(0)
    return cps


class _Ids(dict):
    """Element -> id. An element that is not in b maps to -1."""
    __slots__ = ()

    def __missing__(self, key):
        return -1


class _BSide:
    """The b-side index: what the kernel reads, plus the buffers holding it.
    The arrays stay referenced here so their memory outlives kernel calls."""
    __slots__ = ('n', 'k', 'cp', 'keys', 'ids', 'ctx', 'ctxaddr', 'keep',
                 'junk', 'count', 'pop', 'start', 'pos')

    def idmap(self):
        """Element -> id for the elements of a. In codepoint mode the ids of b
        come from Mojo, so this dict is built on first use."""
        if self.ids is None:
            self.ids = _Ids(zip(self.keys, range(self.k)))
        return self.ids


class SequenceMatcher:
    """
    SequenceMatcher is a flexible class for comparing pairs of sequences of
    any type, so long as the sequence elements are hashable. The algorithm
    and the public API are those of difflib.SequenceMatcher, and results are
    identical; see that class for the full description.

    Methods: set_seqs, set_seq1, set_seq2, find_longest_match,
    get_matching_blocks, get_opcodes, get_grouped_opcodes, ratio,
    quick_ratio, real_quick_ratio.
    """

    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self._bs = None
        self._b2j = self._bjunk = self._bpopular = None
        self.set_seqs(a, b)

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b):
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self.__chain_b()

    # The b-side index. Ids are assigned in first-occurrence order, which is
    # the key order of difflib's b2j dict, so the b2j, bjunk and bpopular
    # views below come out in the same order as the reference.
    def __chain_b(self):
        b = self.b
        n = len(b)
        bs = _BSide()
        bs.n = n
        bs.ids = None
        bs.keep = ()
        ctx = bs.ctx = array('q', [0]) * _NSLOTS
        ctx[_S_N] = n
        ctx[_S_AUTOJUNK] = 1 if self.autojunk else 0
        ctx[_S_GEN] = 0
        bids = array('i', [0]) * max(n, 1)
        cps = _codepoints(b)
        if cps is not None:
            bs.cp = True
            size = 4
            while size < 2 * n:
                size <<= 1
            hkeys = array('i', [-1]) * size
            hvals = array('i', [0]) * size
            kcp = array('i', [0]) * max(n, 1)
            ctx[_S_BCP] = _addr(cps)
            ctx[_S_HKEYS] = _addr(hkeys)
            ctx[_S_HVALS] = _addr(hvals)
            ctx[_S_HMASK] = size - 1
            ctx[_S_KCP] = _addr(kcp)
            ctx[_S_BIDS] = _addr(bids)
            bs.ctxaddr = _addr(ctx)
            k = _kernel.intern_b(bs.ctxaddr)
            keys = [chr(c) for c in kcp[:k]]
            bs.keep = (cps, hkeys, hvals, kcp)
        else:
            bs.cp = False
            keys = list(dict.fromkeys(b))
            k = len(keys)
            ids = _Ids(zip(keys, range(k)))
            bids = array('i', list(map(ids.__getitem__, b)) or [0])
            bs.ids = ids
            ctx[_S_BIDS] = _addr(bids)
            bs.ctxaddr = _addr(ctx)
        bs.k = k
        bs.keys = keys
        ctx[_S_K] = k
        isjunk = self.isjunk
        if isjunk:
            junk = array('i', [1 if isjunk(elt) else 0 for elt in keys] or [0])
        else:
            junk = array('i', [0]) * max(k, 1)
        bs.junk = junk
        bs.count = count = array('i', [0]) * max(k, 1)
        bs.pop = pop = array('i', [0]) * max(k, 1)
        bs.start = start = array('i', [0]) * (k + 1)
        bs.pos = pos = array('i', [0]) * max(n, 1)
        lens = array('i', [0]) * (2 * max(n, 1))
        stamps = array('q', [0]) * (2 * max(n, 1))
        bs.keep = bs.keep + (bids, junk, count, pop, start, pos, lens, stamps)
        ctx[_S_JUNK] = _addr(junk)
        ctx[_S_COUNT] = _addr(count)
        ctx[_S_POP] = _addr(pop)
        ctx[_S_START] = _addr(start)
        ctx[_S_POS] = _addr(pos)
        ctx[_S_LENS] = _addr(lens)
        ctx[_S_STAMPS] = _addr(stamps)
        _kernel.build_index(bs.ctxaddr)
        self._bs = bs
        self._b2j = self._bjunk = self._bpopular = None

    @property
    def b2j(self):
        if self._b2j is None:
            bs = self._bs
            start, pos, keys = bs.start, bs.pos, bs.keys
            self._b2j = {keys[t]: pos[start[t]:start[t + 1]].tolist()
                         for t in range(bs.k) if start[t + 1] > start[t]}
        return self._b2j

    @property
    def bjunk(self):
        if self._bjunk is None:
            bs = self._bs
            self._bjunk = {elt for elt, f in zip(bs.keys, bs.junk) if f}
        return self._bjunk

    @property
    def bpopular(self):
        if self._bpopular is None:
            bs = self._bs
            self._bpopular = {elt for elt, f in zip(bs.keys, bs.pop) if f}
        return self._bpopular

    def _load_a(self, a, alo, ahi):
        """Point the kernel's a-side slots at a[alo:ahi]. Returns the buffers,
        which the caller keeps alive for the kernel call."""
        bs = self._bs
        ctx = bs.ctx
        count = ahi - alo
        seg = a if (alo == 0 and ahi == len(a)) else a[alo:ahi]
        cps = _codepoints(seg) if bs.cp else None
        if cps is not None:
            aids = array('i', [0]) * max(count, 1)
            ctx[_S_ACP] = _addr(cps)
        else:
            ids = bs.idmap()
            aids = array('i', list(map(ids.__getitem__, seg)) or [0])
            ctx[_S_ACP] = 0
        ctx[_S_AIDS] = _addr(aids)
        ctx[_S_AOFF] = alo
        return cps, aids

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        """Find longest matching block in a[alo:ahi] and b[blo:bhi].

        Returns Match(i, j, k): a[i:i+k] == b[j:j+k], with k maximal. The
        match is extended over junk on both sides, as in difflib.
        """
        a, b = self.a, self.b
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        if ahi <= alo or bhi <= blo:
            return Match(alo, blo, 0)
        if not (0 <= alo and ahi <= len(a) and 0 <= blo and bhi <= len(b)):
            raise IndexError("find_longest_match: range out of bounds")
        bs = self._bs
        keep = self._load_a(a, alo, ahi)
        i, j, k = _kernel.find_longest_match(bs.ctxaddr, alo, ahi, blo, bhi)
        del keep
        return Match(i, j, k)

    def get_matching_blocks(self):
        """Return list of triples (i, j, n) with a[i:i+n] == b[j:j+n].

        The triples are increasing in i and j, and adjacent triples never
        describe adjacent equal blocks. The last triple is the dummy
        (len(a), len(b), 0).
        """
        if self.matching_blocks is not None:
            return self.matching_blocks
        a = self.a
        bs = self._bs
        la = len(a)
        keep = self._load_a(a, 0, la)
        out = array('q', [0]) * (3 * (min(la, bs.n) + 2))
        ctx = bs.ctx
        ctx[_S_LA] = la
        ctx[_S_OUT] = _addr(out)
        m = _kernel.matching_blocks(bs.ctxaddr)
        del keep
        vals = out[:3 * m].tolist()
        self.matching_blocks = list(starmap(Match, zip(vals[0::3], vals[1::3], vals[2::3])))
        return self.matching_blocks

    def get_opcodes(self):
        """Return list of 5-tuples (tag, i1, i2, j1, j2) turning a into b.

        Tags: 'replace', 'delete', 'insert', 'equal'.
        """
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            # invariant: a[:i] has been turned into b[:j]; the next matching
            # block is a[ai:ai+size] == b[bj:bj+size]
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                answer.append((tag, i, ai, j, bj))
            i, j = ai+size, bj+size
            # the list of matching blocks is terminated by a sentinel of size 0
            if size:
                answer.append(('equal', ai, i, bj, j))
        return answer

    def get_grouped_opcodes(self, n=3):
        """Isolate change clusters by eliminating ranges with no changes.

        Return a generator of groups with up to n lines of context.
        Each group is in the same format as returned by get_opcodes().
        """
        codes = self.get_opcodes()
        if not codes:
            codes = [("equal", 0, 1, 0, 1)]
        # Fixup leading and trailing groups if they show no changes.
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2-n), i2, max(j1, j2-n), j2
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1+n), j1, min(j2, j1+n)

        nn = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            # End the current group and start a new one whenever
            # there is a large range with no changes.
            if tag == 'equal' and i2-i1 > nn:
                group.append((tag, i1, min(i2, i1+n), j1, min(j2, j1+n)))
                yield group
                group = []
                i1, j1 = max(i1, i2-n), max(j1, j2-n)
            group.append((tag, i1, i2, j1, j2))
        if group and not (len(group) == 1 and group[0][0] == 'equal'):
            yield group

    def ratio(self):
        """Return a measure of the sequences' similarity (float in [0,1]).

        Where T is the total number of elements in both sequences, and
        M is the number of matches, this is 2.0*M / T.
        """
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self):
        """Return an upper bound on ratio() relatively quickly.

        Matches are counted as the multiset intersection of a and b, which
        is the sum of min(count in a, count in b) over the elements of a.
        """
        if self.fullbcount is None:
            bs = self._bs
            self.fullbcount = dict(zip(bs.keys, bs.count))
        get = self.fullbcount.get
        avail = {}
        _count_elements(avail, self.a)
        matches = 0
        for elt, numa in avail.items():
            numb = get(elt, 0)
            matches += numa if numa < numb else numb
        length = len(self.a) + len(self.b)
        if length:
            return 2.0 * matches / length
        return 1.0

    def real_quick_ratio(self):
        """Return an upper bound on ratio() very quickly."""
        la, lb = len(self.a), len(self.b)
        length = la + lb
        if length:
            return 2.0 * min(la, lb) / length
        return 1.0

    __class_getitem__ = classmethod(GenericAlias)
