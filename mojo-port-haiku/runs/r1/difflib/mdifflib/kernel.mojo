from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer


# Context block: an int64 array owned by the Python shim (mdifflib/__init__.py).
# Each slot holds the address of a buffer the shim allocated, or a scalar.
# Buffers are read and written in place; nothing here owns memory.
comptime S_N = 0          # len(b)
comptime S_K = 1          # number of distinct elements of b (ids are 0..K-1)
comptime S_BIDS = 2       # int32[n]   id of each b element
comptime S_JUNK = 3       # int32[K]   1 if the element is junk
comptime S_COUNT = 4      # int32[K]   occurrences of each id in b (out)
comptime S_POP = 5        # int32[K]   1 if the element is popular (out)
comptime S_START = 6      # int32[K+1] CSR offsets into POS (out)
comptime S_POS = 7        # int32[n]   b positions per id, junk/popular removed (out)
comptime S_AUTOJUNK = 8   # 0 or 1
comptime S_LENS = 9       # int32[2n]  run lengths, double-buffered by row parity
comptime S_STAMPS = 10    # int64[2n]  generation stamp of each run-length entry
comptime S_GEN = 11       # int64      generation counter, persists across calls
comptime S_AIDS = 12      # int32      ids of the a-side range (out when S_ACP != 0)
comptime S_AOFF = 13      # a-index of the first entry of the a-side buffers
comptime S_LA = 14        # len(a)
comptime S_OUT = 15       # int64[3*cap] matching blocks out
comptime S_BCP = 16       # int32[n]   code points of b (codepoint mode only)
comptime S_HKEYS = 17     # int32[M]   interning table: code point, -1 if empty
comptime S_HVALS = 18     # int32[M]   interning table: dense id
comptime S_HMASK = 19     # M - 1
comptime S_KCP = 20       # int32[n]   distinct code points of b, first-occurrence order (out)
comptime S_ACP = 21       # int32      code points of the a-side range, 0 in dict mode
comptime S_NSLOTS = 24

comptime P32 = Pointer[Int32, MutUntrackedOrigin]
comptime P64 = Pointer[Int64, MutUntrackedOrigin]


@always_inline
def slot_of(c: Int, mask: Int) -> Int:
    return ((c * 2654435761) >> 11) & mask


struct Ctx:
    var slots: P64

    def __init__(out self, addr: Int):
        self.slots = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)

    def get(self, slot: Int) -> Int:
        return Int(self.slots[unsafe_offset=slot])

    def put(self, slot: Int, value: Int):
        self.slots[unsafe_offset=slot] = Int64(value)

    def ptr32(self, slot: Int) -> P32:
        return P32(unsafe_from_address=self.get(slot))

    def ptr64(self, slot: Int) -> P64:
        return P64(unsafe_from_address=self.get(slot))


struct Index(Movable):
    var n: Int
    var k: Int
    var aoff: Int
    var gen: Int
    var bids: P32
    var junk: P32
    var count: P32
    var pop: P32
    var start: P32
    var pos: P32
    var lens: P32
    var stamps: P64
    var aids: P32

    def __init__(out self, ctx: Ctx):
        self.n = ctx.get(S_N)
        self.k = ctx.get(S_K)
        self.aoff = ctx.get(S_AOFF)
        self.gen = ctx.get(S_GEN)
        self.bids = ctx.ptr32(S_BIDS)
        self.junk = ctx.ptr32(S_JUNK)
        self.count = ctx.ptr32(S_COUNT)
        self.pop = ctx.ptr32(S_POP)
        self.start = ctx.ptr32(S_START)
        self.pos = ctx.ptr32(S_POS)
        self.lens = ctx.ptr32(S_LENS)
        self.stamps = ctx.ptr64(S_STAMPS)
        self.aids = ctx.ptr32(S_AIDS)

    def save(self, ctx: Ctx):
        ctx.put(S_GEN, self.gen)

    @always_inline
    def bid(self, x: Int) -> Int:
        return Int(self.bids[unsafe_offset=x])

    @always_inline
    def is_junk_pos(self, x: Int) -> Bool:
        return Int(self.junk[unsafe_offset=self.bid(x)]) != 0

    @always_inline
    def aid(self, i: Int) -> Int:
        return Int(self.aids[unsafe_offset=i - self.aoff])

    def intern_a(self, ctx: Ctx, count: Int):
        """Codepoint mode: map the a-side code points to the b-side ids
        through the interning table built by intern_b. A code point that is
        not among b's gets -1."""
        var hkeys = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=ctx.get(S_HKEYS))
        var hvals = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=ctx.get(S_HVALS))
        var acp = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=ctx.get(S_ACP))
        var mask = ctx.get(S_HMASK)
        for x in range(count):
            var c = Int(acp[unsafe_offset=x])
            var h = slot_of(c, mask)
            var found = -1
            while True:
                var kc = Int(hkeys[unsafe_offset=h])
                if kc == c:
                    found = Int(hvals[unsafe_offset=h])
                    break
                if kc == -1:
                    break
                h = (h + 1) & mask
            self.aids[unsafe_offset=x] = Int32(found)

    def build(self, autojunk: Bool):
        """Build the b2j index in CSR form: for each id, its positions in b in ascending
        order, skipping junk and (when autojunk and n >= 200) popular ids."""
        var n = self.n
        var kk = self.k
        for t in range(kk):
            self.count[unsafe_offset=t] = 0
        for x in range(n):
            var t = self.bid(x)
            self.count[unsafe_offset=t] = self.count[unsafe_offset=t] + 1
        var ntest = n // 100 + 1
        var popular_on = autojunk and n >= 200
        for t in range(kk):
            var is_pop = popular_on and Int(self.count[unsafe_offset=t]) > ntest and Int(self.junk[unsafe_offset=t]) == 0
            self.pop[unsafe_offset=t] = Int32(1) if is_pop else Int32(0)
        # start[t] first holds the end offset of t's block, then (after the
        # reverse fill below) its begin offset; start[k] is the total.
        var total = 0
        for t in range(kk):
            if Int(self.junk[unsafe_offset=t]) == 0 and Int(self.pop[unsafe_offset=t]) == 0:
                total += Int(self.count[unsafe_offset=t])
            self.start[unsafe_offset=t] = Int32(total)
        self.start[unsafe_offset=kk] = Int32(total)
        var x = n - 1
        while x >= 0:
            var t = self.bid(x)
            if Int(self.junk[unsafe_offset=t]) == 0 and Int(self.pop[unsafe_offset=t]) == 0:
                var e = Int(self.start[unsafe_offset=t]) - 1
                self.start[unsafe_offset=t] = Int32(e)
                self.pos[unsafe_offset=e] = Int32(x)
            x -= 1

    def find_longest_match(mut self, alo: Int, ahi: Int, blo: Int, bhi: Int) -> Tuple[Int, Int, Int]:
        """Mirror of difflib.SequenceMatcher.find_longest_match. j2len is a
        sparse map keyed by j; here it is two dense rows (double-buffered by
        row parity) whose entries are valid only when stamped with the
        previous row's generation."""
        var besti = alo
        var bestj = blo
        var bestsize = 0
        var n = self.n
        var prev_gen = -1
        var cur = 0
        for i in range(alo, ahi):
            self.gen += 1
            var g = self.gen
            var ai = self.aid(i)
            if ai >= 0:
                var lo = Int(self.start[unsafe_offset=ai])
                var hi = Int(self.start[unsafe_offset=ai + 1])
                # first position in the block that is >= blo
                var l = lo
                var r = hi
                while l < r:
                    var mid = (l + r) // 2
                    if Int(self.pos[unsafe_offset=mid]) < blo:
                        l = mid + 1
                    else:
                        r = mid
                var p = l
                var base_cur = cur * n
                var base_prev = (1 - cur) * n
                while p < hi:
                    var j = Int(self.pos[unsafe_offset=p])
                    if j >= bhi:
                        break
                    var prevlen = 0
                    if j > 0 and Int(self.stamps[unsafe_offset=base_prev + j - 1]) == prev_gen:
                        prevlen = Int(self.lens[unsafe_offset=base_prev + j - 1])
                    var kk = prevlen + 1
                    self.lens[unsafe_offset=base_cur + j] = Int32(kk)
                    self.stamps[unsafe_offset=base_cur + j] = Int64(g)
                    if kk > bestsize:
                        besti = i - kk + 1
                        bestj = j - kk + 1
                        bestsize = kk
                    p += 1
            prev_gen = g
            cur = 1 - cur

        # Extend the best by non-junk elements on each end.
        while besti > alo and bestj > blo and not self.is_junk_pos(bestj - 1) and self.aid(besti - 1) == self.bid(bestj - 1):
            besti -= 1
            bestj -= 1
            bestsize += 1
        while besti + bestsize < ahi and bestj + bestsize < bhi and not self.is_junk_pos(bestj + bestsize) and self.aid(besti + bestsize) == self.bid(bestj + bestsize):
            bestsize += 1
        # Then absorb the junk on each side of it.
        while besti > alo and bestj > blo and self.is_junk_pos(bestj - 1) and self.aid(besti - 1) == self.bid(bestj - 1):
            besti -= 1
            bestj -= 1
            bestsize += 1
        while besti + bestsize < ahi and bestj + bestsize < bhi and self.is_junk_pos(bestj + bestsize) and self.aid(besti + bestsize) == self.bid(bestj + bestsize):
            bestsize += 1
        return (besti, bestj, bestsize)

    def matching_blocks(mut self, la: Int, outp: P64) -> Int:
        """Mirror of get_matching_blocks: the queue of sub-problems, the
        sorted blocks, and the adjacent-block collapse. Writes triples
        (i, j, k) into outp and returns the number of triples, including
        the (la, lb, 0) sentinel."""
        var lb = self.n
        var stack = List[Int]()
        stack.append(0)
        stack.append(la)
        stack.append(0)
        stack.append(lb)
        var m = 0
        while len(stack) > 0:
            var bhi = stack.pop()
            var blo = stack.pop()
            var ahi = stack.pop()
            var alo = stack.pop()
            var r = self.find_longest_match(alo, ahi, blo, bhi)
            var i = r[0]
            var j = r[1]
            var k = r[2]
            if k:
                outp[unsafe_offset=3 * m] = Int64(i)
                outp[unsafe_offset=3 * m + 1] = Int64(j)
                outp[unsafe_offset=3 * m + 2] = Int64(k)
                m += 1
                if alo < i and blo < j:
                    stack.append(alo)
                    stack.append(i)
                    stack.append(blo)
                    stack.append(j)
                if i + k < ahi and j + k < bhi:
                    stack.append(i + k)
                    stack.append(ahi)
                    stack.append(j + k)
                    stack.append(bhi)
        # Sort by i (the blocks lie in disjoint ranges of a, so i is unique).
        for t in range(1, m):
            var ci = Int(outp[unsafe_offset=3 * t])
            var cj = Int(outp[unsafe_offset=3 * t + 1])
            var ck = Int(outp[unsafe_offset=3 * t + 2])
            var s = t - 1
            while s >= 0 and Int(outp[unsafe_offset=3 * s]) > ci:
                outp[unsafe_offset=3 * (s + 1)] = outp[unsafe_offset=3 * s]
                outp[unsafe_offset=3 * (s + 1) + 1] = outp[unsafe_offset=3 * s + 1]
                outp[unsafe_offset=3 * (s + 1) + 2] = outp[unsafe_offset=3 * s + 2]
                s -= 1
            outp[unsafe_offset=3 * (s + 1)] = Int64(ci)
            outp[unsafe_offset=3 * (s + 1) + 1] = Int64(cj)
            outp[unsafe_offset=3 * (s + 1) + 2] = Int64(ck)
        # Collapse adjacent blocks in place: the write index never passes the
        # read index, and each block is read into locals before it is written.
        var w = 0
        var i1 = 0
        var j1 = 0
        var k1 = 0
        for t in range(m):
            var i2 = Int(outp[unsafe_offset=3 * t])
            var j2 = Int(outp[unsafe_offset=3 * t + 1])
            var k2 = Int(outp[unsafe_offset=3 * t + 2])
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    outp[unsafe_offset=3 * w] = Int64(i1)
                    outp[unsafe_offset=3 * w + 1] = Int64(j1)
                    outp[unsafe_offset=3 * w + 2] = Int64(k1)
                    w += 1
                i1 = i2
                j1 = j2
                k1 = k2
        if k1:
            outp[unsafe_offset=3 * w] = Int64(i1)
            outp[unsafe_offset=3 * w + 1] = Int64(j1)
            outp[unsafe_offset=3 * w + 2] = Int64(k1)
            w += 1
        outp[unsafe_offset=3 * w] = Int64(la)
        outp[unsafe_offset=3 * w + 1] = Int64(lb)
        outp[unsafe_offset=3 * w + 2] = Int64(0)
        return w + 1


def intern_b(ctx_addr: PythonObject) raises -> PythonObject:
    """Codepoint mode, first step: assign dense ids to the code points of b in
    first-occurrence order, fill the interning table, return K."""
    var ctx = Ctx(Int(py=ctx_addr))
    var n = ctx.get(S_N)
    var bcp = ctx.ptr32(S_BCP)
    var bids = ctx.ptr32(S_BIDS)
    var kcp = ctx.ptr32(S_KCP)
    var hkeys = ctx.ptr32(S_HKEYS)
    var hvals = ctx.ptr32(S_HVALS)
    var mask = ctx.get(S_HMASK)
    var k = 0
    for x in range(n):
        var c = Int(bcp[unsafe_offset=x])
        var h = slot_of(c, mask)
        var found: Int
        while True:
            var kc = Int(hkeys[unsafe_offset=h])
            if kc == c:
                found = Int(hvals[unsafe_offset=h])
                break
            if kc == -1:
                hkeys[unsafe_offset=h] = Int32(c)
                hvals[unsafe_offset=h] = Int32(k)
                kcp[unsafe_offset=k] = Int32(c)
                found = k
                k += 1
                break
            h = (h + 1) & mask
        bids[unsafe_offset=x] = Int32(found)
    ctx.put(S_K, k)
    return PythonObject(k)


def build_index(ctx_addr: PythonObject) raises -> PythonObject:
    """Build the b-side CSR index from the ids in S_BIDS and the junk flags."""
    var ctx = Ctx(Int(py=ctx_addr))
    var idx = Index(ctx)
    idx.build(ctx.get(S_AUTOJUNK) != 0)
    idx.save(ctx)
    return Python.none()


def find_longest_match(
    ctx_addr: PythonObject,
    alo: PythonObject,
    ahi: PythonObject,
    blo: PythonObject,
    bhi: PythonObject,
) raises -> PythonObject:
    var ctx = Ctx(Int(py=ctx_addr))
    var idx = Index(ctx)
    var a0 = Int(py=alo)
    var a1 = Int(py=ahi)
    if ctx.get(S_ACP) != 0:
        idx.intern_a(ctx, a1 - a0)
    var r = idx.find_longest_match(a0, a1, Int(py=blo), Int(py=bhi))
    idx.save(ctx)
    return Python.tuple(r[0], r[1], r[2])


def matching_blocks(ctx_addr: PythonObject) raises -> PythonObject:
    var ctx = Ctx(Int(py=ctx_addr))
    var idx = Index(ctx)
    var la = ctx.get(S_LA)
    if ctx.get(S_ACP) != 0:
        idx.intern_a(ctx, la)
    var m = idx.matching_blocks(la, ctx.ptr64(S_OUT))
    idx.save(ctx)
    return PythonObject(m)


@export
def PyInit__kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_kernel")
        m.def_function[intern_b]("intern_b", docstring="Intern the code points of b.")
        m.def_function[build_index]("build_index", docstring="Build the b-side CSR index.")
        m.def_function[find_longest_match]("find_longest_match", docstring="Longest matching block over ids.")
        m.def_function[matching_blocks]("matching_blocks", docstring="Matching blocks over ids.")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
