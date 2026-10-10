"""Mojo kernel for the yake port: everything after tokenization.

Python (myake/__init__.py) segments and tokenizes the text with segtok, then
hands over flat code-point buffers. This file does term statistics, the
co-occurrence graph, n-gram candidates, candidate scoring and deduplication.
Arithmetic follows yake 0.7.3 operation by operation so scores agree.
"""
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer
from std.math import sqrt
from std.ffi import external_call


def log(x: Float64) -> Float64:
    # libm's log, the one CPython's math.log calls; std.math.log is ~1e-10 off
    return external_call["log", Float64](x)


def addr_of(arr: PythonObject) raises -> Int:
    return Int(py=arr.__array_interface__["data"][0])


def mix(h: Int, c: Int) -> Int:
    return ((h ^ c) * 16777619) & 0xFFFFFFFF


def is_punct_cp(c: Int) -> Bool:
    # string.punctuation, the yake `exclude` set
    return (c >= 33 and c <= 47) or (c >= 58 and c <= 64) or (c >= 91 and c <= 96) or (c >= 123 and c <= 126)


def is_space_cp(c: Int) -> Bool:
    # str.isspace for the code points that matter here
    if c == 32 or (c >= 9 and c <= 13) or (c >= 28 and c <= 31):
        return True
    if c == 133 or c == 160 or c == 5760 or (c >= 8192 and c <= 8202):
        return True
    return c == 8232 or c == 8233 or c == 8239 or c == 8287 or c == 12288


def pairwise_sum(v: List[Float64], start: Int, n: Int) -> Float64:
    # numpy's pairwise summation, so mean/std round the same way
    if n < 8:
        var res = 0.0
        for i in range(n):
            res += v[start + i]
        return res
    if n <= 128:
        var r0 = v[start]
        var r1 = v[start + 1]
        var r2 = v[start + 2]
        var r3 = v[start + 3]
        var r4 = v[start + 4]
        var r5 = v[start + 5]
        var r6 = v[start + 6]
        var r7 = v[start + 7]
        var i = 8
        var lim = n - (n % 8)
        while i < lim:
            r0 += v[start + i]
            r1 += v[start + i + 1]
            r2 += v[start + i + 2]
            r3 += v[start + i + 3]
            r4 += v[start + i + 4]
            r5 += v[start + i + 5]
            r6 += v[start + i + 6]
            r7 += v[start + i + 7]
            i += 8
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += v[start + i]
            i += 1
        return res
    var n2 = n // 2
    n2 -= n2 % 8
    return pairwise_sum(v, start, n2) + pairwise_sum(v, start + n2, n - n2)


# ---------------------------------------------------------------- interning

struct SliceTable:
    """Open-addressing set of code-point slices; entry id = insertion order."""
    var slot: List[Int]
    var eoff: List[Int]
    var elen: List[Int]
    var ehash: List[Int]

    def __init__(out self):
        self.slot = List[Int](length=64, fill=-1)
        self.eoff = List[Int]()
        self.elen = List[Int]()
        self.ehash = List[Int]()


def slice_hash(p: Pointer[UInt32, MutUntrackedOrigin], off: Int, n: Int) -> Int:
    var h = 2166136261
    for i in range(n):
        h = mix(h, Int(p[unsafe_offset=off + i]))
    return h


def table_rehash(mut t: SliceTable):
    var cap = len(t.slot) * 2
    t.slot = List[Int](length=cap, fill=-1)
    var mask = cap - 1
    for e in range(len(t.eoff)):
        var idx = t.ehash[e] & mask
        while t.slot[idx] >= 0:
            idx = (idx + 1) & mask
        t.slot[idx] = e


def table_find(
    mut t: SliceTable,
    kb: Pointer[UInt32, MutUntrackedOrigin],
    qb: Pointer[UInt32, MutUntrackedOrigin],
    qoff: Int,
    qlen: Int,
    insert: Bool,
) -> Int:
    var h = slice_hash(qb, qoff, qlen)
    var mask = len(t.slot) - 1
    var idx = h & mask
    while True:
        var e = t.slot[idx]
        if e < 0:
            if not insert:
                return -1
            var nid = len(t.eoff)
            t.eoff.append(qoff)
            t.elen.append(qlen)
            t.ehash.append(h)
            t.slot[idx] = nid
            if 2 * len(t.eoff) > len(t.slot):
                table_rehash(t)
            return nid
        if t.ehash[e] == h and t.elen[e] == qlen:
            var eo = t.eoff[e]
            var same = True
            for i in range(qlen):
                if kb[unsafe_offset=eo + i] != qb[unsafe_offset=qoff + i]:
                    same = False
                    break
            if same:
                return e
        idx = (idx + 1) & mask


# ---------------------------------------------------------- string metrics
# All strings are code-point slices of one buffer: (buf, off, len).


def sorted_unique(v: List[Int]) -> List[Int]:
    var a = v.copy()
    var n = len(a)
    for i in range(1, n):
        var x = a[i]
        var j = i - 1
        while j >= 0 and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    var out = List[Int]()
    for i in range(n):
        if i == 0 or a[i] != a[i - 1]:
            out.append(a[i])
    return out^


def count_common(a: List[Int], b: List[Int]) -> Int:
    # both inputs sorted and unique
    var i = 0
    var j = 0
    var c = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            c += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return c


def slice_eq(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Bool:
    if al != bl:
        return False
    for i in range(al):
        if buf[ao + i] != buf[bo + i]:
            return False
    return True


def words_of(buf: List[Int], ao: Int, al: Int) -> List[Int]:
    # flat (start, len) pairs of whitespace-separated words
    var out = List[Int]()
    var i = 0
    while i < al:
        while i < al and is_space_cp(buf[ao + i]):
            i += 1
        if i >= al:
            break
        var s = i
        while i < al and not is_space_cp(buf[ao + i]):
            i += 1
        out.append(ao + s)
        out.append(i - s)
    return out^


def lev_distance(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Int:
    # yake's Levenshtein.distance, including its length-gap shortcut
    if al == 0:
        return bl
    if bl == 0:
        return al
    var mx = al if al > bl else bl
    var gap = al - bl if al > bl else bl - al
    if Float64(gap) > Float64(mx) * 0.7:
        return mx
    var s1 = ao
    var n1 = al
    var s2 = bo
    var n2 = bl
    if n1 > n2:
        s1 = bo
        n1 = bl
        s2 = ao
        n2 = al
    var prev = List[Int](length=n2 + 1, fill=0)
    var cur = List[Int](length=n2 + 1, fill=0)
    for j in range(n2 + 1):
        prev[j] = j
    for i in range(1, n1 + 1):
        cur[0] = i
        var ci = buf[s1 + i - 1]
        for j in range(1, n2 + 1):
            var cost = 0 if ci == buf[s2 + j - 1] else 1
            var a = cur[j - 1] + 1
            var b = prev[j] + 1
            var c = prev[j - 1] + cost
            var m = a if a < b else b
            cur[j] = m if m < c else c
        for j in range(n2 + 1):
            prev[j] = cur[j]
    return prev[n2]


def dedup_levs(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Float64:
    var mx = al if al > bl else bl
    return 1.0 - Float64(lev_distance(buf, ao, al, bo, bl)) / Float64(mx)


def jaro_sim(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Float64:
    # jellyfish.jaro_similarity over grapheme clusters (ids in buf), with the
    # integer half of transpositions
    if al == 0 or bl == 0:
        return 0.0
    var mx = al if al > bl else bl
    var rng = mx // 2 - 1
    if rng < 0:
        rng = 0
    var f1 = List[Bool](length=al, fill=False)
    var f2 = List[Bool](length=bl, fill=False)
    var m = 0
    for i in range(al):
        var lo = i - rng
        if lo < 0:
            lo = 0
        var hi = i + rng + 1
        if hi > bl:
            hi = bl
        for j in range(lo, hi):
            if (not f2[j]) and buf[ao + i] == buf[bo + j]:
                f1[i] = True
                f2[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    var k = 0
    var tr = 0
    for i in range(al):
        if f1[i]:
            while not f2[k]:
                k += 1
            if buf[ao + i] != buf[bo + k]:
                tr += 1
            k += 1
    var t = Float64(tr // 2)
    var fm = Float64(m)
    return (fm / Float64(al) + fm / Float64(bl) + (fm - t) / fm) / 3.0


def prefilter(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Bool:
    # yake's _aggressive_pre_filter
    if slice_eq(buf, ao, al, bo, bl):
        return True
    var mx = al if al > bl else bl
    var gap = al - bl if al > bl else bl - al
    if Float64(gap) > Float64(mx) * 0.6:
        return False
    if mx > 3:
        if buf[ao] != buf[bo] or buf[ao + al - 1] != buf[bo + bl - 1]:
            return False
        var mn = al if al < bl else bl
        if mn >= 3 and (buf[ao] != buf[bo] or buf[ao + 1] != buf[bo + 1]):
            return False
    var sa = 0
    for i in range(al):
        if buf[ao + i] == 32:
            sa += 1
    var sb = 0
    for i in range(bl):
        if buf[bo + i] == 32:
            sb += 1
    var dsp = sa - sb if sa > sb else sb - sa
    if dsp > 1:
        return False
    return True


def ultra_similarity(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Float64:
    # yake's _ultra_fast_similarity
    if slice_eq(buf, ao, al, bo, bl):
        return 1.0
    var mx = al if al > bl else bl
    if mx == 0:
        return 0.0
    var mn = al if al < bl else bl
    var len_ratio = Float64(mn) / Float64(mx)
    if len_ratio < 0.3:
        return 0.0
    var ca = List[Int]()
    for i in range(al):
        ca.append(buf[ao + i])
    var cb = List[Int]()
    for i in range(bl):
        cb.append(buf[bo + i])
    var sa = sorted_unique(ca)
    var sb = sorted_unique(cb)
    var inter = count_common(sa, sb)
    var uni = len(sa) + len(sb) - inter
    if uni == 0:
        return 0.0
    var char_overlap = Float64(inter) / Float64(uni)
    if char_overlap < 0.2:
        return 0.0
    if mx <= 4:
        return char_overlap * len_ratio
    var wa = words_of(buf, ao, al)
    var wb = words_of(buf, bo, bl)
    var nwa = len(wa) // 2
    var nwb = len(wb) // 2
    if nwa > 1 or nwb > 1:
        # unique words of each side
        var ua = List[Int]()
        for i in range(nwa):
            var dup = False
            for j in range(len(ua) // 2):
                if slice_eq(buf, ua[2 * j], ua[2 * j + 1], wa[2 * i], wa[2 * i + 1]):
                    dup = True
                    break
            if not dup:
                ua.append(wa[2 * i])
                ua.append(wa[2 * i + 1])
        var ub = List[Int]()
        for i in range(nwb):
            var dup = False
            for j in range(len(ub) // 2):
                if slice_eq(buf, ub[2 * j], ub[2 * j + 1], wb[2 * i], wb[2 * i + 1]):
                    dup = True
                    break
            if not dup:
                ub.append(wb[2 * i])
                ub.append(wb[2 * i + 1])
        var common = 0
        for i in range(len(ua) // 2):
            for j in range(len(ub) // 2):
                if slice_eq(buf, ua[2 * i], ua[2 * i + 1], ub[2 * j], ub[2 * j + 1]):
                    common += 1
                    break
        var wunion = len(ua) // 2 + len(ub) // 2 - common
        if wunion > 0:
            var word_overlap = Float64(common) / Float64(wunion)
            if word_overlap > 0.4:
                return word_overlap
    # trigrams packed into one Int each (21 bits per code point)
    var ta = List[Int]()
    for i in range(al - 2):
        ta.append((buf[ao + i] << 42) | (buf[ao + i + 1] << 21) | buf[ao + i + 2])
    var tb = List[Int]()
    for i in range(bl - 2):
        tb.append((buf[bo + i] << 42) | (buf[bo + i + 1] << 21) | buf[bo + i + 2])
    var ga = sorted_unique(ta)
    var gb = sorted_unique(tb)
    var tinter = count_common(ga, gb)
    var tuni = len(ga) + len(gb) - tinter
    var trigram_overlap = 0.0
    if tuni > 0:
        trigram_overlap = Float64(tinter) / Float64(tuni)
    var v = 0.3 * len_ratio + 0.2 * char_overlap + 0.5 * trigram_overlap
    return v if v < 1.0 else 1.0


def dedup_seqm(buf: List[Int], ao: Int, al: Int, bo: Int, bl: Int) -> Float64:
    if prefilter(buf, ao, al, bo, bl):
        return ultra_similarity(buf, ao, al, bo, bl)
    return 0.0


# -------------------------------------------------------------- the engine


def sort_by_key(mut a: List[Int], key: List[Float64]):
    # stable bottom-up merge sort on candidate ids, ordered by key
    var n = len(a)
    var buf = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var i = 0
        while i < n:
            var mid = i + width
            if mid > n:
                mid = n
            var hi = i + 2 * width
            if hi > n:
                hi = n
            var l = i
            var r = mid
            var o = i
            while l < mid and r < hi:
                if key[a[r]] < key[a[l]]:
                    buf[o] = a[r]
                    r += 1
                else:
                    buf[o] = a[l]
                    l += 1
                o += 1
            while l < mid:
                buf[o] = a[l]
                l += 1
                o += 1
            while r < hi:
                buf[o] = a[r]
                r += 1
                o += 1
            i += 2 * width
        for j in range(n):
            a[j] = buf[j]
        width *= 2


struct Engine:
    var T: Int
    var NS: Int
    var nmax: Int
    var window: Int
    var ob: Pointer[UInt32, MutUntrackedOrigin]
    var oo: Pointer[Int64, MutUntrackedOrigin]
    var lb: Pointer[UInt32, MutUntrackedOrigin]
    var lo: Pointer[Int64, MutUntrackedOrigin]
    var sb: Pointer[UInt32, MutUntrackedOrigin]
    var cls: Pointer[UInt16, MutUntrackedOrigin]
    var clen: Int
    var sent: Pointer[Int64, MutUntrackedOrigin]
    var sstart: Pointer[Int64, MutUntrackedOrigin]
    var lt_tab: SliceTable
    var sw_tab: SliceTable
    var term_tab: SliceTable
    var tok_lt: List[Int]
    var tok_tag: List[Int]
    var tok_term: List[Int]
    var lt_sw: List[Int]
    var lt_term: List[Int]
    var tm_tf: List[Int]
    var tm_tfa: List[Int]
    var tm_tfn: List[Int]
    var tm_sw: List[Int]
    var tm_last: List[Int]
    var tm_nsent: List[Int]
    var tm_h: List[Float64]
    var pair_term: List[Int]
    var pair_sent: List[Int]
    var ed_map: Dict[Int, Int]
    var ed_src: List[Int]
    var ed_dst: List[Int]
    var ed_tf: List[Int]
    var pool: List[Int]
    var cs_slot: List[Int]
    var cd_pool: List[Int]
    var cd_k: List[Int]
    var cd_tf: List[Int]
    var cd_first: List[Int]
    var cd_valid: List[Int]
    var cd_bad: List[Int]
    var cd_hash: List[Int]
    var cd_h: List[Float64]
    var kwb: List[Int]
    var kwo: List[Int]
    var kwl: List[Int]
    var cl_tab: SliceTable
    var cl_tab2: SliceTable
    var clpool: List[Int]
    var lt_cl_start: List[Int]
    var lt_cl_n: List[Int]
    var lt_first_sp: List[Int]
    var jb: List[Int]
    var jo: List[Int]
    var jl: List[Int]

    def __init__(
        out self,
        T: Int,
        NS: Int,
        nmax: Int,
        window: Int,
        ob: Pointer[UInt32, MutUntrackedOrigin],
        oo: Pointer[Int64, MutUntrackedOrigin],
        lb: Pointer[UInt32, MutUntrackedOrigin],
        lo: Pointer[Int64, MutUntrackedOrigin],
        sb: Pointer[UInt32, MutUntrackedOrigin],
        cls: Pointer[UInt16, MutUntrackedOrigin],
        clen: Int,
        sent: Pointer[Int64, MutUntrackedOrigin],
        sstart: Pointer[Int64, MutUntrackedOrigin],
    ):
        self.T = T
        self.NS = NS
        self.nmax = nmax
        self.window = window
        self.ob = ob
        self.oo = oo
        self.lb = lb
        self.lo = lo
        self.sb = sb
        self.cls = cls
        self.clen = clen
        self.sent = sent
        self.sstart = sstart
        self.lt_tab = SliceTable()
        self.sw_tab = SliceTable()
        self.term_tab = SliceTable()
        self.tok_lt = List[Int](length=T, fill=-1)
        self.tok_tag = List[Int](length=T, fill=-1)
        self.tok_term = List[Int](length=T, fill=-1)
        self.lt_sw = List[Int]()
        self.lt_term = List[Int]()
        self.tm_tf = List[Int]()
        self.tm_tfa = List[Int]()
        self.tm_tfn = List[Int]()
        self.tm_sw = List[Int]()
        self.tm_last = List[Int]()
        self.tm_nsent = List[Int]()
        self.tm_h = List[Float64]()
        self.pair_term = List[Int]()
        self.pair_sent = List[Int]()
        self.ed_map = Dict[Int, Int]()
        self.ed_src = List[Int]()
        self.ed_dst = List[Int]()
        self.ed_tf = List[Int]()
        self.pool = List[Int]()
        self.cs_slot = List[Int](length=256, fill=-1)
        self.cd_pool = List[Int]()
        self.cd_k = List[Int]()
        self.cd_tf = List[Int]()
        self.cd_first = List[Int]()
        self.cd_valid = List[Int]()
        self.cd_bad = List[Int]()
        self.cd_hash = List[Int]()
        self.cd_h = List[Float64]()
        self.kwb = List[Int]()
        self.kwo = List[Int]()
        self.kwl = List[Int]()
        self.cl_tab = SliceTable()
        self.cl_tab2 = SliceTable()
        self.clpool = List[Int]()
        self.lt_cl_start = List[Int]()
        self.lt_cl_n = List[Int]()
        self.lt_first_sp = List[Int]()
        self.jb = List[Int]()
        self.jo = List[Int]()
        self.jl = List[Int]()

    # ---- token access
    def ow_off(self, t: Int) -> Int:
        return Int(self.oo[unsafe_offset=t])

    def ow_len(self, t: Int) -> Int:
        return Int(self.oo[unsafe_offset=t + 1]) - Int(self.oo[unsafe_offset=t])

    def lw_off(self, t: Int) -> Int:
        return Int(self.lo[unsafe_offset=t])

    def lw_len(self, t: Int) -> Int:
        return Int(self.lo[unsafe_offset=t + 1]) - Int(self.lo[unsafe_offset=t])

    def is_punct(self, t: Int) -> Bool:
        var off = self.ow_off(t)
        for i in range(self.ow_len(t)):
            if not is_punct_cp(Int(self.ob[unsafe_offset=off + i])):
                return False
        return True

    def flag(self, c: Int) -> Int:
        if c < self.clen:
            return Int(self.cls[unsafe_offset=c])
        return 0

    def tag_of(self, t: Int, i: Int) -> Int:
        # yake's get_tag: 0 d, 1 u, 2 a, 3 n, 4 p
        # flags: 1 isdigit, 2 isalpha, 4 isupper, 8 islower, 16 istitle
        var off = self.ow_off(t)
        var n = self.ow_len(t)
        # digits: word.replace(",", "").isdigit() or with one "." removed
        var cnt = 0
        var all_dig = True
        for j in range(n):
            var c = Int(self.ob[unsafe_offset=off + j])
            if c == 44:
                continue
            cnt += 1
            if (self.flag(c) & 1) == 0:
                all_dig = False
                break
        if cnt > 0 and all_dig:
            return 0
        cnt = 0
        all_dig = True
        var dot_seen = False
        for j in range(n):
            var c = Int(self.ob[unsafe_offset=off + j])
            if c == 44:
                continue
            if c == 46 and not dot_seen:
                dot_seen = True
                continue
            cnt += 1
            if (self.flag(c) & 1) == 0:
                all_dig = False
                break
        if cnt > 0 and all_dig:
            return 0
        var cdig = 0
        var calp = 0
        var cexc = 0
        var cup = 0
        var lower_or_title = False
        for j in range(n):
            var c = Int(self.ob[unsafe_offset=off + j])
            var f = self.flag(c)
            if (f & 1) != 0:
                cdig += 1
            if (f & 2) != 0:
                calp += 1
            if is_punct_cp(c):
                cexc += 1
            if (f & 4) != 0:
                cup += 1
            if (f & 24) != 0:
                lower_or_title = True
        if (cdig > 0 and calp > 0) or (cdig == 0 and calp == 0) or cexc > 1:
            return 1
        if n > 0 and (not lower_or_title) and cup > 0:
            return 2
        if n > 1 and i > 0 and (self.flag(Int(self.ob[unsafe_offset=off])) & 4) != 0 and cup == 1:
            return 3
        return 4

    def lowtok(mut self, t: Int) -> Int:
        var off = self.lw_off(t)
        var n = self.lw_len(t)
        var before = len(self.lt_tab.eoff)
        var k = table_find(self.lt_tab, self.lb, self.lb, off, n, True)
        if k == before:
            var s = table_find(self.sw_tab, self.sb, self.lb, off, n, False)
            self.lt_sw.append(1 if s >= 0 else 0)
            self.lt_term.append(-1)
            self.make_clusters(k)
        return k

    def make_clusters(mut self, k: Int):
        # Grapheme clusters of one lowercase token, UAX #29 as Rust's jaro sees
        # them: extend (marks, ZWJ, skin tones) attach to the cluster before;
        # ZWJ joins a pictographic after a pictographic base; regional
        # indicators pair up. Ids: 2*entry for a cluster, 2*entry+1 for a
        # cluster that begins a token whose first character attaches to the
        # separating space.
        var off = self.lt_tab.eoff[k]
        var n = self.lt_tab.elen[k]
        self.lt_cl_start.append(len(self.clpool))
        var starts = List[Int]()
        var prev_f = 0
        var ri_run = 0
        var pict_base = False
        var zwj_ok = False
        for q in range(n):
            var f = self.flag(Int(self.lb[unsafe_offset=off + q]))
            var brk = True
            if q == 0:
                brk = True
            elif (f & 64) != 0:
                brk = False
            elif (f & 32) != 0:
                brk = False
            elif (prev_f & 64) != 0 and zwj_ok and (f & 128) != 0:
                brk = False
            elif (f & 256) != 0 and (prev_f & 256) != 0 and ri_run % 2 == 1:
                brk = False
            if brk:
                starts.append(q)
            if (f & 64) != 0:
                zwj_ok = pict_base
                pict_base = False
            elif (f & 32) == 0:
                pict_base = (f & 128) != 0
            ri_run = ri_run + 1 if (f & 256) != 0 else 0
            prev_f = f
        starts.append(n)
        var first_sp = -1
        if n > 0 and (self.flag(Int(self.lb[unsafe_offset=off])) & 32) != 0:
            first_sp = 2 * table_find(self.cl_tab2, self.lb, self.lb, off, starts[1], True) + 1
        self.lt_first_sp.append(first_sp)
        var cnt = len(starts) - 1
        for i in range(cnt):
            var cid = table_find(self.cl_tab, self.lb, self.lb, off + starts[i], starts[i + 1] - starts[i], True)
            self.clpool.append(2 * cid)
        self.lt_cl_n.append(cnt)

    def get_term(mut self, k: Int) -> Int:
        if self.lt_term[k] >= 0:
            return self.lt_term[k]
        var off = self.lt_tab.eoff[k]
        var n = self.lt_tab.elen[k]
        var kl = n
        if n > 3 and Int(self.lb[unsafe_offset=off + n - 1]) == 115:
            kl = n - 1
        var before = len(self.term_tab.eoff)
        var tid = table_find(self.term_tab, self.lb, self.lb, off, kl, True)
        if tid == before:
            var in_sw = table_find(self.sw_tab, self.sb, self.lb, off, kl, False) >= 0
            var bare = 0
            for i in range(kl):
                if not is_punct_cp(Int(self.lb[unsafe_offset=off + i])):
                    bare += 1
            var stop = self.lt_sw[k] == 1 or in_sw or bare < 3
            self.tm_tf.append(0)
            self.tm_tfa.append(0)
            self.tm_tfn.append(0)
            self.tm_sw.append(1 if stop else 0)
            self.tm_last.append(-1)
            self.tm_nsent.append(0)
            self.tm_h.append(0.0)
        self.lt_term[k] = tid
        return tid

    def ed_find(self, a: Int, b: Int) -> Int:
        var key = a * 4294967296 + b
        return self.ed_map.get(key, -1)

    def add_cooccur(mut self, a: Int, b: Int):
        var key = a * 4294967296 + b
        var e = self.ed_map.get(key, -1)
        if e >= 0:
            self.ed_tf[e] += 1
        else:
            self.ed_map[key] = len(self.ed_src)
            self.ed_src.append(a)
            self.ed_dst.append(b)
            self.ed_tf.append(1)

    def cs_rehash(mut self):
        var cap = len(self.cs_slot) * 2
        self.cs_slot = List[Int](length=cap, fill=-1)
        var mask = cap - 1
        for e in range(len(self.cd_k)):
            var idx = self.cd_hash[e] & mask
            while self.cs_slot[idx] >= 0:
                idx = (idx + 1) & mask
            self.cs_slot[idx] = e

    def cand_add(mut self, seq: List[Int], first: Int, clean: Bool):
        var k = len(seq)
        var h = 2166136261
        for i in range(k):
            h = mix(h, seq[i])
        var mask = len(self.cs_slot) - 1
        var idx = h & mask
        var found = -1
        while True:
            var e = self.cs_slot[idx]
            if e < 0:
                break
            if self.cd_hash[e] == h and self.cd_k[e] == k:
                var po = self.cd_pool[e]
                var same = True
                for i in range(k):
                    if self.pool[po + i] != seq[i]:
                        same = False
                        break
                if same:
                    found = e
                    break
            idx = (idx + 1) & mask
        if found < 0:
            found = len(self.cd_k)
            self.cd_pool.append(len(self.pool))
            for i in range(k):
                self.pool.append(seq[i])
            self.cd_k.append(k)
            self.cd_tf.append(0)
            self.cd_first.append(first)
            self.cd_valid.append(0)
            var bad = self.tm_sw[self.tok_term[first]] == 1 or self.tm_sw[self.lt_term[seq[k - 1]]] == 1
            self.cd_bad.append(1 if bad else 0)
            self.cd_hash.append(h)
            self.cd_h.append(0.0)
            self.cs_slot[idx] = found
            if 2 * len(self.cd_k) > len(self.cs_slot):
                self.cs_rehash()
        self.cd_tf[found] += 1
        if clean:
            self.cd_valid[found] = 1

    def run(mut self):
        # pass over tokens: term stats, co-occurrence edges, candidates
        var block = List[Int]()
        var cur = -1
        for t in range(self.T):
            var s = Int(self.sent[unsafe_offset=t])
            if s != cur:
                cur = s
                block = List[Int]()
            var sst = Int(self.sstart[unsafe_offset=t])
            if self.is_punct(t):
                block = List[Int]()
                continue
            var tag = self.tag_of(t, t - sst)
            self.tok_tag[t] = tag
            var k = self.lowtok(t)
            self.tok_lt[t] = k
            var term = self.get_term(k)
            self.tok_term[t] = term
            self.tm_tf[term] += 1
            if tag == 2:
                self.tm_tfa[term] += 1
            if tag == 3:
                self.tm_tfn[term] += 1
            if self.tm_last[term] != s:
                self.tm_last[term] = s
                self.tm_nsent[term] += 1
                self.pair_term.append(term)
                self.pair_sent.append(s)
            var blen = len(block)
            if tag >= 2:
                var wlo = blen - self.window
                if wlo < 0:
                    wlo = 0
                for w in range(wlo, blen):
                    var tw = block[w]
                    if self.tok_tag[tw] >= 2:
                        self.add_cooccur(self.tok_term[tw], term)
            var one = List[Int]()
            one.append(k)
            self.cand_add(one, t, tag >= 2)
            var clo = blen - (self.nmax - 1)
            if clo < 0:
                clo = 0
            var w = blen - 1
            while w >= clo:
                var seq = List[Int]()
                var clean = tag >= 2
                for j in range(w, blen):
                    seq.append(self.tok_lt[block[j]])
                    if self.tok_tag[block[j]] < 2:
                        clean = False
                seq.append(k)
                self.cand_add(seq, block[w], clean)
                w -= 1
            block.append(t)

    def term_features(mut self):
        # yake's SingleWord.update_h for every term (skipped when no valid term)
        var nt = len(self.tm_tf)
        var maxtf = 0
        var vals = List[Float64]()
        for i in range(nt):
            if self.tm_tf[i] > maxtf:
                maxtf = self.tm_tf[i]
            if self.tm_sw[i] == 0:
                vals.append(Float64(self.tm_tf[i]))
        var nvalid = len(vals)
        if nvalid == 0:
            return
        var avg = pairwise_sum(vals, 0, nvalid) / Float64(nvalid)
        var dev = List[Float64]()
        for i in range(nvalid):
            var d = vals[i] - avg
            dev.append(d * d)
        var std = sqrt(pairwise_sum(dev, 0, nvalid) / Float64(nvalid))
        var out_deg = List[Int](length=nt, fill=0)
        var out_sum = List[Int](length=nt, fill=0)
        var in_deg = List[Int](length=nt, fill=0)
        var in_sum = List[Int](length=nt, fill=0)
        for e in range(len(self.ed_src)):
            var a = self.ed_src[e]
            var b = self.ed_dst[e]
            var f = self.ed_tf[e]
            out_deg[a] += 1
            out_sum[a] += f
            in_deg[b] += 1
            in_sum[b] += f
        # sorted distinct sentence ids per term, flat
        var base = List[Int](length=nt, fill=0)
        var acc = 0
        for i in range(nt):
            base[i] = acc
            acc += self.tm_nsent[i]
        var fillp = List[Int](length=nt, fill=0)
        var occ = List[Int](length=acc, fill=0)
        for p in range(len(self.pair_term)):
            var tt = self.pair_term[p]
            occ[base[tt] + fillp[tt]] = self.pair_sent[p]
            fillp[tt] += 1
        var maxf = Float64(maxtf)
        for i in range(nt):
            var tf = Float64(self.tm_tf[i])
            var pwr = 0.0
            if out_sum[i] != 0:
                pwr = Float64(out_deg[i]) / Float64(out_sum[i])
            var pwl = 0.0
            if in_sum[i] != 0:
                pwl = Float64(in_deg[i]) / Float64(in_sum[i])
            var wrel = (0.5 + pwl * (tf / maxf)) + (0.5 + pwr * (tf / maxf))
            var wfreq = tf / (avg + std)
            var m = self.tm_nsent[i]
            var wspread = Float64(m) / Float64(self.NS)
            var tfa = self.tm_tfa[i]
            var tfn = self.tm_tfn[i]
            var wcase = Float64(tfa if tfa > tfn else tfn) / (1.0 + log(tf))
            var b0 = base[i]
            var med: Float64
            if m % 2 == 1:
                med = Float64(occ[b0 + m // 2])
            else:
                med = Float64(occ[b0 + m // 2 - 1] + occ[b0 + m // 2]) / 2.0
            var wpos = log(log(3.0 + med))
            self.tm_h[i] = (wpos * wrel) / (wcase + (wfreq / wrel) + (wspread / wrel))

    def cand_scores(mut self):
        # yake's ComposedWord.update_h for valid candidates
        for c in range(len(self.cd_k)):
            if self.cd_valid[c] == 0 or self.cd_bad[c] == 1:
                continue
            var k = self.cd_k[c]
            var po = self.cd_pool[c]
            var sum_h = 0.0
            var prod_h = 1.0
            for t in range(k):
                var term = self.lt_term[self.pool[po + t]]
                if self.tm_sw[term] == 0:
                    sum_h += self.tm_h[term]
                    prod_h *= self.tm_h[term]
                else:
                    var prob_t1 = 0.0
                    if t > 0:
                        var prev = self.lt_term[self.pool[po + t - 1]]
                        var e = self.ed_find(prev, term)
                        if e >= 0:
                            prob_t1 = Float64(self.ed_tf[e]) / Float64(self.tm_tf[prev])
                    var prob_t2 = 0.0
                    if t < k - 1:
                        var nxt = self.lt_term[self.pool[po + t + 1]]
                        var e2 = self.ed_find(term, nxt)
                        if e2 >= 0:
                            prob_t2 = Float64(self.ed_tf[e2]) / Float64(self.tm_tf[nxt])
                    var prob = prob_t1 * prob_t2
                    prod_h *= 1.0 + (1.0 - prob)
                    sum_h -= 1.0 - prob
            self.cd_h[c] = prod_h / ((sum_h + 1.0) * Float64(self.cd_tf[c]))

    def select(mut self, top: Int, dedup_mode: Int, no_dedup: Bool, dedup_lim: Float64) -> List[Int]:
        var order = List[Int]()
        for c in range(len(self.cd_k)):
            if self.cd_valid[c] == 1 and self.cd_bad[c] == 0:
                order.append(c)
        sort_by_key(order, self.cd_h)
        var out = List[Int]()
        if no_dedup:
            var L = len(order)
            var cnt: Int
            if top >= 0:
                cnt = top if top < L else L
            else:
                cnt = L + top
                if cnt < 0:
                    cnt = 0
            for i in range(cnt):
                out.append(order[i])
            return out^
        # unique lowercase strings of the candidates, joined by single spaces
        self.kwo = List[Int](length=len(self.cd_k), fill=0)
        self.kwl = List[Int](length=len(self.cd_k), fill=0)
        self.kwb = List[Int]()
        if dedup_mode == 3:
            # jaro compares grapheme clusters of the lowercase keyword string
            self.jo = List[Int](length=len(self.cd_k), fill=0)
            self.jl = List[Int](length=len(self.cd_k), fill=0)
            self.jb = List[Int]()
            for i in range(len(order)):
                var c = order[i]
                var start = len(self.jb)
                var po = self.cd_pool[c]
                for j in range(self.cd_k[c]):
                    var lk = self.pool[po + j]
                    var cs = self.lt_cl_start[lk]
                    var cn = self.lt_cl_n[lk]
                    if j == 0:
                        for q in range(cn):
                            self.jb.append(self.clpool[cs + q])
                    elif self.lt_first_sp[lk] >= 0:
                        self.jb.append(self.lt_first_sp[lk])
                        for q in range(1, cn):
                            self.jb.append(self.clpool[cs + q])
                    else:
                        self.jb.append(-1)
                        for q in range(cn):
                            self.jb.append(self.clpool[cs + q])
                self.jo[c] = start
                self.jl[c] = len(self.jb) - start
        else:
            for i in range(len(order)):
                var c = order[i]
                var start = len(self.kwb)
                var po = self.cd_pool[c]
                for j in range(self.cd_k[c]):
                    if j > 0:
                        self.kwb.append(32)
                    var lk = self.pool[po + j]
                    var off = self.lt_tab.eoff[lk]
                    for q in range(self.lt_tab.elen[lk]):
                        self.kwb.append(Int(self.lb[unsafe_offset=off + q]))
                self.kwo[c] = start
                self.kwl[c] = len(self.kwb) - start
        for i in range(len(order)):
            var c = order[i]
            var should_add = True
            for p in range(len(out)):
                var prev = out[p]
                var sim: Float64
                if dedup_mode == 1:
                    sim = dedup_seqm(self.kwb, self.kwo[c], self.kwl[c], self.kwo[prev], self.kwl[prev])
                elif dedup_mode == 3:
                    sim = jaro_sim(self.jb, self.jo[c], self.jl[c], self.jo[prev], self.jl[prev])
                else:
                    sim = dedup_levs(self.kwb, self.kwo[c], self.kwl[c], self.kwo[prev], self.kwl[prev])
                if sim > dedup_lim:
                    should_add = False
                    break
            if should_add:
                out.append(c)
            if len(out) == top:
                break
        return out^


def extract(ctl: PythonObject, fl: PythonObject) raises -> PythonObject:
    var c = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr_of(ctl))
    var T = Int(c[unsafe_offset=0])
    var NS = Int(c[unsafe_offset=1])
    var nmax = Int(c[unsafe_offset=2])
    var window = Int(c[unsafe_offset=3])
    var dedup_mode = Int(c[unsafe_offset=4])
    var no_dedup = Int(c[unsafe_offset=5]) == 1
    var top = Int(c[unsafe_offset=6])
    var clen = Int(c[unsafe_offset=7])
    var ob = Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=8]))
    var oo = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=9]))
    var lb = Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=10]))
    var lo = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=11]))
    var sb = Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=12]))
    var cls = Pointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=13]))
    var sent = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=14]))
    var sstart = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=15]))
    var out_first = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=16]))
    var out_len = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=17]))
    var out_h = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=18]))
    var f = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr_of(fl))
    var swo = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(c[unsafe_offset=19]))
    var nsw = Int(c[unsafe_offset=20])
    var dedup_lim = f[unsafe_offset=0]

    var eng = Engine(T, NS, nmax, window, ob, oo, lb, lo, sb, cls, clen, sent, sstart)
    for i in range(nsw):
        var so = Int(swo[unsafe_offset=i])
        _ = table_find(eng.sw_tab, sb, sb, so, Int(swo[unsafe_offset=i + 1]) - so, True)
    eng.run()
    eng.term_features()
    eng.cand_scores()
    var sel = eng.select(top, dedup_mode, no_dedup, dedup_lim)
    for i in range(len(sel)):
        var cid = sel[i]
        out_first[unsafe_offset=i] = Int64(eng.cd_first[cid])
        out_len[unsafe_offset=i] = Int64(eng.cd_k[cid])
        out_h[unsafe_offset=i] = eng.cd_h[cid]
    return PythonObject(len(sel))


@export
def PyInit__kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_kernel")
        m.def_function[extract]("extract", docstring="yake kernel: candidates, scores, dedup")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
