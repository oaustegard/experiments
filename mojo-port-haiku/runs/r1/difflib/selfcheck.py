"""Edge-case comparison of mdifflib.SequenceMatcher against difflib's, beyond the dev fuzzer."""
import difflib, random, sys
sys.path.insert(0, '.')
import mdifflib

def views(sm):
    return dict(
        blocks=[tuple(m) for m in sm.get_matching_blocks()],
        opcodes=sm.get_opcodes(),
        grouped=[list(g) for g in sm.get_grouped_opcodes(2)],
        ratio=sm.ratio(), quick=sm.quick_ratio(), rq=sm.real_quick_ratio(),
        b2j=sorted((repr(k), v) for k, v in sm.b2j.items()),
        bjunk=sorted(map(repr, sm.bjunk)), bpop=sorted(map(repr, sm.bpopular)),
        keys_order=[repr(k) for k in sm.b2j],
    )

def same(a, b, isjunk=None, autojunk=True):
    r = difflib.SequenceMatcher(isjunk, a, b, autojunk=autojunk)
    m = mdifflib.SequenceMatcher(isjunk, a, b, autojunk=autojunk)
    vr, vm = views(r), views(m)
    assert vr == vm, (repr(a)[:80], repr(b)[:80], isjunk, [k for k in vr if vr[k] != vm[k]])
    return True

cases = [
    ("", ""), ("", "abc"), ("abc", ""), ("a", "a"), ("ab", "ba"),
    ("a\ud800b", "ab"), ("ab", "a\ud800b"), ("a\ud800b", "a\ud800b"),
    ("x\x00y", "y\x00x"), ("\U0001F600ab", "ab\U0001F600"), ("日本語テキスト", "日本語テキ"),
    ("1", [1]), ("1", ["1"]), ([1, True, 1.0, "1"], "1"), ([1, 1.0, True], [True, 1]),
    ([None, b"y", (1, 2)], [(1, 2), None, b"y"]), (["a", 1, "b"], "ab"), ("ab", ["a", 1, "b", "x"]),
    ([float("nan")] * 2, [float("nan")] * 2),
]
for a, b in cases:
    for autojunk in (True, False):
        for isjunk in (None, lambda x: x == " " or x == 1, lambda x: False):
            try:
                same(a, b, isjunk, autojunk)
            except Exception as e:
                print("MISMATCH", repr(a)[:50], repr(b)[:50], autojunk, type(e).__name__, str(e)[:200])
print("edge cases done")

# long inputs exercising popular-element pruning (n >= 200) in both modes
rng = random.Random(5)
for trial in range(12):
    n = rng.choice([250, 600, 2000, 6000])
    a = "".join(rng.choice("eee tttaon\n") for _ in range(n))
    b = "".join(c if rng.random() > 0.05 else rng.choice("xyz") for c in a)
    for autojunk in (True, False):
        same(a, b, None, autojunk)
        same(a, b, lambda c: c == " ", autojunk)
    la = [f"row {rng.randint(0, 40)}\n" for _ in range(n // 3)]
    lb = [x if rng.random() > 0.1 else f"row {rng.randint(0, 40)}\n" for x in la]
    same(la, lb, None, True)
print("long cases done")

# set_seq1 / set_seq2 reuse and find_longest_match ranges on the same objects
for trial in range(200):
    a = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 30)))
    b = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 30)))
    r = difflib.SequenceMatcher(None)
    m = mdifflib.SequenceMatcher(None)
    for seq in [(a, b), (b, a), (a, a)]:
        r.set_seqs(*seq); m.set_seqs(*seq)
        for _ in range(10):
            alo = rng.randint(0, len(seq[0])); ahi = rng.randint(alo, len(seq[0]))
            blo = rng.randint(0, len(seq[1])); bhi = rng.randint(blo, len(seq[1]))
            assert tuple(r.find_longest_match(alo, ahi, blo, bhi)) == tuple(m.find_longest_match(alo, ahi, blo, bhi))
        assert [tuple(x) for x in r.get_matching_blocks()] == [tuple(x) for x in m.get_matching_blocks()]
print("reuse cases done")

# argument errors are raised, not silently wrong
m = mdifflib.SequenceMatcher(None, "abc", "abd")
for args in [(0, 4, 0, 3), (0, 3, 0, 4), (-1, 2, 0, 3)]:
    try:
        m.find_longest_match(*args); print("no error for", args)
    except IndexError:
        pass
print("selfcheck complete")
