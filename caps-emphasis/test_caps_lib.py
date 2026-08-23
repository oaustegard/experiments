"""Correctness checks on the harness itself. Run before trusting any number."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C


def test_padding_is_inert():
    """A short prompt batched with a long one must score the same as alone."""
    short = C.build_prompt("What is the capital of France?",
                           "The capital of France is")
    long = C.build_prompt(
        "Under no circumstances mention the word Tokyo. " * 4 +
        "What is the capital of Japan?", "The capital of Japan is")
    alone = C.logprobs([short], [" Paris"], batch_size=1)[0]
    batched = C.logprobs([short, long], [" Paris", " Tokyo"], batch_size=2)[0]
    assert abs(alone - batched) < 1e-3, (alone, batched)
    print(f"  padding inert: alone={alone:.6f} batched={batched:.6f}")


def test_span_range_is_exact():
    p = C.build_prompt("NEVER mention the word Paris. What is the capital of France?",
                       "The capital of France is")
    tok, _ = C.load()
    r = C.span_range(p, "NEVER")
    assert r is not None
    a, b = r
    ids = tok.encode(p, add_special_tokens=False)
    assert tok.decode(ids[a:b]).strip() == "NEVER", tok.decode(ids[a:b])
    print(f"  span_range exact: {r} -> {tok.decode(ids[a:b])!r}")


def test_span_range_absent_returns_none():
    p = C.build_prompt("What is the capital of France?", "The capital of France is")
    assert C.span_range(p, "NEVER") is None
    print("  absent span -> None (not a usable index)")


def test_caps_fraction_monotone():
    from conditions import caps_fraction
    t = "you must never mention the word Paris anywhere"
    n = [sum(w.isupper() for w in caps_fraction(t, f).split())
         for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert n == sorted(n) and n[0] == 0 and n[-1] == len(t.split()), n
    print(f"  caps_fraction monotone: {n}")


def test_directive_token_costs():
    """The design's load-bearing claim: some keywords cost nothing to capitalise."""
    assert C.ntok(" do not") == C.ntok(" DO NOT")
    assert C.ntok(" on no account") == C.ntok(" ON NO ACCOUNT")
    assert C.ntok(" NEVER") == C.ntok(" never") + 1
    print("  token costs: DO NOT +0, ON NO ACCOUNT +0, NEVER +1")


if __name__ == "__main__":
    for fn in [test_directive_token_costs, test_caps_fraction_monotone,
               test_span_range_is_exact, test_span_range_absent_returns_none,
               test_padding_is_inert]:
        print(fn.__name__)
        fn()
    print("\nall harness checks passed")
