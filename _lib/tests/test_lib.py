"""Tests for the shared helpers in `_lib/`.

Run: python3 -m pytest _lib/tests/ -q     (or: python3 _lib/tests/test_lib.py)

These cover the pieces that several experiments depend on, so a change
here that breaks them is caught before it breaks a pipeline that takes
hours to fail.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _lib.pipeline import chunked, load_json, retry, save_json  # noqa: E402
from _lib.paths import EXPERIMENTS_ROOT, experiment, spoke, spokes_root  # noqa: E402
from _lib.textnorm import ascii_fold  # noqa: E402


# --------------------------------------------------------------------- retry

def test_retry_returns_first_success():
    calls = []
    assert retry(lambda: (calls.append(1), "ok")[1], sleep=lambda _: None) == "ok"
    assert len(calls) == 1, "should not retry a call that succeeded"


def test_retry_succeeds_after_transient_failures():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert retry(flaky, sleep=lambda _: None, log=lambda _: None) == "ok"
    assert state["n"] == 3


def test_retry_reraises_last_exception():
    def always():
        raise ValueError("boom")

    try:
        retry(always, attempts=3, sleep=lambda _: None, log=lambda _: None)
    except ValueError as e:
        assert str(e) == "boom", "must surface the real error, not a wrapper"
    else:
        raise AssertionError("expected ValueError")


def test_retry_does_not_sleep_after_final_attempt():
    """The original slept after the last failure too — up to `cap` seconds of
    dead time before raising. Deliberate fix; pinned so it stays fixed."""
    waits = []

    def always():
        raise ValueError("boom")

    try:
        retry(always, attempts=4, sleep=waits.append, log=lambda _: None)
    except ValueError:
        pass
    assert len(waits) == 3, f"4 attempts should sleep 3 times, slept {len(waits)}"


def test_retry_backoff_is_capped_and_jittered():
    waits = []

    def always():
        raise ValueError("boom")

    try:
        retry(always, attempts=8, base=2.0, cap=5.0, sleep=waits.append, log=lambda _: None)
    except ValueError:
        pass
    assert all(w <= 5.5 for w in waits), f"cap+jitter exceeded: {waits}"
    assert waits[0] < waits[-1], "backoff should grow"


def test_retry_only_catches_listed_exceptions():
    def wrong_type():
        raise KeyError("not retryable")

    try:
        retry(wrong_type, retry_on=(ValueError,), sleep=lambda _: None, log=lambda _: None)
    except KeyError:
        pass
    else:
        raise AssertionError("should not have caught KeyError")


def test_retry_rejects_zero_attempts():
    try:
        retry(lambda: "x", attempts=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for attempts=0")


# ------------------------------------------------------------------- chunked

def test_chunked_splits_and_keeps_short_tail():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_exact_multiple_has_no_empty_tail():
    assert list(chunked(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]


def test_chunked_empty_yields_nothing():
    assert list(chunked([], 3)) == []


def test_chunked_consumes_lazily():
    def gen():
        yield from range(5)

    assert list(chunked(gen(), 2)) == [[0, 1], [2, 3], [4]]


def test_chunked_rejects_zero_size():
    try:
        list(chunked([1], 0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n=0")


# ---------------------------------------------------------------- checkpoint

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        save_json(p, {"a": [1, 2]})
        assert load_json(p) == {"a": [1, 2]}


def test_load_json_missing_returns_default():
    with tempfile.TemporaryDirectory() as d:
        assert load_json(Path(d) / "nope.json", default={"fallback": True}) == {"fallback": True}


def test_save_json_creates_parents():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "deep" / "nested" / "x.json"
        save_json(p, [1])
        assert p.exists()


def test_save_json_leaves_no_tmp_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        save_json(p, {"a": 1})
        assert list(Path(d).iterdir()) == [p], "atomic write should not leave a .tmp behind"


def test_save_json_overwrite_is_atomic_in_effect():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        save_json(p, {"v": 1})
        save_json(p, {"v": 2})
        assert json.loads(p.read_text()) == {"v": 2}


def test_save_json_serialises_non_json_types():
    """default=str keeps a checkpoint from dying on a stray Path or datetime."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        save_json(p, {"p": Path("/tmp/z")})
        assert load_json(p) == {"p": "/tmp/z"}


# ------------------------------------------------------------------ textnorm

def test_ascii_fold_folds_stroked_letters_nfkd_would_drop():
    # The bug this module exists for: plain NFKD drops the l-stroke.
    assert ascii_fold("Odrzywołek") == "odrzywolek"


def test_ascii_fold_covers_the_stroked_set():
    assert ascii_fold("Øst") == "ost"
    assert ascii_fold("Þór") == "thor"
    assert ascii_fold("Straße") == "strasse"
    assert ascii_fold("cœur") == "coeur"
    assert ascii_fold("Đặng") == "dang"


def test_ascii_fold_strips_ordinary_diacritics():
    assert ascii_fold("Erdős") == "erdos"
    assert ascii_fold("Gyárfás") == "gyarfas"


def test_ascii_fold_handles_empty_and_none():
    assert ascii_fold("") == ""
    assert ascii_fold(None) == ""  # type: ignore[arg-type]


def test_ascii_fold_is_idempotent():
    once = ascii_fold("Odrzywołek")
    assert ascii_fold(once) == once


# --------------------------------------------------------------------- paths

def test_experiments_root_is_this_repo():
    assert (EXPERIMENTS_ROOT / "_lib" / "paths.py").exists()


def test_experiment_resolves_a_real_sibling():
    assert experiment("ms13-campaign").is_dir()


def test_experiment_raises_on_unknown():
    try:
        experiment("no-such-experiment")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_spokes_root_honours_env(monkeypatch=None):
    import os
    prev = os.environ.get("EXPERIMENTS_SPOKES_ROOT")
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ["EXPERIMENTS_SPOKES_ROOT"] = d
            assert spokes_root() == Path(d).resolve()
            assert spoke("remax") == Path(d).resolve() / "remax"
    finally:
        if prev is None:
            os.environ.pop("EXPERIMENTS_SPOKES_ROOT", None)
        else:
            os.environ["EXPERIMENTS_SPOKES_ROOT"] = prev


def test_spoke_does_not_require_existence():
    assert spoke("definitely-not-checked-out").name == "definitely-not-checked-out"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - test runner reports and continues
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
