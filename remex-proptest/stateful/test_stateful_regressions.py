"""Deterministic regression tests for the findings of the stateful hunt (area C).

Every test in the first section fails on remex 2.2.0 (the revision under test) for the
reason in its docstring, and passes once the named root cause is fixed.  Tests marked
``control`` pass today; they pin the neighbouring behaviour that works, so a failure
elsewhere is attributable.  The ``AMBIGUOUS`` section holds ``xfail(strict=True)`` tests
for behaviour whose intended contract is not written down.

No Hypothesis, no randomness beyond fixed seeds, no files outside tmp_path.
Run:  python3 -m pytest test_stateful_regressions.py -q
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from remex import CompressedVectors, PackedVectors, Quantizer, RemexIndex, corpus_mean
from remex.gpu import GPUSearcher
from remex.ivf import IVFCoarseIndex


def _gauss(n, d, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


# ===========================================================================
# F1  A rejected first batch poisons a model-less index (remex/st.py)
# ===========================================================================

def test_f1_empty_add_on_modelless_index_is_a_valueerror():
    """``RemexIndex(bits=4).add([])`` should be rejected like every other empty batch
    (``_embed`` raises ValueError("expected a non-empty (n, d) batch")).  The list ``[]``
    becomes a (1, 0) array after the 1-D promotion at st.py:312-313, slips past the
    emptiness test at 314, and the index then dies with ZeroDivisionError building a d=0
    Quantizer."""
    idx = RemexIndex(bits=4)
    try:
        idx.add([])
    except Exception as exc:                       # noqa: BLE001
        raised = exc
    else:
        raised = None
    assert isinstance(raised, ValueError), f"add([]) raised {type(raised).__name__}: {raised}"


def test_f1_rejected_empty_add_leaves_dim_unset():
    """A rejected ``add`` must not change the index: ``_check_dim`` (st.py:304-308) records
    ``self.dim`` from the batch before the batch is accepted, so ``add([])`` leaves dim=0."""
    idx = RemexIndex(bits=4)
    with pytest.raises(Exception):
        idx.add([])
    assert idx.dim is None, f"rejected add left dim={idx.dim!r}"


def test_f1_modelless_index_is_usable_after_rejected_empty_add():
    """Consequence of the above: the next valid add is refused with 'expected dim=0, got 16'
    and the index can never be used again."""
    idx = RemexIndex(bits=4)
    with pytest.raises(Exception):
        idx.add([])
    ids = idx.add(_gauss(5, 16))
    assert list(ids) == [0, 1, 2, 3, 4] and len(idx) == 5


def test_f1_rejected_mixed_dim_maxsim_batch_leaves_dim_unset():
    """Same root cause through ``_embed_tokens`` (st.py:321-336): the loop records the first
    matrix's dimension, then raises on the second.  The index keeps dim=8 although nothing
    was added, and a later batch of dimension 5 is refused."""
    rng = np.random.default_rng(0)
    idx = RemexIndex(bits=2, similarity="maxsim")
    with pytest.raises(ValueError):
        idx.add([rng.standard_normal((3, 8)).astype(np.float32),
                 rng.standard_normal((2, 5)).astype(np.float32)])
    assert idx.dim is None, f"rejected batch left dim={idx.dim!r}"
    idx.add([rng.standard_normal((3, 5)).astype(np.float32)])
    assert len(idx) == 1


def test_control_modelled_index_rejects_empty_add_without_state_change():
    """control: with a model (dim known) the same call is a ValueError and changes nothing."""
    sys.path.insert(0, str(__import__("pathlib").Path(__import__("remex").__file__).resolve().parents[1] / "tests"))
    from _fake_st import FakeModel
    idx = RemexIndex(FakeModel(dim=16), bits=4)
    with pytest.raises(ValueError):
        idx.add([])
    assert idx.dim == 16 and idx.quantizer is None and len(idx) == 0


# ===========================================================================
# F2  Quantizer.search skips the bits / mean guards on a warm cache (remex/core.py)
# ===========================================================================
#
# search() runs _check_rotation, then reads the cached float32 matrix, which bypasses
# _resolve_centroids (where _check_bits and _check_mean live).  The only other place they
# run is _effective_norms, and that returns early when renorm=False or the codes are scalar.
# search_batch (core.py:1868-1871), search_adc, search_twostage and decode all guard every time.

def _warm_centered_container():
    X = _gauss(40, 16) + 3.0
    owner = Quantizer(16, 2, mean=corpus_mean(X))
    c = owner.encode(X)
    q = _gauss(1, 16, seed=9)[0]
    correct = owner.search(c, q, k=3)                     # also warms the cache
    return owner, c, q, correct


def test_f2_warm_search_refuses_centred_codes_when_quantizer_has_no_mean_and_renorm_false():
    """Cold, the codes are refused ('mean mismatch'); warm, they are scored without their
    mean and the answer is silently wrong (about 14 against the correct 31)."""
    owner, c, q, correct = _warm_centered_container()
    reader = Quantizer(16, 2, renorm=False)
    with pytest.raises(ValueError, match="mean mismatch"):
        reader.search(c, q, k=3)
    # (on failure the call above returned: its scores are not the correct ones)


def test_f2_warm_search_refuses_codes_of_another_bit_width_when_renorm_false():
    owner = Quantizer(16, 4)
    c = owner.encode(_gauss(40, 16))
    q = _gauss(1, 16, seed=9)[0]
    owner.search(c, q, k=3)
    with pytest.raises(ValueError, match="bits mismatch"):
        Quantizer(16, 2, renorm=False).search(c, q, k=3)


def test_f2_warm_search_refuses_codes_of_another_bit_width_in_scalar_mode():
    """Scalar codes carry no norms, so ``_effective_norms`` never reaches the guards at all."""
    owner = Quantizer(8, 3, normalize=False, rotation="none")
    c = owner.encode(_gauss(20, 8))
    q = _gauss(1, 8, seed=9)[0]
    owner.search(c, q, k=3)
    with pytest.raises(ValueError, match="bits mismatch"):
        Quantizer(8, 2, normalize=False, rotation="none").search(c, q, k=3)


@pytest.mark.parametrize("method", ["search_batch", "search_adc", "search_twostage"])
def test_control_other_read_paths_refuse_a_warm_foreign_container(method):
    """control: every sibling read path refuses the same warm container."""
    owner, c, q, _ = _warm_centered_container()
    reader = Quantizer(16, 2, renorm=False)
    call = {"search_batch": lambda: reader.search_batch(c, q[None], k=3),
            "search_adc": lambda: reader.search_adc(c, q, k=3),
            "search_twostage": lambda: reader.search_twostage(c, q, k=3)}[method]
    with pytest.raises(ValueError, match="mean mismatch"):
        call()


def test_control_cold_search_refuses_the_same_container():
    owner, c, q, _ = _warm_centered_container()
    c.drop_cache()
    with pytest.raises(ValueError, match="mean mismatch"):
        Quantizer(16, 2, renorm=False).search(c, q, k=3)


# ===========================================================================
# F3  RemexIndex.load reads the file with np.load, not remex._io.open_npz (remex/st.py:534)
# ===========================================================================

def _small_index():
    idx = RemexIndex(bits=4)
    idx.add(_gauss(10, 8))
    return idx


def test_f3_load_finds_the_file_save_wrote(tmp_path):
    """CHANGELOG 2.1.0: 'save("index") writes index.npz ... load("index") now reads it'
    (tests/test_integrity.py B-04 covers CompressedVectors/PackedVectors).  RemexIndex.load
    calls np.load directly and raises FileNotFoundError for the suffix-less name."""
    written = _small_index().save(str(tmp_path / "index"))
    assert written.endswith("index.npz")
    assert len(RemexIndex.load(str(tmp_path / "index"))) == 10


@pytest.mark.parametrize("kind", ["garbage", "empty", "truncated"])
def test_f3_load_names_a_damaged_file(tmp_path, kind):
    """CHANGELOG 2.1.0: damaged files raise 'is not a readable .npz archive' rather than
    EOFError / BadZipFile / NumPy's pickle advice.  RemexIndex.load passes all three through."""
    good = _small_index().save(str(tmp_path / "good"))
    data = {"garbage": b"this is not an archive", "empty": b"",
            "truncated": open(good, "rb").read()[:200]}[kind]
    bad = tmp_path / f"{kind}.npz"
    bad.write_bytes(data)
    with pytest.raises(ValueError, match="not a readable .npz archive"):
        RemexIndex.load(str(bad))


def test_control_container_load_handles_both(tmp_path):
    """control: CompressedVectors.load, which uses open_npz, has the behaviour F3 expects."""
    path = _small_index().save(str(tmp_path / "index"))
    assert CompressedVectors.load(str(tmp_path / "index")).n == 10
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="not a readable .npz archive"):
        CompressedVectors.load(str(bad))


# ===========================================================================
# F4  RHTOperator caches raw pointers and a ctypes handle: copy and pickle crash (remex/rotation.py)
# ===========================================================================
#
# RHTOperator.__init__ (rotation.py:403-405) stores `array.ctypes.data` integers for its
# permutation and sign tables, and `_apply2` (448-450) caches the loaded CDLL in `_kernel`.
# Quantizer(rotation="rht") is the default for even d, so every default Quantizer holds one.
# pickle / deepcopy copy those integers verbatim: they point into the *original* object's
# arrays (or into another process), and the native kernel dereferences them.

_CHILD_PICKLE = textwrap.dedent("""
    import pickle, sys, numpy as np
    q = pickle.loads(open(sys.argv[1], "rb").read())
    X = np.random.default_rng(0).standard_normal((50, 64)).astype(np.float32)
    np.save(sys.argv[2], q.encode(X).indices)
""")


def test_f4_pickled_rht_quantizer_encodes_in_a_fresh_process(tmp_path):
    """Pickling a never-used default Quantizer succeeds; unpickling it in a new process and
    encoding crashes the interpreter (SIGSEGV, return code -11) because the stored pointers
    belong to the parent's address space.  Acceptable outcomes: pickling is refused with a
    TypeError/PicklingError, or the child reproduces the parent's codes."""
    import pickle
    q = Quantizer(64, 4)                                     # rotation='rht' (even d)
    assert q.rotation == "rht"
    X = np.random.default_rng(0).standard_normal((50, 64)).astype(np.float32)
    want = Quantizer(64, 4).encode(X).indices
    try:
        blob = pickle.dumps(q)
    except (TypeError, pickle.PicklingError):
        return                                               # refusing is a legitimate fix
    (tmp_path / "q.pkl").write_bytes(blob)
    proc = subprocess.run([sys.executable, "-c", _CHILD_PICKLE, str(tmp_path / "q.pkl"), str(tmp_path / "out.npy")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"child died with return code {proc.returncode}: {proc.stderr[-300:]}"
    assert np.array_equal(np.load(tmp_path / "out.npy"), want)


_CHILD_DEEPCOPY = textwrap.dedent("""
    import copy, gc, numpy as np
    from remex import Quantizer
    X = np.random.default_rng(0).standard_normal((50, 64)).astype(np.float32)
    want = Quantizer(64, 4).encode(X).indices
    q = Quantizer(64, 4)
    try:
        q2 = copy.deepcopy(q)
    except (TypeError, ValueError):
        raise SystemExit(0)                                   # refusing the copy is acceptable
    del q
    gc.collect()
    junk = []
    for _ in range(20000):                                    # reuse the freed tables
        junk.append(np.full(128, 7, dtype=np.int32))
        junk.append(np.full(128, 3.0, dtype=np.float32))
        junk.append(np.full(128, 3.0, dtype=np.float64))
    got = q2.encode(X).indices
    raise SystemExit(0 if np.array_equal(got, want) else 3)
""")


def test_f4_deepcopied_rht_quantizer_survives_the_original_being_freed():
    """deepcopy of an unused Quantizer 'works', but its operator still points into the
    original's tables; once the original is freed and its memory reused the copy reads
    garbage or segfaults (return code -11 / 139 here)."""
    proc = subprocess.run([sys.executable, "-c", _CHILD_DEEPCOPY], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"child ended with return code {proc.returncode} (3 = wrong codes, negative = signal)"


def test_control_haar_quantizer_pickles_and_copies():
    """control: the Haar path holds no raw pointers; pickle and deepcopy round-trip it."""
    import copy, pickle
    q = Quantizer(16, 4, rotation="haar")
    X = _gauss(20, 16)
    want = q.encode(X).indices
    assert np.array_equal(pickle.loads(pickle.dumps(q)).encode(X).indices, want)
    assert np.array_equal(copy.deepcopy(q).encode(X).indices, want)


# ===========================================================================
# F5  IVFCoarseIndex.__init__ runs none of the quantizer's guards (remex/ivf.py:94-137)
# ===========================================================================

def test_f5_ivf_constructor_names_a_bit_width_mismatch():
    """A 2-bit container under a 1-bit quantizer: the MSB shift is computed from the
    quantizer's width (ivf.py:143-145), cell ids exceed n_cells and construction dies in
    np.cumsum with 'provided out is the wrong size'.  Expected: the 'bits mismatch' message
    every other read path gives."""
    c = Quantizer(8, 2).encode(_gauss(10, 8))
    with pytest.raises(ValueError, match="bits mismatch"):
        IVFCoarseIndex(Quantizer(8, 1), c, n_bits=1, mode="rotated_prefix")


def test_f5_ivf_constructor_refuses_codes_it_would_hash_wrongly():
    """A 2-bit container under a 4-bit quantizer: shift=3 turns every code into 0, so the
    constructor silently builds an index with all rows in one cell (it only fails later, at
    the first search).  Expected: refusal at construction."""
    c = Quantizer(8, 2).encode(_gauss(10, 8))
    with pytest.raises(ValueError, match="bits mismatch"):
        IVFCoarseIndex(Quantizer(8, 4), c, n_bits=2, mode="lsh")


def test_control_ivf_never_answers_for_codes_the_quantizer_cannot_read():
    """control: whether or not the constructor refuses (F5), no wrong *answer* comes out:
    the guard at query time exists and runs before any candidate cell is read."""
    c = Quantizer(8, 2).encode(_gauss(10, 8))
    try:
        ivf = IVFCoarseIndex(Quantizer(8, 4), c, n_bits=2, mode="lsh")
    except ValueError as exc:
        assert "bits mismatch" in str(exc)
        return
    with pytest.raises(ValueError, match="bits mismatch"):
        ivf.search_coarse(_gauss(1, 8, seed=2)[0], k=3, nprobe=4)


# ===========================================================================
# AMBIGUOUS: behaviour whose intended contract is not written down
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="AMBIGUOUS F6: the R setter on a rotation='none' quantizer is ignored by "
                                       "the CPU paths (identity short-circuit) but read by GPUSearcher")
def test_f6_gpu_and_cpu_agree_after_assigning_R_to_an_identity_quantizer():
    rng = np.random.default_rng(0)
    X = _gauss(30, 8)
    q = Quantizer(8, 4, rotation="none")
    M, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    q.R = M.astype(np.float32)
    c = q.encode(X)
    cpu = q.search(c, X[3], k=3)[0]
    gpu = GPUSearcher(q, c, backend="numpy").search(X[3], k=3)[0]
    assert list(cpu) == list(gpu)


@pytest.mark.xfail(strict=True, reason="AMBIGUOUS F7: resident_bytes is documented as 'actual RAM including any "
                                       "active caches' but ignores the per-precision _dir_len arrays")
def test_f7_resident_bytes_counts_the_direction_length_cache():
    q = Quantizer(16, 4)
    c = q.encode(_gauss(2000, 16))
    before = c.resident_bytes
    q.search_adc(c, _gauss(1, 16, seed=1)[0], k=5)          # fills _dir_len[None] only
    assert c._x_hat_rot is None and c._dir_len
    assert c.resident_bytes > before


@pytest.mark.xfail(strict=True, reason="AMBIGUOUS F7: PackedVectors.resident_bytes says 'no caches' and has no "
                                       "drop_cache, yet ADC search fills _dir_len")
def test_f7_packed_vectors_cache_is_visible_and_droppable():
    q = Quantizer(16, 4)
    pv = PackedVectors.from_compressed(q.encode(_gauss(2000, 16)))
    before = pv.resident_bytes
    q.search_adc(pv, _gauss(1, 16, seed=1)[0], k=5)
    assert pv.resident_bytes > before or hasattr(pv, "drop_cache")


@pytest.mark.xfail(strict=True, reason="AMBIGUOUS F8: RemexIndex is not documented as thread-safe, but a search that "
                                       "straddles an add publishes a stale decoded-token cache (st.py:464-466)")
def test_f8_search_straddling_an_add_does_not_leave_a_stale_decode_cache():
    """Deterministic simulation of the interleaving: another thread's ``add`` lands between
    the moment a MaxSim search decodes the old corpus and the moment it stores the result in
    ``self._decoded``.  ``add`` has already reset the cache, so the stale decode is published
    after the reset and every later search fails with an IndexError (or mis-scores) until
    the next add."""
    sys.path.insert(0, str(__import__("pathlib").Path(__import__("remex").__file__).resolve().parents[1] / "tests"))
    from _fake_st import FakeMultiVectorModel
    model = FakeMultiVectorModel(dim=16)
    docs1 = [f"w{i} w{i + 1} w{i + 2}" for i in range(6)]
    docs2 = [f"w{i} w{i + 7}" for i in range(30, 34)]
    idx = RemexIndex(model, bits=4)
    idx.add(docs1)
    original = idx.quantizer._score_vectors
    fired = []

    def decode_then_add(compressed, precision=None):
        out = original(compressed, precision)
        if not fired:
            fired.append(1)
            idx.add(docs2)
        return out

    idx.quantizer._score_vectors = decode_then_add
    idx.search("w30 w37", top_k=3)
    idx.quantizer._score_vectors = original
    fresh = RemexIndex(model, bits=4)
    fresh.add(docs1)
    fresh.add(docs2)
    assert idx.search("w30 w37", top_k=3) == fresh.search("w30 w37", top_k=3)
