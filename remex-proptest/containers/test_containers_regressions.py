"""Deterministic regression tests for remex area B (containers, serialization,
Matryoshka, packing).  Plain pytest, no Hypothesis, no shared helpers.

Every test in ``TestConfirmed`` encodes a documented contract and FAILS on
remex 2.2.0 (/home/user/remex @ b8d527a) for the reason in its docstring.
``TestAmbiguous`` holds findings where the contract is unclear; they are
marked xfail(strict=True) so they document current behaviour and flip loudly
when remex changes.  Nothing here modifies remex.

Run:  pytest test_containers_regressions.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from remex import (
    CompressedVectors, PackedVectors, Quantizer, load_pq, save_pq,
)

pytest.importorskip("pyarrow")


def rows(n, d, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# Confirmed findings
# ---------------------------------------------------------------------------

class TestConfirmed:

    # F1 --------------------------------------------------------------------
    @pytest.mark.parametrize("d,bits,n", [
        (7, 3, 5),    # silently wrong codes: the byte counts coincide (15 == 15)
        (3, 2, 5),    # obscure numpy reshape error: 4 bytes vs 5 rows x 1 byte
        (5, 1, 9),    # silently wrong or reshape error depending on n
        (6, 3, 4),    # 4-bit-aligned only: still row padded for 3-bit codes
    ])
    def test_F1_compressed_npz_read_by_packedvectors_load_gives_the_same_codes(
            self, tmp_path, d, bits, n):
        """CompressedVectors.save packs the whole (n*d) code stream flat
        (core.py:329); PackedVectors.save writes row-padded rows
        (core.py:694).  Both use the same .npz keys and neither records its
        layout, so PackedVectors.load (core.py:716-718) reinterprets flat bytes
        as rows whenever d*bits % 8 != 0 and returns other codes or raises a
        reshape error.  Contract: PackedVectors docstring (core.py:410-412)
        offers load() for disk data; README.md:319 documents one .npz format."""
        c = Quantizer(d, bits, seed=1, rotation="haar").encode(rows(n, d))
        path = c.save(tmp_path / "c.npz")
        got = PackedVectors.load(path).unpack_rows(0, n)
        assert np.array_equal(got, c.indices)

    @pytest.mark.parametrize("d,bits,n", [(7, 3, 5), (3, 2, 5), (5, 1, 9)])
    def test_F1_packed_npz_read_by_compressedvectors_load_gives_the_same_codes(
            self, tmp_path, d, bits, n):
        """Same defect, other direction: CompressedVectors.load (core.py:364)
        unpacks row-padded bytes as one flat stream and silently returns
        different indices."""
        c = Quantizer(d, bits, seed=1, rotation="haar").encode(rows(n, d))
        path = PackedVectors.from_compressed(c).save(tmp_path / "p.npz")
        got = CompressedVectors.load(path).indices
        assert np.array_equal(got, c.indices)

    # F2 --------------------------------------------------------------------
    def test_F2_search_on_a_warm_scalar_container_refuses_another_bit_width(self):
        """Quantizer.search runs only _check_rotation before reusing a warm
        compressed._x_hat_rot (core.py:1665-1669).  The bits guard lives in
        _resolve_centroids (core.py:2028) and _effective_norms (core.py:1411),
        and _effective_norms returns early when norms is None (core.py:1406).
        A cold container raises 'bits mismatch'; a warm scalar-mode one does
        not.  Contract: _check_bits docstring (core.py:1261-1266); CHANGELOG
        2.0.0 'The mean and rotation guards ran in _resolve_centroids, which a
        warm container skips' (same class, not fully fixed)."""
        X = rows(6, 8, seed=1)
        q4 = Quantizer(8, 4, rotation="none", normalize=False)
        q3 = Quantizer(8, 3, rotation="none", normalize=False)
        cold = q4.encode(X)
        with pytest.raises(ValueError, match="bits mismatch"):
            q3.search(cold, X[0], k=2)              # control: cold raises
        warm = q4.encode(X)
        q4.search(warm, X[0], k=2)                  # warm _x_hat_rot
        with pytest.raises(ValueError, match="bits mismatch"):
            q3.search(warm, X[0], k=2)

    def test_F2_search_on_a_warm_centred_container_refuses_an_uncentred_reader(self):
        """Same mechanism with the mean guard and a renorm=False reader, which
        also returns *wrong scores* (uncentred estimator on centred codes)
        rather than only skipping a refusal.  Contract: _check_mean docstring
        (core.py:1277-1283) and CLAUDE.md design decision 4 ('a mismatch raises
        rather than returning vectors shifted')."""
        X = rows(30, 16, seed=2) + 3.0
        mu = X.mean(axis=0).astype(np.float32)
        qc = Quantizer(16, 4, rotation="rht", seed=1, mean=mu)
        qr = Quantizer(16, 4, rotation="rht", seed=1, renorm=False)
        cold = qc.encode(X)
        with pytest.raises(ValueError, match="mean mismatch"):
            qr.search(cold, X[0], k=3)              # control: cold raises
        warm = qc.encode(X)
        qc.search(warm, X[0], k=3)
        with pytest.raises(ValueError, match="mean mismatch"):
            qr.search(warm, X[0], k=3)

    def test_F2_a_subset_inherits_the_warm_cache_and_so_the_hole(self):
        """subset() copies _x_hat_rot (core.py:268-269), so the unguarded warm
        path is reachable from a container that was never searched itself."""
        X = rows(10, 8, seed=3)
        q4 = Quantizer(8, 4, rotation="none", normalize=False)
        q3 = Quantizer(8, 3, rotation="none", normalize=False)
        warm = q4.encode(X)
        q4.search(warm, X[0], k=2)
        sub = warm.subset([0, 2, 4])
        with pytest.raises(ValueError, match="bits mismatch"):
            q3.search(sub, X[0], k=2)

    # F3 --------------------------------------------------------------------
    @pytest.mark.parametrize("seed", [2**63, 2**64 - 1])
    @pytest.mark.parametrize("packed", [False, True])
    def test_F3_npz_records_a_seed_that_quantizer_and_arrow_accept(
            self, tmp_path, seed, packed):
        """_tag_npz stores the seed as np.int64 (core.py:54), so any seed
        >= 2**63 raises OverflowError at save, although Quantizer encodes with
        it (numpy's default_rng takes it) and Arrow records it as text
        (core.py:768).  Contract: CLAUDE.md 'every persisted container records
        the seed'; CHANGELOG 2.1.0 'Containers now record seed (.npz key ...)'."""
        c = Quantizer(8, 3, seed=seed, rotation="rht").encode(rows(3, 8))
        c.save_arrow(str(tmp_path / "a.arrow"))                       # control
        assert PackedVectors.load_arrow(str(tmp_path / "a.arrow")).seed == seed
        obj = PackedVectors.from_compressed(c) if packed else c
        back = type(obj).load(obj.save(tmp_path / "x.npz"))
        assert back.seed == seed

    # F4 --------------------------------------------------------------------
    @pytest.mark.parametrize("key,value", [
        ("rotation", "haar"),
        ("bits", 2),
        ("d", 4),
        ("n", 2),
    ])
    def test_F4_arrow_extra_metadata_cannot_relabel_the_encoding(
            self, tmp_path, key, value):
        """save_arrow copies **extra_metadata over the reserved keys
        (core.py:771-773) after writing them (core.py:761-766), so
        save_arrow(p, rotation='haar') on an 'rht' container labels the file
        'haar' and load_arrow then decodes against the wrong rotation without
        complaint.  The same method refuses a contradicting seed=
        (core.py:746-750), so contradiction is meant to be refused.  Either a
        ValueError or ignoring the key satisfies this test."""
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(rows(4, 8))
        p = str(tmp_path / "x.arrow")
        try:
            c.save_arrow(p, **{key: value})
        except ValueError:
            return
        back = PackedVectors.load_arrow(p)
        assert (back.rotation, back.bits, back.d, back.n) == (
            c.rotation, c.bits, c.d, c.n)

    def test_F4_extra_metadata_mean_turns_an_uncentred_file_into_a_centred_one(
            self, tmp_path):
        """Same hole through the 'mean' key: with no mean the writer does not
        overwrite it, so a caller keyword becomes the container's mean bytes."""
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(rows(4, 8))
        p = str(tmp_path / "x.arrow")
        try:
            c.save_arrow(p, mean="x" * 32)      # str() of it becomes 32 raw bytes = 8 float32
        except ValueError:
            return
        assert PackedVectors.load_arrow(p).mean is None

    # F5 --------------------------------------------------------------------
    def test_F5_packedvectors_load_of_a_legacy_indices_only_npz(self, tmp_path):
        """open_npz (_io.py:98) and CompressedVectors.load (core.py:363-368)
        accept the pre-bit-packing layout (an 'indices' array and no
        'packed_indices'); PackedVectors.load indexes data['packed_indices']
        unconditionally (core.py:717) and leaks KeyError, the raw-error class
        open_npz exists to prevent (CHANGELOG 2.1.0 'Damaged and foreign files
        say so').  Loading it or a named ValueError would both be fine."""
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(rows(3, 8))
        p = tmp_path / "legacy.npz"
        np.savez(p, indices=c.indices, norms=c.norms, d=np.int32(8),
                 bits=np.int32(4), n=np.int32(3))
        assert CompressedVectors.load(p).n == 3        # control: CV handles it
        try:
            got = PackedVectors.load(p)
        except ValueError:
            return
        assert np.array_equal(got.unpack_rows(0, got.n), c.indices)


class TestConfirmedDamagedFiles:

    @pytest.mark.parametrize("packed", [False, True])
    def test_F6_a_damaged_npz_member_raises_the_named_value_error(self, tmp_path, packed):
        """open_npz (_io.py:87-93) turns BadZipFile/EOFError/ValueError into
        'is not a readable .npz archive' but only around np.load; the members
        are read lazily afterwards (core.py:357-373, 710-722), so a corrupted
        member leaks BadZipFile / zlib.error / NotImplementedError / OSError.
        Contract: _io.py:3-14 ('np.load on a damaged file raises EOFError,
        BadZipFile ... open_npz turns all of those into one message naming the
        file') and CHANGELOG 2.1.0 'Damaged and foreign files say so'.
        Brute force: flip every byte of a small archive once; every outcome
        must be a clean load or ValueError.  Truncation is already handled."""
        c = Quantizer(16, 4, seed=1, rotation="rht").encode(rows(20, 16))
        obj = PackedVectors.from_compressed(c) if packed else c
        p = tmp_path / "a.npz"
        obj.save(p)
        raw = p.read_bytes()
        bad = {}
        for pos in range(len(raw)):
            b = bytearray(raw)
            b[pos] ^= 0x01
            p.write_bytes(bytes(b))
            try:
                type(obj).load(p)
            except ValueError:
                pass
            except Exception as e:
                bad.setdefault(type(e).__name__, pos)
        assert not bad, f"exception types that escaped open_npz (first byte offset): {bad}"

    def test_F6_a_truncated_npz_is_already_named(self, tmp_path):
        """Control for F6: the documented truncated case does raise ValueError."""
        c = Quantizer(16, 4, seed=1, rotation="rht").encode(rows(20, 16))
        p = tmp_path / "a.npz"
        c.save(p)
        raw = p.read_bytes()
        for cut in range(0, len(raw), 17):
            p.write_bytes(raw[:cut])
            with pytest.raises(ValueError):
                CompressedVectors.load(p)


# ---------------------------------------------------------------------------
# Ambiguous: contract unclear, behaviour recorded.  xfail(strict) = "currently
# does X"; a change in remex makes the test XPASS and fail, prompting review.
# ---------------------------------------------------------------------------

def xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


class TestAmbiguous:

    @xfail("A1: compression_ratio is n*d*4 / nbytes with no guard for n == 0 (core.py:285-287, 672-674)")
    @pytest.mark.parametrize("packed", [False, True])
    def test_A1_compression_ratio_of_an_empty_container(self, packed):
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(np.empty((0, 8)))
        obj = PackedVectors.from_compressed(c) if packed else c
        assert obj.nbytes == 0
        ratio = obj.compression_ratio          # ZeroDivisionError today
        assert ratio == ratio                  # any non-raising value (nan, inf, 0) is fine

    @xfail("A2: subset(int) is not rejected: CompressedVectors builds a malformed container whose n is d (core.py:259-270)")
    def test_A2_subset_with_a_scalar_index(self):
        c = Quantizer(7, 3, seed=1, rotation="haar").encode(rows(6, 7))
        try:
            s = c.subset(2)
        except (IndexError, TypeError, ValueError):
            return
        assert s.indices.shape == (1, 7) and s.n == 1 and s.norms.shape == (1,)

    @xfail("A3: unpack_rows on a byte-unaligned row layout wraps negative starts and can return more rows than the container has; the aligned layout raises (core.py:476-484)")
    def test_A3_unpack_rows_never_returns_more_rows_than_exist(self):
        c = Quantizer(3, 2, seed=1, rotation="haar").encode(rows(6, 3))   # 6 bits per row: unaligned
        pv = PackedVectors.from_compressed(c)
        try:
            got = pv.unpack_rows(-1, 6)
        except (IndexError, ValueError):
            return
        assert got.shape[0] <= pv.n

    @xfail("A4: unpack_at([]) raises IndexError (float64 empty index) although subset([]) was made to accept an empty list (CHANGELOG 2.0.0) (core.py:495-499)")
    def test_A4_unpack_at_accepts_an_empty_list(self):
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(rows(4, 8))
        pv = PackedVectors.from_compressed(c)
        assert pv.unpack_at([]).shape == (0, 8)

    @xfail("A5: from_rows accepts a norms vector of the wrong length and rows of the wrong byte width without complaint (core.py:571-584)")
    def test_A5_from_rows_validates_lengths(self):
        rows_ = [bytes(4)] * 3
        with pytest.raises(ValueError):
            PackedVectors.from_rows(rows_, np.ones(2, np.float32), d=8, bits=4)

    @xfail("A6: from_rows([]) cannot build an empty container although every other path can (np.stack on an empty list, core.py:577)")
    def test_A6_from_rows_with_no_rows(self):
        pv = PackedVectors.from_rows([], np.empty(0, np.float32), d=8, bits=4)
        assert pv.n == 0

    @xfail("A7: at_precision documents 1..self.bits (core.py:592-599) but 5,6,7 raise from pack(); Quantizer(bits=8).search(precision=5) works")
    @pytest.mark.parametrize("target", [5, 6, 7])
    def test_A7_at_precision_to_unpackable_widths(self, target):
        c = Quantizer(8, 8, seed=1, rotation="rht").encode(rows(4, 8))
        pv = PackedVectors.from_compressed(c)
        assert pv.at_precision(target).bits == target

    @xfail("A8: CompressedVectors.subset hands the parent's mean array to the child (core.py:264) while every PackedVectors conversion copies it; encode() also shares it with the Quantizer, so an in-place edit defeats _check_mean")
    def test_A8_in_place_edit_of_a_derived_mean_does_not_reach_the_parent(self):
        mu = np.full(8, 2.0, np.float32)
        q = Quantizer(8, 4, seed=1, rotation="rht", mean=mu)
        c = q.encode(rows(5, 8) + 2.0)
        s = c.subset([0, 1])
        s.mean += 1.0
        assert np.array_equal(c.mean, np.full(8, 2.0, np.float32))
        assert np.array_equal(q.mean, np.full(8, 2.0, np.float32))

    @xfail("A9: load_pq ignores bytes after the norms section and accepts d=0 (pq_format.py:178-179)")
    def test_A9_pq_trailing_bytes_are_rejected(self, tmp_path):
        c = Quantizer(8, 4, seed=1, rotation="rht").encode(rows(3, 8))
        p = tmp_path / "x.pq"
        save_pq(p, c)
        p.write_bytes(p.read_bytes() + b"garbage")
        with pytest.raises(ValueError):
            load_pq(p)

    @xfail("A10: .npz stores n (and d, bits) as np.int32 (core.py:332-334, 695-697) so a container of >= 2**31 rows cannot be saved, while .pq (u64) and Arrow (text) can")
    def test_A10_npz_row_count_narrowing(self, tmp_path):
        # Demonstrated on a deliberately inconsistent container (one stored row, n claimed
        # as 2**31) so that no multi-GB allocation is needed: the OverflowError comes from
        # np.int32(self.n) before the data is looked at.  A real 2**31-row container hits
        # the same line after building the whole payload.
        pv = PackedVectors(np.zeros((1, 1), np.uint8), None, 2**31, 8, 1)
        pv.save(tmp_path / "x.npz")

    @xfail("A11: an at_precision(k) container is indistinguishable from a native k-bit one, so the only Quantizer that accepts it (bits=k) decodes it with the STANDALONE k-bit table, not the nested level table that Quantizer(bits=b).decode(c, precision=k) uses; Quantizer(bits=b) refuses it (_check_bits), so no public call reads it with the nested table")
    def test_A11_reduced_container_reads_like_the_nested_precision_read(self):
        d, b, k = 64, 8, 4
        X = rows(300, d)
        q = Quantizer(d, b, seed=1, rotation="rht")
        c = q.encode(X)
        nested = q.decode(c, precision=k)                    # documented Matryoshka read (CLAUDE.md decision 2)
        reduced = PackedVectors.from_compressed(c).at_precision(k).to_compressed()
        with pytest.raises(ValueError, match="bits mismatch"):
            q.decode(reduced)                                # the nested reader refuses it
        standalone = Quantizer(d, k, seed=1, rotation="rht").decode(reduced)
        assert np.allclose(standalone, nested, atol=1e-4)    # measured max |diff| 0.34, MSE +46% vs nested
