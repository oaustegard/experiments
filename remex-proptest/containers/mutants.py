"""Mutation calibration for test_containers_props.py.

Plants one plausible defect at a time by monkeypatching remex in-process (the
files under /home/user/remex are never touched), runs the property harness with
``-x`` and records whether it goes red.

    python mutants.py list
    python mutants.py run M3                 # one mutant, prints pytest output
    python mutants.py all [M1 M2 ...]        # each in its own subprocess, results
                                             # appended to mutation_results.txt

A mutant that survives is investigated by hand: either it is equivalent (no
observable difference) or the harness has a RIPR gap (reach / infect /
propagate / reveal) that is then closed and the run repeated.
"""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "mutation_results.txt"
SEED = "7"


# ---------------------------------------------------------------------------
# patch machinery
# ---------------------------------------------------------------------------

def _raw(owner, attr):
    orig = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
    fn = orig.__func__ if isinstance(orig, (classmethod, staticmethod)) else orig
    return orig, fn


def patch_src(owner, attr, old, new, count=1):
    """Replace ``old`` by ``new`` in the source of owner.attr and re-exec it."""
    orig, fn = _raw(owner, attr)
    src = textwrap.dedent(inspect.getsource(fn))
    lines = src.splitlines()
    while lines[0].lstrip().startswith("@"):
        lines.pop(0)
    src = "\n".join(lines)
    assert old in src, f"pattern not found in {attr}: {old!r}"
    src = src.replace(old, new, count)
    mod = sys.modules[fn.__module__]
    ns = {}
    exec(compile(src, f"<mutant {attr}>", "exec"), mod.__dict__, ns)
    newfn = ns[fn.__name__]
    if isinstance(orig, classmethod):
        newfn = classmethod(newfn)
    elif isinstance(orig, staticmethod):
        newfn = staticmethod(newfn)
    if isinstance(owner, type):
        setattr(owner, attr, newfn)
    else:
        setattr(owner, attr, newfn)
        return newfn
    return None


def rebind(name, newfn):
    """Replace a module-level function everywhere remex imported it by name."""
    import remex, remex.core, remex.packing, remex.pq_format
    for m in (remex, remex.core, remex.packing, remex.pq_format):
        if hasattr(m, name):
            setattr(m, name, newfn)


def patch_module_fn(mod, name, old, new, rebinds=()):
    fn = patch_src(mod, name, old, new)
    for r in rebinds:
        rebind(r, fn)


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------

def m1():
    from remex.core import CompressedVectors
    patch_src(CompressedVectors, "subset", "self.seed, self.scale,", "None, self.scale,")

def m2():
    from remex.core import CompressedVectors
    patch_src(CompressedVectors, "subset", "sub._x_hat_rot = self._x_hat_rot[idx]",
              "sub._x_hat_rot = self._x_hat_rot")

def m3():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "unpack_rows", "for i in range(n_rows):", "for i in range(n_rows - 1):")

def m4():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "at_precision", "shift = self.bits - target_bits",
              "shift = self.bits - target_bits - 1")

def m5():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "from_compressed", 'getattr(compressed, "scale", None))', "None)")

def m6():
    import remex.packing as P
    patch_module_fn(P, "pack", "| (groups[:, 5] >> 2)", "| (groups[:, 5] >> 1)",
                    rebinds=("pack",))

def m7():
    import remex.core as C
    patch_module_fn(C, "_tag_npz", "if scale is not None:", "if False:")

def m8():
    import remex.core as C
    patch_module_fn(C, "_as_rows", "if idx.shape != (n,):", "if False:")

def m9():
    import remex.pq_format as Q
    patch_module_fn(Q, "load_pq", "has_norms = not (flags & PQ_FLAG_NO_NORMS)", "has_norms = True",
                    rebinds=("load_pq",))

def m10():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "subset", "idx = _as_rows(idx, self.n)",
              "idx = np.unique(_as_rows(idx, self.n))")

def m11():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "save_arrow", 'metadata[b"mean"] = self.mean.astype("<f4").tobytes()', "pass")

def m12():
    from remex.core import Quantizer
    patch_src(Quantizer, "_check_seed", "and int(stored) != int(self.seed)):", "and False):")

def m13():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "at_precision", "self._copy_mean_dots(), self.seed, self.scale,",
              "self._copy_mean_dots(), None, self.scale,")

def m14():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "from_rows", "seed, scale)", "None, scale)")

def m15():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "unpack_at", "idx = np.asarray(idx)", "idx = np.sort(np.asarray(idx))")

def m16():
    import remex.packing as P
    patch_module_fn(P, "packed_nbytes", "return ((total_values + 7) // 8) * 3",
                    "return (total_values * 3 + 7) // 8", rebinds=("packed_nbytes",))

def m17():
    from remex.core import Quantizer
    patch_src(Quantizer, "search", "self._check_rotation(compressed)", "pass")

def m18():
    from remex.core import CompressedVectors
    patch_src(CompressedVectors, "_npz_fields",
              'fields["mean_dots"] = self.mean_dots', 'fields["mean_dots"] = self.mean_dots[::-1].copy()')

def m19():
    import remex.core as C
    patch_module_fn(C, "_tag_from_npz",
                    'scale = float(data["scale"]) if "scale" in data.files else None', "scale = None")

def m20():
    from remex.core import CompressedVectors
    patch_src(CompressedVectors, "subset", "None if self.norms is None else self.norms[idx],",
              "None if self.norms is None else self.norms[np.sort(idx)],")

def m21():
    from remex.core import PackedVectors
    patch_src(PackedVectors, "to_compressed", "self._copy_mean_dots(), self.seed, self.scale,",
              "self._copy_mean_dots(), self.seed, None,")

def m22():
    # off-by-one in the chunk loop bound of at_precision: the last chunk is dropped
    from remex.core import PackedVectors
    patch_src(PackedVectors, "at_precision", "end = min(start + chunk, self.n)",
              "end = min(start + chunk, self.n - (1 if self.n > 4096 else 0))")

def m23():
    # ADC on PackedVectors forgets the shift for precision reads
    from remex.core import Quantizer
    patch_src(Quantizer, "_adc_score_packed", "if shift > 0:", "if False:")

def m24():
    import remex.pq_format as Q
    patch_module_fn(Q, "save_pq", 'header[17] = ROTATION_CODES[validate_rotation(compressed.rotation)]',
                    'header[17] = 0', rebinds=("save_pq",))


MUTANTS = {
    "M1": ("CompressedVectors.subset drops the seed", m1),
    "M2": ("CompressedVectors.subset carries _x_hat_rot unsliced (stale cache)", m2),
    "M3": ("PackedVectors.unpack_rows off-by-one in the unaligned row loop (last row never written)", m3),
    "M4": ("PackedVectors.at_precision shifts by bits-target-1", m4),
    "M5": ("PackedVectors.from_compressed drops scale", m5),
    "M6": ("pack(): 3-bit byte-1 layout uses >>1 instead of >>2", m6),
    "M7": ("_tag_npz never records scale", m7),
    "M8": ("_as_rows: wrong-length bool mask no longer refused", m8),
    "M9": ("load_pq ignores the no-norms flag", m9),
    "M10": ("PackedVectors.subset de-duplicates and sorts the selection", m10),
    "M11": ("PackedVectors.save_arrow forgets the mean (the 2.0.0 bug class)", m11),
    "M12": ("Quantizer._check_seed never fires", m12),
    "M13": ("PackedVectors.at_precision drops the seed", m13),
    "M14": ("PackedVectors.from_rows ignores its seed argument", m14),
    "M15": ("PackedVectors.unpack_at sorts the index array", m15),
    "M16": ("packed_nbytes 3-bit: no group padding", m16),
    "M17": ("Quantizer.search drops its _check_rotation call", m17),
    "M18": ("CompressedVectors._npz_fields writes mean_dots reversed", m18),
    "M19": ("_tag_from_npz never reads scale", m19),
    "M20": ("CompressedVectors.subset slices norms with sorted indices", m20),
    "M21": ("PackedVectors.to_compressed drops scale", m21),
    "M22": ("at_precision chunk loop drops the final row when n > 4096", m22),
    "M23": ("_adc_score_packed ignores the precision shift", m23),
    "M24": ("save_pq always writes rotation code 0", m24),
}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

# Most mutation-sensitive properties first so that -x ends a caught mutant quickly; the
# whole non-known suite still runs for a survivor.
ORDER = [
    "test_operation_sequences_keep_container_and_model_in_agreement",
    "test_subset_accepts_documented_forms_and_rejects_the_rest",
    "test_every_format_roundtrips_losslessly",
    "test_hand_built_containers_roundtrip_every_writer",
    "test_at_precision_is_right_shift_and_composes",
    "test_mismatched_quantizers_are_refused_cold_and_warm",
    "test_roundtripped_containers_keep_guards_armed",
    "test_pack_matches_bitstream_reference_and_roundtrips",
    "test_packed_nbytes_arithmetic",
    "test_chunk_boundaries_at_4096",
    "test_damaged_pq_is_a_valid_container_or_a_value_error",
    "test_decode_and_scores_match_an_independent_oracle_after_reload",
    "test_derived_containers_do_not_alias_their_parent",
    "test_writers_are_all_or_nothing",
    "test_nbytes_and_compression_ratio_arithmetic",
    "test_encode_produces_wellformed_containers",
    "test_encode_is_independent_of_row_block_size",
    "test_one_bit_matryoshka_level_is_exactly_the_standalone_one_bit_code",
    "test_save_params_file_contents",
    "test_damaged_npz_and_arrow_never_yield_a_wrong_shaped_container",
    "test_unsupported_bit_widths_are_refused_everywhere",
]
TEST_FILE = str(HERE / "test_containers_props.py")
PYTEST_ARGS = ["-x", "-q", "--no-header", "-p", "no:cacheprovider", "--tb=line",
               f"--hypothesis-seed={SEED}", "-W", "ignore"]


def run_one(mid: str) -> int:
    import pytest
    if mid != "M0":
        MUTANTS[mid][1]()
    scratch = tempfile.mkdtemp(prefix=f"mut-{mid}-")
    os.chdir(scratch)                          # keeps .hypothesis DB out of the clean tree
    sys.path.insert(0, str(HERE))
    return int(pytest.main([f"{TEST_FILE}::{t}" for t in ORDER] + PYTEST_ARGS))


def drive(ids):
    with RESULTS.open("a") as out:
        out.write(f"\n# run {time.strftime('%Y-%m-%d %H:%M:%S')}  seed={SEED}  args: {' '.join(PYTEST_ARGS)}\n")
        for mid in ids:
            desc = "baseline (no mutation)" if mid == "M0" else MUTANTS[mid][0]
            t = time.time()
            try:
                p = subprocess.run([sys.executable, __file__, "run", mid], capture_output=True,
                                   text=True, timeout=600)
                text = p.stdout + p.stderr
                rc = p.returncode
            except subprocess.TimeoutExpired as e:
                text, rc = "TIMEOUT", -1
            dt = time.time() - t
            failed = re.findall(r"FAILED [^\s]*::(\w+)", text)
            line = re.findall(r"(\d+ failed.*|\d+ passed.*)", text)
            verdict = ("CAUGHT" if rc == 1 else "SURVIVED" if rc == 0 else f"ERROR(rc={rc})")
            row = (f"{mid:4s} {verdict:9s} {dt:6.0f}s  {desc}\n"
                   f"       by: {failed[0] if failed else '-'}   [{line[-1] if line else text[-120:]}]\n")
            out.write(row)
            out.flush()
            print(row, end="", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for k, (d, _) in MUTANTS.items():
            print(k, d)
    elif cmd == "run":
        sys.exit(run_one(sys.argv[2]))
    elif cmd == "all":
        ids = sys.argv[2:] or ["M0"] + list(MUTANTS)
        drive(ids)
