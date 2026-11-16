"""Planted defects for harness calibration (never touches /home/user/remex).

Each mutant rewrites one function of the *imported* remex by editing its
source text in memory and re-executing it in the module's namespace, so the
defect is a one-line, plausible edit and stays in sync with the real code.

    REMEX_PT_MUTANT=<name> pytest test_search_props.py ...   # harness applies it
    python mutation_calibration.py                           # runs them all

MUTANTS maps name -> (description, [(owner, attr, old, new, count), ...]).
"""
from __future__ import annotations

import inspect
import sys
import textwrap

from remex.core import Quantizer
from remex.gpu import GPUSearcher
from remex.ivf import IVFCoarseIndex

MUTANTS = {
    "adc_skips_effective_norms": (
        "search_adc multiplies by raw stored norms (skips _effective_norms/_score_norms)",
        [(Quantizer, "search_adc", "self._score_norms(compressed, precision)", "compressed.norms", 2)]),
    "search_drops_query_offset": (
        "Quantizer.search forgets _query_offset (centred scores lose q.mu)",
        [(Quantizer, "search", ") + self._query_offset(query)", ") + 0.0", 1)]),
    "ivf_coarse_drops_query_offset": (
        "IVFCoarseIndex.search_coarse forgets _query_offset",
        [(IVFCoarseIndex, "search_coarse", "offset = self.quantizer._query_offset(query)", "offset = 0.0", 1)]),
    "gpu_batch_drops_query_offset": (
        "GPUSearcher.search_batch forgets the centred q.mu offset",
        [(GPUSearcher, "search_batch", "if self._pq.mean is not None:", "if False:", 1)]),
    "batch_drops_row_bias": (
        "Quantizer.search_batch forgets _row_bias (centred rows lose mean_dots term)",
        [(Quantizer, "search_batch", "bias = self._row_bias(compressed, precision)\n        if bias is not None:",
          "bias = None\n        if bias is not None:", 1)]),
    "twostage_fine_uses_coarse_norms": (
        "search_twostage reranks with the coarse-precision norms (per-precision confusion)",
        [(Quantizer, "search_twostage", "fine_norms = self._score_norms(compressed, None)",
          "fine_norms = self._score_norms(compressed, coarse_precision)", 1)]),
    "dir_len_cache_ignores_precision": (
        "_direction_lengths caches under one key for every precision (stale across levels)",
        [(Quantizer, "_direction_lengths", "cached = compressed._dir_len.get(precision)",
          "cached = compressed._dir_len.get(None)", 1),
         (Quantizer, "_direction_lengths", "compressed._dir_len[precision] = lengths",
          "compressed._dir_len[None] = lengths", 1)]),
    "adc_chunk_boundary_norm_shift": (
        "_adc_score_chunked takes norms from the previous row at every chunk after the first",
        [(Quantizer, "_adc_score_chunked", "chunk_norms = None if norms is None else norms[start:end]",
          "chunk_norms = None if norms is None else norms[max(start - 1, 0):max(start - 1, 0) + (end - start)]", 1)]),
    "packed_adc_ignores_precision_shift": (
        "search_adc on PackedVectors forgets to right-shift for lower precision",
        [(Quantizer, "search_adc",
          "shift = 0 if (precision is None or precision == self.bits) else (self.bits - precision)",
          "shift = 0", 1)]),
    "twostage_returns_candidate_relative_ids": (
        "search_twostage returns positions in the candidate set instead of corpus rows",
        [(Quantizer, "search_twostage", "original_idx = coarse_idx[rerank_order]",
          "original_idx = rerank_order", 1)]),
    "ivf_twostage_drops_bias": (
        "IVFCoarseIndex.search_twostage rerank omits the centred row bias",
        [(IVFCoarseIndex, "search_twostage", "fine_scores = fine_scores + bias[coarse_idx]",
          "fine_scores = fine_scores", 1)]),
    "gpu_adc_full_precision_norms": (
        "GPUSearcher.search_adc scales lower-precision scores with full-precision norms",
        [(GPUSearcher, "search_adc",
          "ops.gather_sum(table, indices, chunk_size=chunk_size),\n            precision=precision,",
          "ops.gather_sum(table, indices, chunk_size=chunk_size),\n            precision=None,", 1)]),
    "search_k_equals_n_boundary": (
        "Quantizer.search: `k >= n` becomes `k > n`, so k == n reaches argpartition(kth=n)",
        [(Quantizer, "search", "if k >= compressed.n:", "if k > compressed.n:", 1)]),
    "ivf_probe_tiebreak_reversed": (
        "IVFCoarseIndex.probe_cells breaks Hamming ties by descending cell id",
        [(IVFCoarseIndex, "probe_cells", "order = np.lexsort((all_cells, hd))[:nprobe]",
          "order = np.lexsort((-all_cells.astype(np.int64), hd))[:nprobe]", 1)]),
    "ivf_flat_ignores_precision": (
        "IVFCoarseIndex.search_coarse scores with the full-precision table but shifted codes",
        [(IVFCoarseIndex, "search_coarse", "centroids = self.quantizer._resolve_centroids(\n            self.compressed, precision\n        )\n        table",
          "centroids = self.quantizer._resolve_centroids(\n            self.compressed, None\n        )\n        table", 1)]),
}


def _is_static(owner, attr):
    return isinstance(inspect.getattr_static(owner, attr), staticmethod)


def apply(name: str) -> None:
    desc, edits = MUTANTS[name]
    grouped: dict = {}
    for owner, attr, old, new, count in edits:
        grouped.setdefault((owner, attr), []).append((old, new, count))
    for (owner, attr), subs in grouped.items():
        src = inspect.getsource(getattr(owner, attr))     # original indentation
        for old, new, count in subs:
            got = src.count(old)
            if got != count:
                raise RuntimeError(f"mutant {name}: expected {count} occurrence(s) of {old!r} in "
                                   f"{owner.__name__}.{attr}, found {got}")
            src = src.replace(old, new)
        src = textwrap.dedent(src)
        # drop decorators (staticmethod) from the re-executed source
        lines = [l for l in src.splitlines() if not l.startswith("@staticmethod")]
        ns: dict = {}
        exec(compile("\n".join(lines), f"<mutant {name}>", "exec"),
             sys.modules[owner.__module__].__dict__, ns)
        newfn = ns[attr]
        setattr(owner, attr, staticmethod(newfn) if _is_static(owner, attr) else newfn)
