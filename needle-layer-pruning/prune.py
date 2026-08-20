"""Remove layers from Needle 2's scanned stack, preserving everything else.

The stack is an `nn.scan`, so every per-layer tensor carries a leading
`num_layers` axis (see `needle-depth-growth/`). Dropping layers is a slice along
axis 0 plus a config bump — but two things in this architecture are indexed by
absolute layer position and have to be handled:

* **MHC lanes.** `Stack.__call__` assigns lane `i % 4` to layer `i`, and the
  `pre_off` / `post_off` biases follow from it. Deleting layers renumbers
  everything after the cut, so a surviving layer keeps its lane only if the
  number of layers removed before it is a multiple of 4. Removing a **contiguous
  block whose size is a multiple of 4** satisfies that for every survivor, which
  is why this module only offers that shape.
* **Engram sites.** `engram_layers=(2, 15)` is absolute. A site inside the cut is
  dropped; a site after the cut is renumbered by the number of layers removed
  before it.

`lane_preserving()` states the rule as an assertion rather than a comment, so a
future caller cannot quietly violate it.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

from needle.model.architecture import TransformerConfig


def lane_preserving(start: int, count: int, mhc_lanes: int = 4) -> bool:
    """A contiguous cut preserves every survivor's lane iff its size % lanes == 0."""
    return count % mhc_lanes == 0


def prune(params, cfg, start: int, count: int):
    """Drop layers [start, start+count) and return (params, config).

    Raises on a cut that would silently change the lane assignment, which is a
    perturbation unrelated to depth and would contaminate any measurement.
    """
    if not lane_preserving(start, count, cfg.mhc_lanes):
        raise ValueError(
            f"cut of {count} layers is not a multiple of mhc_lanes={cfg.mhc_lanes}; "
            "every surviving layer after the cut would change lane"
        )
    if start < 0 or start + count > cfg.num_layers:
        raise ValueError(f"cut [{start}, {start + count}) outside 0..{cfg.num_layers}")

    keep = np.array([i for i in range(cfg.num_layers) if not (start <= i < start + count)])
    flat = dict(flatten_dict(params))
    out = {}
    for path, value in flat.items():
        a = np.asarray(value)
        if path[0] == "stack" and path[1] != "final_norm" and a.shape[:1] == (cfg.num_layers,):
            out[path] = jnp.asarray(a[keep])
        else:
            out[path] = value

    sites = tuple(l - count if l >= start + count else l
                  for l in cfg.engram_layers if not (start <= l < start + count))
    new_cfg = TransformerConfig(**{**cfg.__dict__,
                                   "num_layers": cfg.num_layers - count,
                                   "engram_layers": sites})
    return unflatten_dict(out), new_cfg


def valid_cuts(cfg, count: int) -> list[int]:
    """Every start position for a cut of `count` layers."""
    return list(range(cfg.num_layers - count + 1))
