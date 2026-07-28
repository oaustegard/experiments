"""Path resolution for experiments in this repo.

Before this repo was split out of `oaustegard/claude-workspace`, scripts
hardcoded `/home/user/claude-workspace` and reached into
`.spokes/<name>` and `experiments/<name>` beneath it. That prefix no
longer exists, which left 32 scripts non-runnable as checked in.

Two things need resolving, and they resolve differently:

* Sibling experiments live in *this* repo, so they are found from
  `__file__` and need no configuration.
* Spoke checkouts (`remax`, `remax_kb`, `muninn.austegard.com`, ...) live
  outside this repo and vary by machine, so they are found from
  `EXPERIMENTS_SPOKES_ROOT` with a short probe list as fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repo root — the directory holding this `_lib/` package.
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

#: Probed in order when EXPERIMENTS_SPOKES_ROOT is unset. The historical
#: layout is kept last so an unmigrated checkout still resolves.
_SPOKES_FALLBACKS = (
    EXPERIMENTS_ROOT / ".spokes",
    Path.home() / ".spokes",
    Path("/home/user/claude-workspace/.spokes"),
)


def experiment(name: str) -> Path:
    """Return the directory of a sibling experiment in this repo.

    Raises FileNotFoundError if it is absent — a missing sibling is a
    typo or a renamed experiment, and failing loudly here beats a
    confusing ImportError several frames later.
    """
    path = EXPERIMENTS_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"no experiment {name!r} in {EXPERIMENTS_ROOT} "
            f"(expected a sibling directory)"
        )
    return path


def spokes_root() -> Path:
    """Return the directory holding spoke checkouts.

    `EXPERIMENTS_SPOKES_ROOT` wins when set. Otherwise the first existing
    fallback is used; if none exist the first fallback is returned anyway
    so that callers report a missing path rather than a missing config.
    """
    env = os.environ.get("EXPERIMENTS_SPOKES_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in _SPOKES_FALLBACKS:
        if candidate.is_dir():
            return candidate
    return _SPOKES_FALLBACKS[0]


def spoke(name: str) -> Path:
    """Return the checkout directory of a spoke repo.

    Does not check existence: several callers build a path to an artifact
    that a prior step is expected to produce, so absence is the caller's
    error to report with its own context.
    """
    return spokes_root() / name
