"""Building blocks for long-running staged pipelines.

Extracted from `phase-a-bridges/scripts/common.py` and its near-identical
copy `te-bridges/scripts/te_common.py`, which had drifted only in their
data-directory env var and credential set. Both now re-export from here.

Deliberately credential-free and `httpx`-optional so it can be imported —
and tested — without the Cloudflare/Semantic Scholar environment the
original modules require at import time.

The checkpointing here exists because runs get killed: Claude Code on the
Web reaps idle background jobs, and a multi-hour staged sweep that cannot
resume is a sweep that never finishes.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

#: Default retry predicate. Kept as a late import so this module does not
#: hard-depend on httpx; callers doing HTTP should pass `retry_on`
#: explicitly.
DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (Exception,)


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base: float = 1.5,
    cap: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = None,
) -> T:
    """Call `fn` with jittered exponential backoff; return its value or raise.

    Waits `min(cap, base ** i)` plus up to 0.5s of jitter between attempts.
    Re-raises the last exception once `attempts` is exhausted, so a caller
    still sees the real failure rather than a wrapper.

    `sleep` and `log` are injected so the backoff schedule can be tested
    without spending wall-clock.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    emit = log if log is not None else (lambda m: print(m, file=sys.stderr))
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as e:
            last = e
            if i == attempts - 1:
                break
            wait = min(cap, base ** i) + random.uniform(0, 0.5)
            emit(f"  retry {i + 1}/{attempts} after {wait:.1f}s: {e}")
            sleep(wait)
    assert last is not None
    raise last


def chunked(seq: Iterable[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive lists of up to `n` items from `seq`.

    The final chunk is short rather than padded. Accepts any iterable, so
    it works on a generator that is too large to materialise.
    """
    if n < 1:
        raise ValueError(f"chunk size must be >= 1, got {n}")
    buf: list[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def save_json(path: Path, obj: Any, *, log: Callable[[str], None] | None = None) -> Path:
    """Write `obj` to `path` as JSON, atomically.

    Writes to a sibling `.tmp` then renames, so a job killed mid-write
    leaves the previous checkpoint intact rather than a truncated file.
    Parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
    if log is not None:
        log(f"  saved {path}")
    return path


def load_json(path: Path, default: Any = None) -> Any:
    """Read JSON from `path`, or return `default` if it does not exist.

    A missing checkpoint is the normal first-run case, not an error.
    """
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())
