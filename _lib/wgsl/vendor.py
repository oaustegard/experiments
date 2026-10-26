"""Copy a WGSL engine and the shared runtime it imports into one directory of a site repo.

Engines here import `../../_lib/wgsl/runtime.js`. A page published outside this repo gets the
engine next to `wgsl-runtime.js` with that import rewritten, so the pair works from any directory.

    from _lib.wgsl.vendor import vendored_engine, vendor_engine
    vendor_engine(EXP / "ettin-native" / "web" / "engine.js", site / "bsky" / "embedders")
"""
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime.js"
IMPORT = '"../../_lib/wgsl/runtime.js"'
VENDORED = '"./wgsl-runtime.js"'


def vendored_engine(engine: Path) -> str:
    """The engine's source with its runtime import pointed at a sibling wgsl-runtime.js."""
    src = Path(engine).read_text()
    if src.count(IMPORT) != 1:
        raise ValueError(f"{engine}: expected one import of {IMPORT}")
    return src.replace(IMPORT, VENDORED)


def vendor_engine(engine: Path, out_dir: Path, name: str = "engine.js") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(vendored_engine(engine))
    (out_dir / "wgsl-runtime.js").write_text(RUNTIME.read_text())
