"""Copy a WGSL engine and the shared modules it imports into one directory of a site repo.

Engines here import `../../_lib/wgsl/runtime.js` and `../../_lib/wgsl/kernels.js`. A page published
outside this repo gets the engine next to `wgsl-runtime.js` and `wgsl-kernels.js`, with the imports
rewritten, so the set works from any directory.

    from _lib.wgsl.vendor import vendored_engine, vendor_engine
    vendor_engine(EXP / "ettin-native" / "web" / "engine.js", site / "bsky" / "embedders")
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime.js"
KERNELS = HERE / "kernels.js"
# module -> (import specifier in an engine, vendored file name)
MODULES = {RUNTIME: ('"../../_lib/wgsl/runtime.js"', "wgsl-runtime.js"),
           KERNELS: ('"../../_lib/wgsl/kernels.js"', "wgsl-kernels.js")}


def vendored_engine(engine: Path) -> str:
    """The engine's source with its shared-module imports pointed at sibling wgsl-*.js files."""
    src = Path(engine).read_text()
    for spec, name in MODULES.values():
        if src.count(spec) > 1:
            raise ValueError(f"{engine}: more than one import of {spec}")
        src = src.replace(spec, f'"./{name}"')
    if "../../_lib/" in src:
        raise ValueError(f"{engine}: imports a _lib module vendor.py does not know")
    return src


def vendored_modules() -> dict:
    """{file name: source} for the shared modules, kernels.js importing the vendored runtime."""
    return {"wgsl-runtime.js": RUNTIME.read_text(),
            "wgsl-kernels.js": KERNELS.read_text().replace('"./runtime.js"', '"./wgsl-runtime.js"')}


def vendor_engine(engine: Path, out_dir: Path, name: str = "engine.js") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(vendored_engine(engine))
    for fname, src in vendored_modules().items():
        (out_dir / fname).write_text(src)
