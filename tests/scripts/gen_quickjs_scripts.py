#!/usr/bin/env python3
"""Write tests/suites/quickjs/js/<category>_<nn>_<slug>.js from quickjs_scripts.py and copy
the upstream tests/*.js (from the pinned quickjs release in tests/tools/quickjs) into
tests/suites/quickjs/js/upstream/.  Generated files are committed; this is the source of record.
Run: python3 tests/scripts/gen_quickjs_scripts.py
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "drivers"))
import quickjs_scripts as S  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "suites", "quickjs", "js")
UPSTREAM = ["assert.js", "fixture_cyclic_import.js", "test_bigint.js", "test_builtin.js", "test_closure.js",
            "test_cyclic_import.js", "test_language.js", "test_loop.js", "test_rw_handler.js", "test_std.js",
            "test_worker.js", "test_worker_module.js"]


# Included with `qjs --std -I` before every own test: qjs has no timers as globals (only os.setTimeout).
PRELUDE = """// needs `qjs --std` (globals os, std)
globalThis.setTimeout = (f, ms, ...a) => os.setTimeout(() => f(...a), ms);
globalThis.clearTimeout = os.clearTimeout;
globalThis.setInterval = (f, ms, ...a) => { const h = { t: null }; const tick = () => { h.t = os.setTimeout(() => { tick(); f(...a); }, ms); }; tick(); return h; };
globalThis.clearInterval = h => os.clearTimeout(h.t);
globalThis.queueMicrotask = f => Promise.resolve().then(f);
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.js")):
        os.remove(old)
    n = 0
    for cat, items in S.SCRIPTS.items():
        for i, (slug, js) in enumerate(items, 1):
            with open(os.path.join(OUT, "%s_%02d_%s.js" % (cat, i, slug)), "w") as f:
                f.write(js.strip("\n") + "\n")
            n += 1
    with open(os.path.join(OUT, "_prelude.js"), "w") as f:
        f.write(PRELUDE)
    print("wrote %d scripts to %s" % (n, OUT))
    try:
        import quickjs as Q
        src = Q.source_dir()
    except Exception as e:
        print("upstream tests not copied (%s)" % e)
        return
    up = os.path.join(OUT, "upstream")
    os.makedirs(up, exist_ok=True)
    for name in UPSTREAM:
        shutil.copy(os.path.join(src, "tests", name), os.path.join(up, name))
    print("copied %d upstream test files" % len(UPSTREAM))


if __name__ == "__main__":
    main()
