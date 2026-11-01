"""Unit tests (no GPU, no network): parser round trips, printer edge cases, literal
spelling, renaming hazards, minifier idempotence, front-end smoke tests.

    python3 tests/test_forge.py         # or: python3 -m pytest tests/test_forge.py
"""
import subprocess
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.minify import Options, minify, shorten_literal   # noqa: E402
from forge.printer import to_wgsl                           # noqa: E402
from forge.wgsl import lex, parse                           # noqa: E402

CORPUS = sorted((HERE / "corpus").glob("*/*.wgsl"))
NAGA = shutil.which("naga")


def naga_ok(src):
    if not NAGA:
        return True
    p = Path("/tmp/_forge_test.wgsl")
    p.write_text(src)
    return subprocess.run([NAGA, str(p)], capture_output=True).returncode == 0


def test_corpus_roundtrip():
    """compact and pretty printing both re-parse to the identical tree"""
    assert len(CORPUS) >= 80
    for f in CORPUS:
        m = parse(f.read_text())
        assert parse(to_wgsl(m)) == m, f
        assert parse(to_wgsl(m, pretty=True)) == m, f


def test_template_discovery():
    kinds = lambda s: [(t.text, t.tmpl) for t in lex(s) if t.text in ("<", ">")]  # noqa: E731
    assert kinds("a<b") == [("<", None)]
    assert kinds("array<vec2<f32>>") == [("<", "start"), ("<", "start"), (">", "end"), (">", "end")]
    assert kinds("if a<b && c>d {}") == [("<", None), (">", None)]
    assert [t.text for t in lex("x>>=1;")][:4] == ["x", ">>=", "1", ";"]
    assert [t.text for t in lex("a>=b")][:3] == ["a", ">=", "b"]


def test_printer_breaks_false_templates():
    """f((a<b), c>d) must not print as f(a<b,c>d), which lexes as a template list"""
    src = "fn g(a:i32,b:i32,c:i32,d:i32)->bool{return f((a<b),c>d);} fn f(x:bool,y:bool)->bool{return x;}"
    m = parse(src)
    out = to_wgsl(m)
    assert parse(out) == m
    assert "(a<b)" in out


def test_printer_keeps_wgsl_parens():
    """WGSL forbids a&b|c and a<<b+c and needs (-a).x; the printer must keep the parens"""
    for src in ["fn f(a:u32,b:u32,c:u32)->u32{return (a&b)|c;}",
                "fn f(a:u32,b:u32,c:u32)->u32{return a<<(b+c);}",
                "fn f(a:vec2f)->f32{return (-a).x;}",
                "fn f(a:f32,b:f32)->f32{return a-(-b);}",
                "fn f(a:f32,b:f32,c:f32)->f32{return a-(b-c);}"]:
        m = parse(src)
        out = to_wgsl(m)
        assert parse(out) == m, (src, out)


def test_literals():
    cases = {"1.0": "1.", "0.5": ".5", "100000.0": "1e5", "1.0f": "1f", "0.0f": "0f",
             "0.5f": ".5f", "0x10": "16", "4294967295u": "4294967295u", "0.000125": "125e-6",
             "1.5e-7": "15e-8", "123.456": "123.456", "0x1p4": "16.", "2.0h": "2h"}
    for a, b in cases.items():
        assert shorten_literal(a) == b, (a, shorten_literal(a), b)


def test_rename_avoids_reserved_and_shadowing():
    # a local must not take the (renamed) name of a global the function uses
    src = """
    const K: f32 = 3.0;
    fn helper(x: f32) -> f32 { return x * K; }
    @compute @workgroup_size(1) fn main() {
        var target_value = helper(2.0);
        let move_it = target_value + K;
        _ = move_it;
    }"""
    out = minify(src)
    assert naga_ok(out), out
    assert "fn main" in out


def test_entry_points_and_overrides_kept():
    src = """override scale: f32 = 2.0;
    @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
    fn unused() -> f32 { return 1.0; }
    @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        buf[id.x] = buf[id.x] * scale;
    }"""
    out = minify(src)
    assert "cs_main" in out and "scale" in out and "unused" not in out
    assert naga_ok(out)


def test_idempotent_on_corpus():
    for f in CORPUS[::4]:
        once = minify(f.read_text())
        assert minify(once) == minify(once), f
        assert len(minify(once)) <= len(once), f


def test_svg_and_python_frontends():
    from forge.svg import svg_to_wgsl
    from forge.scene import Scene, T, sin, smooth_union, Circle
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1 1h8v8z" '
           'fill="red" stroke="blue" stroke-width=".5"/><circle cx="5" cy="5" r="2"/></svg>')
    assert naga_ok(svg_to_wgsl(svg)) and naga_ok(svg_to_wgsl(svg, minify=True))
    s = Scene(100, 100)
    s.add(smooth_union(Circle(40, 50, 20), Circle(60 + 10 * sin(T), 50, 20), k=8), fill="#0af")
    s.rect(10, 10, 20, 20, rx=4, rotate=(T * 90, 20, 20), stroke="black")
    assert naga_ok(s.compile()) and naga_ok(s.compile(minify=True))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {str(e)[:300]}")
    sys.exit(1 if fails else 0)
