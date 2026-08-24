#!/usr/bin/env python3
"""Check RESULTS.md against the artifacts it describes.

Sub-second: it reads the checked-in spec/config/oracle files and asserts that the
numbers and names the writeup quotes are still the ones in those files. It does NOT
re-run coherence or remex's suite — those numbers are stamped from the trial and are
reproducible from remex#80, not from this directory.

Usage: python3 recheck.py
"""

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
ART = HERE / "artifacts"
FAILS = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else '  -> ' + detail}")
    if not ok:
        FAILS.append(label)


def main():
    results = (HERE / "RESULTS.md").read_text()
    lib = (ART / "lib.spec.md").read_text()
    cfg = json.loads((ART / "coherence.config.json").read_text())
    rot = (ART / "test_rotation_totality.py").read_text()
    pack = (ART / "test_packing_totality.py").read_text()

    print("config")
    check("language is python", cfg.get("language") == "python", repr(cfg.get("language")))
    check("codeExt is [py]", cfg.get("codeExt") == ["py"], repr(cfg.get("codeExt")))
    check("config stays small (writeup says 11 lines)",
          len((ART / "coherence.config.json").read_text().splitlines()) <= 13)

    print("spec claims")
    root = (ART / "root.spec.md").read_text()

    def claims_of(text):
        block = text[text.index("## works when"):]
        block = block[:block.index("## why")]
        return [l[2:].strip() for l in block.splitlines() if l.startswith("- ")]

    claims = claims_of(lib)
    total = len(claims) + len(claims_of(root))
    check("7 grammar claims across both specs", total == 7, f"counted {total}")
    check("two boundary claims", sum(c.startswith("boundary") for c in claims) == 2,
          str([c[:40] for c in claims]))
    check("one parity claim", sum(c.startswith("parity") for c in claims) == 1)
    check("every via-test oracle is named in an artifact",
          all(name in rot or name in pack
              for name in re.findall(r'via test "([^"]+)"', lib)),
          str(re.findall(r'via test "([^"]+)"', lib)))

    print("refutations")
    refs = re.findall(r"^- ([^:]+):", lib[lib.index("## refutations"):lib.index("## works when")],
                      re.M)
    check("3 invariants carry a refutation", len(refs) == 3, str(len(refs)))
    check("each refuted invariant is declared",
          all(r.strip() in lib[lib.index("## invariants"):lib.index("## refutations")]
              for r in refs))

    print("oracles loop live registries, not literals")
    check("rotation oracle loops ROTATION_CODES",
          "ROTATIONS = sorted(ROTATION_CODES)" in rot and "for name in ROTATION_CODES" in rot)
    check("packing oracle loops SUPPORTED_BITS",
          'parametrize("bits", SUPPORTED_BITS)' in pack)
    check("both oracles carry a domain floor",
          "len(ROTATIONS) >= 3" in rot and "len(SUPPORTED_BITS) >= 5" in pack)

    print("writeup numbers match the artifacts")
    check("writeup's claim count matches the spec",
          f"{total} claims, {total} green" in results,
          f"specs carry {total} grammar claims")
    for n in ("267 passed", "288 Python tests passing", "936 tests, 936 pass"):
        check(f"stamped figure present: {n}", n in results)
    check("no via-test name in the writeup is absent from the oracles",
          all(name in rot or name in pack
              for name in re.findall(r"test_[a-z_]+", results)
              if name.startswith("test_every") or name.startswith("test_constructible")))

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
