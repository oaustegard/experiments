#!/usr/bin/env python3
"""Tests for _lib/evidence.py. No deps, no network, no creds.

    python3 _lib/tests/test_evidence.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _lib.evidence import check_file, main  # noqa: E402


def _doc(body: str, files: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, content in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)
    (d / "RESULTS.md").write_text(body)
    return d / "RESULTS.md"


METRICS = json.dumps({"tab": {"bal_acc": 0.7781, "acc": 0.766, "n": 1200, "verdict": "pass"},
                      "runs": [{"score": 2.5}], "a.b": {"c": 3}})


def statuses(body: str, files=None) -> list[str]:
    return [r.status for r in check_file(_doc(body, files or {"m.json": METRICS}))]


def test_matching_number_resolves():
    assert statuses("Balanced accuracy 0.778. <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["resolved"]


def test_number_that_drifted_is_a_mismatch():
    assert statuses("Balanced accuracy 0.791. <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["mismatch"]


def test_coarser_display_precision_still_matches():
    assert statuses("about 0.8 <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["resolved"]


def test_wrong_rounding_does_not_match():
    assert statuses("0.77 <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["mismatch"]


def test_percent_form_matches_fraction():
    assert statuses("77.8% balanced <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["resolved"]


def test_thousands_separator_and_unicode_minus():
    files = {"m.json": json.dumps({"n": 1200, "d": -0.25})}
    assert statuses("n = 1,200 <!-- @evidence[[m.json#n]] -->", files) == ["resolved"]
    assert statuses("delta −0.25 <!-- @evidence[[m.json#d]] -->", files) == ["resolved"]


def test_scientific_notation_matches_at_mantissa_precision():
    files = {"m.json": json.dumps({"r": 3.870157e-08})}
    assert statuses("ratio 3.9e-8 <!-- @evidence[[m.json#r]] -->", files) == ["resolved"]
    assert statuses("ratio 3.7e-8 <!-- @evidence[[m.json#r]] -->", files) == ["mismatch"]
    assert statuses("ratio 3.9e-7 <!-- @evidence[[m.json#r]] -->", files) == ["mismatch"]


def test_count_fraction_matches_stored_rate():
    files = {"m.json": json.dumps({"rate": 0.95, "acc": 0.9375})}
    assert statuses("19/20 runs <!-- @evidence[[m.json#rate]] -->", files) == ["resolved"]
    assert statuses("75/80 (94%) <!-- @evidence[[m.json#acc]] -->", files) == ["resolved"]
    assert statuses("18/20 runs <!-- @evidence[[m.json#rate]] -->", files) == ["mismatch"]


def test_bare_integer_does_not_absorb_a_small_rate():
    files = {"m.json": json.dumps({"rate": 0.025, "zero": 0.0})}
    assert statuses("0 usable commands in 40 <!-- @evidence[[m.json#rate]] -->", files) == ["mismatch"]
    assert statuses("1 of 40 <!-- @evidence[[m.json#rate]] -->", files) == ["resolved"]
    assert statuses("2 of 40 <!-- @evidence[[m.json#rate]] -->", files) == ["mismatch"]
    assert statuses("1/40 runs <!-- @evidence[[m.json#rate]] -->", files) == ["resolved"]
    assert statuses("0 of 40 <!-- @evidence[[m.json#zero]] -->", files) == ["resolved"]


def test_any_number_on_the_line_may_match():
    assert statuses("rose 0.606 → 0.778 <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["resolved"]


def test_two_refs_on_one_line_each_checked():
    body = "0.778 vs 0.766 <!-- @evidence[[m.json#tab.bal_acc]] --> <!-- @evidence[[m.json#tab.acc]] -->"
    assert statuses(body) == ["resolved", "resolved"]


def test_missing_file_is_dangling():
    assert statuses("x 1 <!-- @evidence[[nope.json#a]] -->") == ["dangling"]


def test_missing_key_is_dangling():
    assert statuses("x 1 <!-- @evidence[[m.json#tab.f1]] -->") == ["dangling"]


def test_list_index_and_json_pointer():
    assert statuses("2.5 <!-- @evidence[[m.json#runs.0.score]] -->") == ["resolved"]
    assert statuses("3 <!-- @evidence[[m.json#/a.b/c]] -->") == ["resolved"]


def test_no_locator_is_anchored():
    assert statuses("see file <!-- @evidence[[m.json]] -->") == ["anchored"]


def test_claim_without_number_is_anchored():
    assert statuses("it improved <!-- @evidence[[m.json#tab.bal_acc]] -->") == ["anchored"]


def test_string_value_in_claim_resolves():
    assert statuses("verdict: pass <!-- @evidence[[m.json#tab.verdict]] -->") == ["resolved"]


def test_ref_alone_on_its_line_checks_line_above():
    body = "Balanced accuracy 0.791.\n<!-- @evidence[[m.json#tab.bal_acc]] -->\n"
    assert statuses(body) == ["mismatch"]


def test_code_blocks_and_inline_code_are_ignored():
    body = "```\n0.1 <!-- @evidence[[nope.json#a]] -->\n```\nuse `@evidence[[nope.json#a]]` like this\n"
    assert statuses(body) == []


def test_csv_row_selector():
    files = {"r.csv": "model,arm,acc\nhaiku,prose,0.95\nhaiku,tele,0.70\n"}
    assert statuses("0.70 <!-- @evidence[[r.csv#row(model=haiku,arm=tele).acc]] -->", files) == ["resolved"]
    assert statuses("0.70 <!-- @evidence[[r.csv#row(model=haiku).acc]] -->", files) == ["dangling"]
    assert statuses("0.70 <!-- @evidence[[r.csv#row(model=haiku,arm=tele).f1]] -->", files) == ["dangling"]


def test_jsonl_record_index():
    files = {"r.jsonl": '{"s": 1}\n{"s": 0.25}\n'}
    assert statuses("0.25 <!-- @evidence[[r.jsonl#1.s]] -->", files) == ["resolved"]


def test_line_range_locator():
    files = {"log.txt": "a\nb\nc\n"}
    assert statuses("see <!-- @evidence[[log.txt#L2-L3]] -->", files) == ["anchored"]
    assert statuses("see <!-- @evidence[[log.txt#L2-L9]] -->", files) == ["dangling"]


def test_target_resolves_relative_to_the_doc():
    assert statuses("2.5 <!-- @evidence[[sub/m.json#runs.0.score]] -->", {"sub/m.json": METRICS}) == ["resolved"]


def test_main_exit_codes():
    good = _doc("0.778 <!-- @evidence[[m.json#tab.bal_acc]] -->", {"m.json": METRICS})
    bad = _doc("0.9 <!-- @evidence[[m.json#tab.bal_acc]] -->", {"m.json": METRICS})
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert main([str(good), "--json"]) == 0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - test runner reports and continues
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
