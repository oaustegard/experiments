import pytest
from solution import format_table


def test_exact_example():
    out = format_table([[1, "hi"], [22, "world!"]], ["n", "word"], max_col_width=5)
    expected = (
        "+----+-------+\n"
        "| n  | word  |\n"
        "+----+-------+\n"
        "| 1  | hi    |\n"
        "| 22 | world |\n"
        "|    | !     |\n"
        "+----+-------+"
    )
    assert out == expected


def test_header_wider_than_data():
    out = format_table([["a"]], ["header"])
    assert "| header |" in out
    assert "| a      |" in out


def test_newline_then_wrap():
    out = format_table([["ab\ncdef"]], ["h"], max_col_width=3)
    lines = out.split("\n")
    assert lines[3] == "| ab  |"
    assert lines[4] == "| cde |"
    assert lines[5] == "| f   |"
