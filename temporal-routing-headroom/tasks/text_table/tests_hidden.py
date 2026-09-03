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

def test_simple_no_wrap():
    out = format_table([["a", "bb"]], ["x", "y"])
    expected = (
        "+---+----+\n"
        "| x | y  |\n"
        "+---+----+\n"
        "| a | bb |\n"
        "+---+----+"
    )
    assert out == expected

def test_header_wider_than_data():
    out = format_table([["a"]], ["header"])
    assert "| header |" in out
    assert "| a      |" in out

def test_right_align():
    out = format_table([[1], [22]], ["n"], aligns=["r"])
    lines = out.split("\n")
    assert "|  n |" in lines[1]
    assert "|  1 |" in lines[3]
    assert "| 22 |" in lines[4]

def test_center_align_extra_space_right():
    # width 3 ("abc"), value "a" -> leftover 2 -> 1 left 1 right: " a "
    # value "ab" -> leftover 1 -> extra goes right: "ab " with no left space
    out = format_table([["a"], ["ab"]], ["abc"], aligns=["c"])
    lines = out.split("\n")
    assert lines[3] == "|  a  |"
    assert lines[4] == "| ab  |"

def test_none_renders_empty():
    out = format_table([[None]], ["h"])
    assert out.split("\n")[3] == "|   |"

def test_str_conversion():
    out = format_table([[3.5]], ["v"])
    assert "| 3.5 |" in out

def test_newline_in_cell():
    out = format_table([["a\nb", "x"]], ["c1", "c2"])
    lines = out.split("\n")
    assert lines[3] == "| a  | x  |"
    assert lines[4] == "| b  |    |"

def test_newline_then_wrap():
    out = format_table([["ab\ncdef"]], ["h"], max_col_width=3)
    lines = out.split("\n")
    assert lines[3] == "| ab  |"
    assert lines[4] == "| cde |"
    assert lines[5] == "| f   |"

def test_top_alignment_of_short_cells():
    out = format_table([["tall" * 3, "x"]], ["a", "b"], max_col_width=4)
    lines = out.split("\n")
    # talltalltall -> tall/tall/tall ; x only on first line; col b width is 1
    assert lines[3] == "| tall | x |"
    assert lines[4] == "| tall |   |"
    assert lines[5] == "| tall |   |"

def test_multirow_wrap_heights():
    out = format_table([["aaaa", "b"], ["c", "dddd"]], ["x", "y"], max_col_width=2)
    expected = (
        "+----+----+\n"
        "| x  | y  |\n"
        "+----+----+\n"
        "| aa | b  |\n"
        "| aa |    |\n"
        "+----+----+"
    )
    # row 2: c fits; dddd wraps to dd/dd
    expected = (
        "+----+----+\n"
        "| x  | y  |\n"
        "+----+----+\n"
        "| aa | b  |\n"
        "| aa |    |\n"
        "| c  | dd |\n"
        "|    | dd |\n"
        "+----+----+"
    )
    assert out == expected

def test_header_wraps_too():
    out = format_table([["a"]], ["abcdef"], max_col_width=3)
    lines = out.split("\n")
    assert lines[1] == "| abc |"
    assert lines[2] == "| def |"
    assert lines[3] == "+-----+"
    assert lines[4] == "| a   |"

def test_empty_rows_ok():
    out = format_table([], ["a", "b"])
    expected = (
        "+---+---+\n"
        "| a | b |\n"
        "+---+---+\n"
        "+---+---+"
    )
    assert out == expected

def test_no_trailing_newline():
    assert not format_table([["a"]], ["h"]).endswith("\n")

def test_validation():
    with pytest.raises(ValueError):
        format_table([["a"]], [])
    with pytest.raises(ValueError):
        format_table([["a", "b"]], ["h"])
    with pytest.raises(ValueError):
        format_table([["a"]], ["h"], aligns=["l", "r"])
    with pytest.raises(ValueError):
        format_table([["a"]], ["h"], aligns=["x"])
    with pytest.raises(ValueError):
        format_table([["a"]], ["h"], max_col_width=0)
