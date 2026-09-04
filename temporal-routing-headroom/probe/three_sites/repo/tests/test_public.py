import pytest
from solution import format_table


def test_center_align_extra_space_right():
    # width 3 ("abc"), value "a" -> leftover 2 -> 1 left 1 right: " a "
    # value "ab" -> leftover 1 -> extra goes right: "ab " with no left space
    out = format_table([["a"], ["ab"]], ["abc"], aligns=["c"])
    lines = out.split("\n")
    assert lines[3] == "|  a  |"
    assert lines[4] == "| ab  |"


def test_empty_rows_ok():
    out = format_table([], ["a", "b"])
    expected = (
        "+---+---+\n"
        "| a | b |\n"
        "+---+---+\n"
        "+---+---+"
    )
    assert out == expected
