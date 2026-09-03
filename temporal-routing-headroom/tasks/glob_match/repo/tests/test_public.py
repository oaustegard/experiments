import pytest
from solution import glob_match


def test_char_class_range():
    assert glob_match("[a-z]x", "mx")
    assert not glob_match("[a-z]x", "Mx")
    assert glob_match("[0-9][0-9]", "42")


def test_char_class_negation():
    assert glob_match("[!a-c]x", "dx")
    assert not glob_match("[!a-c]x", "ax")
