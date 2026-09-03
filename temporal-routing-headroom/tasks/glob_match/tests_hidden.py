import pytest
from solution import glob_match


def test_literal():
    assert glob_match("a/b.py", "a/b.py")
    assert not glob_match("a/b.py", "a/b.pyc")

def test_star_within_segment():
    assert glob_match("*.py", "util.py")
    assert not glob_match("*.py", "lib/util.py")
    assert glob_match("a/*.py", "a/x.py")
    assert not glob_match("a/*.py", "a/b/x.py")

def test_star_matches_empty():
    assert glob_match("a*b", "ab")
    assert glob_match("*", "anything")

def test_star_not_empty_path():
    assert not glob_match("*", "")

def test_question_mark():
    assert glob_match("?x", "ax")
    assert not glob_match("?x", "x")
    assert not glob_match("a?c", "a/c")

def test_doublestar_zero_segments():
    assert glob_match("a/**/b", "a/b")

def test_doublestar_many_segments():
    assert glob_match("a/**/b", "a/x/b")
    assert glob_match("a/**/b", "a/x/y/z/b")
    assert not glob_match("a/**/b", "a/x/y/c")

def test_leading_doublestar():
    assert glob_match("**/b", "b")
    assert glob_match("**/b", "x/y/b")
    assert glob_match("**/*.py", "c.py")
    assert glob_match("**/*.py", "a/b/c.py")

def test_trailing_doublestar():
    assert glob_match("a/**", "a/x")
    assert glob_match("a/**", "a/x/y")
    assert not glob_match("a/**", "a")

def test_bare_doublestar():
    assert glob_match("**", "")
    assert glob_match("**", "a")
    assert glob_match("**", "a/b/c")

def test_doublestar_inside_segment_is_star():
    assert glob_match("a**b", "axxb")
    assert glob_match("a**b", "ab")
    assert not glob_match("a**b", "a/b")
    assert not glob_match("f**", "f/x")
    assert glob_match("f**", "foo")

def test_empty_pattern():
    assert glob_match("", "")
    assert not glob_match("", "a")

def test_char_class_basic():
    assert glob_match("[abc]x", "bx")
    assert not glob_match("[abc]x", "dx")

def test_char_class_range():
    assert glob_match("[a-z]x", "mx")
    assert not glob_match("[a-z]x", "Mx")
    assert glob_match("[0-9][0-9]", "42")

def test_char_class_negation():
    assert glob_match("[!a-c]x", "dx")
    assert not glob_match("[!a-c]x", "ax")

def test_class_never_matches_slash():
    assert not glob_match("a[!x]b", "a/b")

def test_literal_bracket_close_first():
    assert glob_match("[]]", "]")
    assert not glob_match("[]]", "x")
    assert glob_match("[!]]", "x")
    assert not glob_match("[!]]", "]")

def test_literal_hyphen():
    assert glob_match("[a-]", "-")
    assert glob_match("[a-]", "a")
    assert not glob_match("[a-]", "b")
    assert glob_match("[-z]", "-")

def test_unterminated_class_raises():
    with pytest.raises(ValueError):
        glob_match("[abc", "a")

def test_multiple_doublestars():
    assert glob_match("**/a/**/b", "x/a/y/b")
    assert glob_match("**/a/**/b", "a/b")

def test_star_backtracking():
    assert glob_match("*ab", "aab")
    assert glob_match("a*b*c", "aXbYc")
    assert glob_match("a*b*c", "abc")
    assert not glob_match("a*b*c", "aXbY")
