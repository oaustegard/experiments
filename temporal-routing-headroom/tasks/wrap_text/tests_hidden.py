import pytest
from solution import wrap_text


def test_basic():
    assert wrap_text("the quick brown fox", 10) == "the quick\nbrown fox"

def test_exact_fit():
    assert wrap_text("aaa bbb", 7) == "aaa bbb"
    assert wrap_text("aaa bbb", 6) == "aaa\nbbb"

def test_word_exactly_width():
    assert wrap_text("abcde", 5) == "abcde"
    assert wrap_text("xx abcde", 5) == "xx\nabcde"

def test_overlong_word_chunks():
    assert wrap_text("abcdefghijklmno", 5) == "abcde\nfghij\nklmno"

def test_overlong_word_final_chunk_joinable():
    # "extraordin" + "ary" -- later word joins the final chunk line
    assert wrap_text("hello extraordinary world", 10) == "hello\nextraordin\nary world"

def test_overlong_after_content():
    assert wrap_text("ab abcdefgh", 5) == "ab\nabcde\nfgh"

def test_overlong_exact_multiple():
    # 10 chars at width 5 -> two full chunks, last chunk len == width stays current
    assert wrap_text("abcdefghij xy", 5) == "abcde\nfghij\nxy"

def test_whitespace_collapse():
    assert wrap_text("a\tb   c\nd", 20) == "a b c d"

def test_paragraphs():
    assert wrap_text("a\n\nb", 10) == "a\n\nb"
    assert wrap_text("a\n\n\n\nb", 10) == "a\n\nb"

def test_paragraph_with_whitespace_only_line():
    assert wrap_text("a\n   \nb", 10) == "a\n\nb"

def test_leading_trailing_blank_lines_stripped():
    assert wrap_text("\n\na b\n\n", 10) == "a b"

def test_empty_and_whitespace_input():
    assert wrap_text("", 10) == ""
    assert wrap_text("   \n \n\t", 10) == ""

def test_width_one():
    assert wrap_text("ab c", 1) == "a\nb\nc"

def test_no_trailing_artifacts():
    out = wrap_text("some words here to wrap", 8)
    assert not out.endswith("\n")
    assert all(not line.endswith(" ") for line in out.split("\n"))
    assert all(len(line) <= 8 for line in out.split("\n"))

def test_multi_paragraph_wrap():
    text = "one two three four\n\nfive six seven"
    assert wrap_text(text, 9) == "one two\nthree\nfour\n\nfive six\nseven"

def test_invalid_width():
    with pytest.raises(ValueError):
        wrap_text("a", 0)
    with pytest.raises(ValueError):
        wrap_text("a", -3)
