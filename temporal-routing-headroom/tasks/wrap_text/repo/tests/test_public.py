import pytest
from solution import wrap_text


def test_paragraphs():
    assert wrap_text("a\n\nb", 10) == "a\n\nb"
    assert wrap_text("a\n\n\n\nb", 10) == "a\n\nb"


def test_paragraph_with_whitespace_only_line():
    assert wrap_text("a\n   \nb", 10) == "a\n\nb"


def test_multi_paragraph_wrap():
    text = "one two three four\n\nfive six seven"
    assert wrap_text(text, 9) == "one two\nthree\nfour\n\nfive six\nseven"
