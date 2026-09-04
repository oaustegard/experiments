import pytest
from solution import parse_csv_line


def test_escaped_quotes():
    assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"', "x"]


def test_only_escaped_quote():
    assert parse_csv_line('""""') == ['"']
