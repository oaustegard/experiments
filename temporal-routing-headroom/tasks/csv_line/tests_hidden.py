import pytest
from solution import parse_csv_line


def test_simple():
    assert parse_csv_line("a,b,c") == ["a", "b", "c"]

def test_empty_string_one_empty_field():
    assert parse_csv_line("") == [""]

def test_single_comma_two_empty_fields():
    assert parse_csv_line(",") == ["", ""]

def test_trailing_comma():
    assert parse_csv_line("a,") == ["a", ""]

def test_quoted_with_comma():
    assert parse_csv_line('"a,b",c') == ["a,b", "c"]

def test_escaped_quotes():
    assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"', "x"]

def test_only_escaped_quote():
    assert parse_csv_line('""""') == ['"']

def test_empty_quoted_field():
    assert parse_csv_line('""') == [""]

def test_empty_quoted_then_comma():
    assert parse_csv_line('"",') == ["", ""]

def test_whitespace_preserved():
    assert parse_csv_line(" a , b ") == [" a ", " b "]

def test_newline_inside_quotes():
    assert parse_csv_line('"a\nb",c') == ["a\nb", "c"]

def test_no_csv_module():
    import solution, inspect
    src = inspect.getsource(solution)
    assert "import csv" not in src

def test_quote_in_unquoted_field_raises():
    with pytest.raises(ValueError):
        parse_csv_line('a"b')

def test_space_then_quote_raises():
    with pytest.raises(ValueError):
        parse_csv_line(' "a"')

def test_char_after_closing_quote_raises():
    with pytest.raises(ValueError):
        parse_csv_line('"a"b')

def test_space_after_closing_quote_raises():
    with pytest.raises(ValueError):
        parse_csv_line('"a" ,b')

def test_unterminated_quote_raises():
    with pytest.raises(ValueError):
        parse_csv_line('"abc')

def test_unterminated_after_escape_raises():
    with pytest.raises(ValueError):
        parse_csv_line('"a""')
