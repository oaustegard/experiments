import pytest
from solution import render


def test_basic():
    assert render("Hello {name}!", {"name": "Ada"}) == "Hello Ada!"

def test_multiple_and_repeat():
    assert render("{a}-{b}-{a}", {"a": 1, "b": 2}) == "1-2-1"

def test_str_conversion():
    assert render("{x}", {"x": [1, 2]}) == "[1, 2]"

def test_escapes():
    assert render("{{literal}}", {}) == "{literal}"
    assert render("{{}}", {}) == "{}"

def test_escape_then_placeholder():
    assert render("{{{x}}}", {"x": 1}) == "{1}"

def test_no_placeholders():
    assert render("plain text", {}) == "plain text"

def test_empty_template():
    assert render("", {}) == ""

def test_underscore_names():
    assert render("{_a}{a_1}", {"_a": "x", "a_1": "y"}) == "xy"

def test_unused_values_ok():
    assert render("{a}", {"a": 1, "b": 2}) == "1"

def test_missing_keys_collected_sorted():
    with pytest.raises(KeyError) as ei:
        render("{b} {a} {b}", {})
    assert ei.value.args[0] == "missing keys: a, b"

def test_missing_keys_message_exact():
    with pytest.raises(KeyError) as ei:
        render("{name} is {age}", {"name": "x"})
    assert ei.value.args[0] == "missing keys: age"

def test_bad_name_digit_start():
    with pytest.raises(ValueError):
        render("{1x}", {"1x": 1})

def test_space_in_braces():
    with pytest.raises(ValueError):
        render("{ name }", {"name": 1})

def test_lone_open_brace():
    with pytest.raises(ValueError):
        render("{", {})

def test_unclosed_placeholder():
    with pytest.raises(ValueError):
        render("{unclosed", {})

def test_lone_close_brace():
    with pytest.raises(ValueError):
        render("a}b", {})

def test_empty_braces():
    with pytest.raises(ValueError):
        render("{}", {})

def test_valueerror_beats_missing_key():
    # template malformed AND key missing -> ValueError (malformedness is structural)
    with pytest.raises(ValueError):
        render("{a} {", {})
