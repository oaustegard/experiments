import pytest
from solution import render


def test_basic():
    assert render("Hello {name}!", {"name": "Ada"}) == "Hello Ada!"


def test_underscore_names():
    assert render("{_a}{a_1}", {"_a": "x", "a_1": "y"}) == "xy"


def test_missing_keys_message_exact():
    with pytest.raises(KeyError) as ei:
        render("{name} is {age}", {"name": "x"})
    assert ei.value.args[0] == "missing keys: age"
