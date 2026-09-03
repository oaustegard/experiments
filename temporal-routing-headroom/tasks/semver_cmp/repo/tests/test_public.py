import pytest
from solution import compare


def test_semver_canonical_chain():
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
             "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for i in range(len(chain) - 1):
        assert compare(chain[i], chain[i + 1]) == -1, (chain[i], chain[i + 1])
        assert compare(chain[i + 1], chain[i]) == 1


def test_numeric_identifiers_numeric_order():
    assert compare("1.0.0-beta.2", "1.0.0-beta.11") == -1
