import pytest
from solution import compare


def test_core_ordering():
    assert compare("1.0.0", "2.0.0") == -1
    assert compare("2.1.0", "2.0.9") == 1
    assert compare("2.1.3", "2.1.3") == 0
    assert compare("0.9.9", "0.10.0") == -1

def test_semver_canonical_chain():
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
             "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    for i in range(len(chain) - 1):
        assert compare(chain[i], chain[i + 1]) == -1, (chain[i], chain[i + 1])
        assert compare(chain[i + 1], chain[i]) == 1

def test_prerelease_lower_than_release():
    assert compare("1.0.0-rc.1", "1.0.0") == -1
    assert compare("1.0.0", "1.0.0-rc.1") == 1

def test_numeric_vs_alphanumeric_identifier():
    assert compare("1.0.0-1", "1.0.0-a") == -1
    assert compare("1.0.0-999", "1.0.0-a") == -1

def test_numeric_identifiers_numeric_order():
    assert compare("1.0.0-beta.2", "1.0.0-beta.11") == -1

def test_shorter_prerelease_lower():
    assert compare("1.0.0-alpha", "1.0.0-alpha.0") == -1

def test_build_metadata_ignored():
    assert compare("1.0.0+build1", "1.0.0+build2") == 0
    assert compare("1.0.0-alpha+x", "1.0.0-alpha+y") == 0
    assert compare("1.0.0+big", "1.0.0") == 0

def test_hyphen_in_prerelease_identifier():
    assert compare("1.0.0-alpha-1", "1.0.0-alpha-1") == 0

def test_equal_prereleases():
    assert compare("1.0.0-rc.1.x-y", "1.0.0-rc.1.x-y") == 0

@pytest.mark.parametrize("bad", [
    "1.2", "1", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03",
    "1.0.0-alpha..1", "1.0.0-", "1.0.0-alpha.01", "1.0.0-01",
    "v1.2.3", " 1.2.3", "1.2.3 ", "1.2.3-al pha", "1.2.3+", "1.2.3+a..b",
    "1.2.3-alpha_1", "", "a.b.c", "1.2.-3",
])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        compare(bad, "1.0.0")
    with pytest.raises(ValueError):
        compare("1.0.0", bad)

def test_zero_versions_ok():
    assert compare("0.0.0", "0.0.0") == 0
    assert compare("1.0.0-0", "1.0.0-1") == -1

def test_leading_zero_in_numeric_prerelease_invalid_but_alnum_ok():
    # "0a1" contains a letter -> alphanumeric, leading zero fine
    assert compare("1.0.0-0a1", "1.0.0-0a1") == 0
    with pytest.raises(ValueError):
        compare("1.0.0-00", "1.0.0")
