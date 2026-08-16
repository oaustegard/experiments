# Task: semver_cmp

Write a Python module defining exactly one public function:

```python
def compare(a: str, b: str) -> int
```

Compare two Semantic Versioning 2.0.0 version strings. Return -1 if `a < b`, 0 if equal
precedence, 1 if `a > b`. Do not use any third-party library.

Rules (SemVer 2.0.0 precedence, https://semver.org):
- A version is `MAJOR.MINOR.PATCH` with optional `-PRERELEASE` and optional `+BUILD`
  (in that order: `1.2.3-alpha.1+build.5`).
- MAJOR/MINOR/PATCH are non-negative integers **without leading zeros** (`0` is fine,
  `01` is not). All three must be present (`1.2` is invalid).
- PRERELEASE is a dot-separated list of identifiers. Each identifier is either numeric
  (digits only, no leading zeros unless it is exactly `0`) or alphanumeric (letters,
  digits, hyphens, containing at least one non-digit). Empty identifiers are invalid
  (`1.0.0-alpha..1`).
- BUILD metadata (`+...`) is **ignored** for precedence: `1.0.0+a` vs `1.0.0+b` -> 0.
  Build identifiers are dot-separated, non-empty, letters/digits/hyphens only
  (leading zeros allowed there).
- Precedence: compare MAJOR, then MINOR, then PATCH numerically. If still equal:
  a version **with** a prerelease has lower precedence than the same version without one.
  Two prereleases compare identifier by identifier, left to right:
  - numeric identifiers compare numerically
  - alphanumeric identifiers compare as ASCII strings
  - numeric identifiers always have **lower** precedence than alphanumeric ones
  - if all compared identifiers are equal, the shorter list has lower precedence
- Canonical example (ascending): `1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta
  < 1.0.0-beta < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0`.
- Invalid input (either argument) raises `ValueError`: missing parts, leading zeros in
  numeric fields, empty identifiers, illegal characters, `v` prefix, whitespace.

No I/O. Standard library only (`re` allowed). Only `compare` is tested.
