# Bug Review: merge_user_prefs

## Critical Bug: Mutation of Input

The function **mutates the input** despite claiming it doesn't. The line:

```python
result = default_prefs
```

assigns a reference to the same dict object, not a copy. All subsequent modifications via `result[key] = value` directly mutate `default_prefs`.

### Example:
```python
defaults = {"theme": "light", "lang": "en"}
overrides = {"theme": "dark"}
merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"theme": "dark", "lang": "en"}  ← MUTATED!
```

## Fix

Create a shallow copy:

```python
result = default_prefs.copy()
```

Or use dict unpacking (Python 3.5+):

```python
result = {**default_prefs}
```

The docstring promise "Does not mutate inputs" is violated by the current implementation.
