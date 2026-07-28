# Bug Review: merge_user_prefs()

## Critical Issue: Mutation of Input

**The function mutates `default_prefs` despite the docstring claiming it doesn't.**

```python
result = default_prefs  # This is a reference, not a copy
```

This line assigns a reference to `default_prefs` to `result`. When you later modify `result[key] = value`, you're directly modifying the original `default_prefs` dictionary that was passed in.

### Example:
```python
defaults = {"theme": "dark", "lang": "en"}
overrides = {"theme": "light"}
merged = merge_user_prefs(defaults, overrides)

print(defaults)  # {"theme": "light", "lang": "en"} — MUTATED!
```

### Fix:
Create a proper copy before modifying:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # Create a shallow copy
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

Alternatively, use `dict()` constructor:
```python
result = dict(default_prefs)
```

## Summary

This is a **silent contract violation** — the function's behavior contradicts its docstring. Callers relying on the docstring's guarantee that inputs are not mutated will experience unexpected bugs where their original dictionaries change unexpectedly.
