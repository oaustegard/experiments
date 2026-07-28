# Bug Review: merge_user_prefs()

## Critical Issue: Mutation of Input

The function violates its own contract. The docstring states "Does not mutate inputs," but it **does mutate** `default_prefs`.

```python
result = default_prefs  # This is a reference, not a copy
```

This line creates a reference to the same dict object, not a new dict. When the function modifies `result[key]`, it directly modifies `default_prefs`.

### Example:
```python
defaults = {"theme": "light", "notifications": True}
overrides = {"theme": "dark"}

merge_user_prefs(defaults, overrides)

print(defaults)  # Output: {"theme": "dark", "notifications": True}
# defaults was mutated!
```

## Fix

Use `dict.copy()` to create a shallow copy:

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

Alternatively, use the dict unpacking operator:
```python
result = {**default_prefs}
```

## Summary

- **Severity**: High
- **Type**: Correctness / Contract violation
- **Root cause**: Aliasing instead of copying
- **Impact**: Caller's default preferences get unexpectedly modified, leading to bugs in subsequent calls
