# Bug Report: merge_user_prefs Function

## Critical Bug: Mutation of Input

The function violates its own contract. The docstring states "Does not mutate inputs," but the code **mutates `default_prefs`** directly.

### The Problem

```python
result = default_prefs  # This is an alias, not a copy
```

This line creates a reference to the same dict object, not a new dict. When the function modifies `result`, it modifies the original `default_prefs` that was passed in.

### Example

```python
defaults = {"color": "blue", "size": "medium"}
overrides = {"color": "red"}
merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"color": "red", "size": "medium"}
# defaults was mutated!
```

## The Fix

Create a shallow copy of `default_prefs`:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # or dict(default_prefs)
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

## Additional Consideration

If `default_prefs` or `user_overrides` contain nested mutable objects (lists, dicts), a shallow copy will still allow those nested objects to be mutated. If true immutability is required, use `copy.deepcopy()`.
