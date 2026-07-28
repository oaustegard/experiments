This function has a critical bug: **it mutates the `default_prefs` dict instead of creating a copy**.

The line `result = default_prefs` assigns a reference to the same dict object, not a copy. When the function later does `result[key] = value`, it's directly modifying the original `default_prefs` dict. This violates the docstring's promise: "Does not mutate inputs."

**Fix:**
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

**Additional consideration:** If `default_prefs` or `user_overrides` contain nested mutable objects (dicts, lists), a shallow copy won't fully protect them from mutation. For deep nesting, use `copy.deepcopy()` instead, but shallow copy is usually sufficient for preference dicts.
