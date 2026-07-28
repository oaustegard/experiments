# Bug Review: merge_user_prefs

## Critical Issue: Mutation of Input Dictionary

The function claims in its docstring that it "Does not mutate inputs," but this is **false**. The line:

```python
result = default_prefs
```

does not create a copy — it creates a reference to the same dictionary object. Any modifications to `result` directly mutate the original `default_prefs` dictionary.

### Example:
```python
defaults = {"theme": "dark", "notifications": True}
overrides = {"theme": "light"}
merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"theme": "light", "notifications": True}
# defaults was mutated!
```

## Fix

Create a shallow copy of `default_prefs` before modifying it:

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

This ensures the original `default_prefs` dictionary remains unchanged, matching the documented behavior.
