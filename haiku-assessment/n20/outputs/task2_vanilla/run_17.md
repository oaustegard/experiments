# Code Review: merge_user_prefs Function

## Critical Bug: Mutation of Input Dictionary

The function has a critical bug that **violates its own contract**. The docstring states "Does not mutate inputs," but it does.

**The Problem:**
```python
result = default_prefs
```

This line does NOT create a copy. It creates a reference to the same dictionary object. Any modifications to `result` directly mutate the original `default_prefs` dictionary.

**Example:**
```python
defaults = {"theme": "light", "lang": "en"}
overrides = {"theme": "dark"}

merged = merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"theme": "dark", "lang": "en"}  ← MUTATED!
```

## The Fix

Replace the first line with a dictionary copy:

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

Or use the dict constructor:
```python
result = dict(default_prefs)
```

## Secondary Consideration

For nested dictionaries (dict values containing other dicts), a shallow copy may not be sufficient—you might need `copy.deepcopy()`. However, this depends on the intended semantics for nested structures, which the docstring doesn't specify.
