# Bug Review: merge_user_prefs()

## Critical Issue: Mutation of Input

**Problem:** The function claims it "Does not mutate inputs" in the docstring, but it **does mutate** `default_prefs`.

```python
result = default_prefs  # This creates an alias, not a copy
```

This line assigns a reference to the same dictionary object. When you later do `result[key] = value`, you're directly modifying the original `default_prefs` dictionary.

**Example demonstrating the bug:**
```python
defaults = {"theme": "light", "font_size": 12}
overrides = {"theme": "dark"}

merge_user_prefs(defaults, overrides)
# defaults is now {"theme": "dark", "font_size": 12} — MUTATED!
```

## How to Fix

Create a shallow copy of `default_prefs` instead:

```python
result = default_prefs.copy()  # or dict(default_prefs)
```

This ensures the original dictionary remains unchanged while the function returns the merged result.

## Secondary Issues

1. **No handling of nested dicts:** If the prefs contain nested dictionaries, `copy()` creates a shallow copy. Nested dicts will still be shared references. Consider `copy.deepcopy()` if deep merging is needed.

2. **No validation:** The function doesn't validate that the keys in `user_overrides` are actually valid preference keys. Invalid overrides are silently accepted.
