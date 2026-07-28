# Bug Review: merge_user_prefs

## Critical Bug: Mutates Input

The function violates its own docstring promise: **"Does not mutate inputs."**

```python
result = default_prefs
```

This line assigns a reference, not a copy. When the loop modifies `result[key]`, it directly mutates the original `default_prefs` dict.

### Example
```python
defaults = {'theme': 'light', 'font_size': 12}
overrides = {'theme': 'dark'}

merged = merge_user_prefs(defaults, overrides)
# Bug: defaults is now {'theme': 'dark', 'font_size': 12}  ← mutated!
```

## Fix

Replace the first line with:
```python
result = default_prefs.copy()
```

This creates a shallow copy, preserving the original `default_prefs` unchanged while still allowing modifications to `result`.

## Secondary Issue: Shallow Copy Limitation

If `default_prefs` contains nested dicts or lists, `.copy()` creates a shallow copy. Modifications to nested objects would still affect the original. For deep nesting, use `copy.deepcopy()`, but the shallow copy fixes the immediate bug for typical preference dicts.
