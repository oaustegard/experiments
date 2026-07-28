# Bug Review: merge_user_prefs Function

## Critical Issue: Mutation of Input Parameter

**Severity: High**

The function violates its own contract. The docstring states "Does not mutate inputs," but the code mutates `default_prefs`.

### The Problem
```python
result = default_prefs  # This creates a reference, not a copy
```

This line assigns a reference to the same dictionary object, not a new copy. When you modify `result[key]`, you're modifying the original `default_prefs` dictionary.

### Example of the Bug
```python
defaults = {"theme": "light", "font_size": 12}
overrides = {"theme": "dark"}

merge_user_prefs(defaults, overrides)

print(defaults)  # Output: {"theme": "dark", "font_size": 12}
                 # Expected: {"theme": "light", "font_size": 12}
```

The original `defaults` dict has been unexpectedly modified.

## The Fix

Create a shallow copy instead:
```python
result = default_prefs.copy()  # or dict(default_prefs)
```

Or use the copy module for nested dicts (if needed):
```python
import copy
result = copy.deepcopy(default_prefs)
```

For most preference-merging use cases, `.copy()` is sufficient and more efficient.
