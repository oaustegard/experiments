# Bug Review: `merge_user_prefs` Function

## Critical Issue: Mutation of Input

The function **mutates the input `default_prefs` dictionary** despite the docstring explicitly stating "Does not mutate inputs."

### Root Cause
```python
result = default_prefs
```

This line assigns a reference to the original dictionary, not a copy. All subsequent modifications to `result` directly modify `default_prefs`.

### Example
```python
defaults = {"theme": "light", "notifications": True}
overrides = {"theme": "dark"}
merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"theme": "dark", "notifications": True}
                 # MUTATED!
```

## Solution
Create a shallow copy of the default preferences:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # ← Fix: shallow copy
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

Alternatively:
```python
result = dict(default_prefs)  # Also creates a shallow copy
```

## Secondary Consideration
If nested dictionaries exist in the preferences, a shallow copy will not protect them from mutation. A deep copy may be needed depending on the data structure:
```python
import copy
result = copy.deepcopy(default_prefs)
```
