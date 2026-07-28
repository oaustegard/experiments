# Bug Review: `merge_user_prefs` Function

## Critical Issue: Mutation of Input

**Problem:** The function claims "Does not mutate inputs" but it violates this contract.

```python
result = default_prefs  # This is a reference, not a copy
```

This line assigns a reference to `default_prefs` instead of creating a copy. When the function modifies `result[key] = value`, it directly mutates the `default_prefs` dict passed in. The caller's original default preferences dict is altered.

**Example:**
```python
defaults = {"theme": "light", "lang": "en"}
overrides = {"theme": "dark"}
merge_user_prefs(defaults, overrides)
print(defaults)  # Output: {"theme": "dark", "lang": "en"} — mutated!
```

## Fix

Use a shallow copy:
```python
result = default_prefs.copy()
```

Or for nested dicts, use a deep copy:
```python
import copy
result = copy.deepcopy(default_prefs)
```

## Secondary Consideration

If nested dicts are involved, shallow copy won't fully protect them. Use `copy.deepcopy()` if any preference values are themselves mutable objects.
