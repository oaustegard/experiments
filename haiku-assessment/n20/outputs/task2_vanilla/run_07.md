# Bug Review: merge_user_prefs

## Critical Bug: Mutation of Input

The function violates its own contract. The docstring says "Does not mutate inputs," but it **does mutate the `default_prefs` dict**.

### What's happening:
```python
result = default_prefs
```

This line creates an alias, not a copy. Both `result` and `default_prefs` reference the same dict object. When the code modifies `result[key]`, it's actually modifying the original `default_prefs` dict.

### Demonstration:
```python
defaults = {"theme": "light", "notifications": True}
user = {"theme": "dark"}

merge_user_prefs(defaults, user)
print(defaults)  # Output: {"theme": "dark", "notifications": True}
# defaults was mutated!
```

## Fix:
Replace the first line with:
```python
result = default_prefs.copy()
```

This creates a shallow copy that can be modified without affecting the original `default_prefs`.

## Additional Consideration:
If the dicts could contain nested mutable objects (lists, dicts), a shallow copy may not be sufficient and `copy.deepcopy()` might be needed. However, for simple preference dicts with scalar values, `.copy()` is the correct fix.
