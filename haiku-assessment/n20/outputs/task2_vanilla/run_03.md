# Code Review: merge_user_prefs Function

## Critical Bug: Mutation of Input Dictionary

**Issue:** The function mutates the `default_prefs` input despite the docstring claiming "Does not mutate inputs."

**Root Cause:** Line 3 assigns `result = default_prefs` creates an alias to the same dictionary object, not a copy. When `result[key] = value` executes, it modifies the original `default_prefs` dictionary.

**Example:**
```python
defaults = {'theme': 'dark', 'font_size': 12}
overrides = {'theme': 'light'}
merge_user_prefs(defaults, overrides)
# defaults is now {'theme': 'light', 'font_size': 12} — mutated!
```

## Fix

Replace line 3 with:
```python
result = default_prefs.copy()
```

This creates a shallow copy so modifications affect only the returned dictionary.

## Impact

- **Severity:** High — violates the documented contract and causes unexpected side effects
- **Risk:** Callers assume inputs are safe; mutations can lead to subtle bugs in complex workflows
- **Test Coverage Gap:** The docstring claim masks the bug; no test validates immutability

## Additional Considerations

- For deeply nested dictionaries, `copy()` (shallow) may still share references to nested objects; use `copy.deepcopy()` if nested mutation is a concern.
- Consider using an immutable structure (e.g., `types.MappingProxyType`) if the caller should never modify the result.
