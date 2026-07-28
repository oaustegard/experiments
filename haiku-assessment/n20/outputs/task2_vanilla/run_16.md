# Bug Review: merge_user_prefs Function

## Critical Issues

### 1. **Mutates Input - Violates Contract (CRITICAL)**
The function docstring claims "Does not mutate inputs," but it directly mutates `default_prefs`:
```python
result = default_prefs  # This creates a reference, not a copy
```

When you do `result[key] = value`, you are modifying the original `default_prefs` dict. This breaks the documented contract.

**Fix:**
```python
result = default_prefs.copy()  # Create a shallow copy
```

Or use:
```python
result = {**default_prefs}  # Dict unpacking (Python 3.5+)
```

## Additional Issues

### 2. **Shallow Copy Only (Potential Issue)**
If `default_prefs` or `user_overrides` contain nested mutable objects (dicts, lists), a shallow copy won't prevent mutation of nested structures. Depending on use case, you may need:
```python
result = copy.deepcopy(default_prefs)
```

### 3. **Silent Overwrite Behavior**
The condition `if value is not None` means `None` values in `user_overrides` are ignored. This is probably intentional, but it's worth noting: you cannot use this function to explicitly set a preference to `None` (it will just keep the default). This might be desired, but it's a behavioral quirk worth documenting.

## Summary
**Main bug:** The function mutates `default_prefs` despite claiming not to. Add `.copy()` to fix.
