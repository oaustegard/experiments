# Bug Review: merge_user_prefs

## Critical Issue: Mutation of Input

**Problem:** The function mutates `default_prefs` despite claiming not to mutate inputs.

```python
result = default_prefs  # This creates a reference, not a copy
for key, value in user_overrides.items():
    if value is not None:
        result[key] = value  # This modifies the original dict
return result
```

When you assign `result = default_prefs`, you're not creating a new dictionary—you're creating another reference to the same object. Any modifications to `result` directly modify `default_prefs`.

## Example of the Bug

```python
defaults = {'theme': 'light', 'lang': 'en'}
overrides = {'lang': 'fr'}

merged = merge_user_prefs(defaults, overrides)

print(defaults)  # Output: {'theme': 'light', 'lang': 'fr'}  ← MUTATED!
print(merged)    # Output: {'theme': 'light', 'lang': 'fr'}
```

## Solution

Use `.copy()` to create a shallow copy:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # Create a new dict
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

Alternatively, use the dict unpacking syntax:

```python
result = {**default_prefs, **{k: v for k, v in user_overrides.items() if v is not None}}
```

## Summary

The function has a **critical contract violation**—it explicitly promises not to mutate inputs but does so. This is a logic bug that can cause silent data corruption in calling code.
