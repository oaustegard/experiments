# Code Review: merge_user_prefs Function

## Critical Bug: Mutation of Input

The function mutates the `default_prefs` input dictionary despite the docstring claiming it "Does not mutate inputs."

**Problem:**
```python
result = default_prefs  # This is a reference, not a copy
```

This line assigns a reference to the same dictionary object, not a copy. When the function later modifies `result[key] = value`, it is directly modifying the `default_prefs` dictionary.

**Example demonstrating the bug:**
```python
defaults = {'a': 1, 'b': 2}
overrides = {'b': 3}
result = merge_user_prefs(defaults, overrides)

print(defaults)  # Output: {'a': 1, 'b': 3}  <- MUTATED!
print(result)    # Output: {'a': 1, 'b': 3}
```

## Solution

Use `.copy()` or the `dict()` constructor to create a shallow copy:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # Create a copy
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

Alternatively, using `dict()`:
```python
result = dict(default_prefs)
```

## Additional Note

The logic itself is correct (only merging non-None overrides), but the mutation bug violates the contract stated in the docstring and will cause unexpected behavior in code that relies on immutability.
