# Code Review: merge_user_prefs Function

## Critical Bug: Mutation of Input

The function docstring claims "Does not mutate inputs," but this is **false**. The line `result = default_prefs` creates a reference to the same dictionary object, not a copy. Any modifications to `result` directly mutate the original `default_prefs` dictionary.

### Example:
```python
defaults = {'a': 1, 'b': 2}
overrides = {'b': 99}
merged = merge_user_prefs(defaults, overrides)
print(defaults)  # {'a': 1, 'b': 99} — MUTATED!
```

## Fix

Replace the first line with:
```python
result = default_prefs.copy()
```

Or use:
```python
result = dict(default_prefs)
```

For nested dictionaries, use `copy.deepcopy()` if deep copies are needed.

## Additional Issues

1. **No validation**: The function doesn't handle non-dictionary inputs, which could cause AttributeError if `None` or a non-dict is passed.

2. **Limited error handling**: If `user_overrides` is not iterable or lacks an `.items()` method, the function crashes without clear messaging.

3. **Shallow merge only**: If preference values are themselves dictionaries, they are replaced entirely rather than merged recursively. This may or may not be intended.

## Corrected Version

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```
