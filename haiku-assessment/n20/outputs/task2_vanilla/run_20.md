# Bug Review: merge_user_prefs

## Critical Issue: Mutates Input

The function **mutates the input** `default_prefs` despite the docstring claiming "Does not mutate inputs."

```python
result = default_prefs  # This is a reference, not a copy
for key, value in user_overrides.items():
    if value is not None:
        result[key] = value  # Modifies default_prefs directly
return result
```

When you assign `result = default_prefs`, you're creating a reference to the same dictionary object. Any modifications to `result` directly modify `default_prefs`.

### Example of the Bug
```python
defaults = {'theme': 'light', 'font_size': 12}
overrides = {'theme': 'dark'}
merged = merge_user_prefs(defaults, overrides)

print(defaults)  # {'theme': 'dark', 'font_size': 12} ← MUTATED!
print(merged)    # {'theme': 'dark', 'font_size': 12}
```

## Fix

Create a copy of the default preferences:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs.copy()  # Create a shallow copy
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

## Alternative Using dict() Constructor

```python
result = dict(default_prefs)
```

Both approaches create a shallow copy, preventing mutation of the original.
