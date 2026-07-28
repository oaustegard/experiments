# Bug Review: merge_user_prefs Function

## Critical Issue: Mutation of Input Dictionary

**The docstring claims the function "Does not mutate inputs," but it violates this contract.**

### Problem
```python
result = default_prefs
```

This line creates a reference to the same dictionary object, not a copy. Any modifications to `result` directly mutate `default_prefs`:

```python
defaults = {"theme": "light", "font_size": 12}
overrides = {"theme": "dark"}
merge_user_prefs(defaults, overrides)
print(defaults)  # {"theme": "dark", "font_size": 12} — MUTATED!
```

### Fix
Create a shallow copy before modifying:

```python
result = default_prefs.copy()  # or dict(default_prefs)
for key, value in user_overrides.items():
    if value is not None:
        result[key] = value
return result
```

Or use a dict comprehension:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = {**default_prefs, **{k: v for k, v in user_overrides.items() if v is not None}}
    return result
```

## Secondary Issue: Silent Filtering of None Values

The function silently skips `None` values in overrides. This is a design choice, but it's worth noting: a user cannot explicitly clear a preference by passing `None`. If this is intentional, the docstring should document it. If unintended, change the condition to process all keys.
