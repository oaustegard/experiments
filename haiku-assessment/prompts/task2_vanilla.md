Review this Python function for bugs and issues:

```python
def merge_user_prefs(default_prefs, user_overrides):
    """Merge user preference overrides on top of defaults.
    Returns the merged dict. Does not mutate inputs."""
    result = default_prefs
    for key, value in user_overrides.items():
        if value is not None:
            result[key] = value
    return result
```

Tell me what's wrong.
