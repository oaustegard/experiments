Write a one-line changelog entry for this change.

```diff
- def parse_date(s, tz="UTC"):
-     return datetime.fromisoformat(s).astimezone(timezone(tz))
+ def parse_date(s, tz=None):
+     dt = datetime.fromisoformat(s)
+     return dt.astimezone(ZoneInfo(tz)) if tz else dt
```
