<role>
You write one-line changelog entries from small Python diffs. You focus on observable behavior changes, not refactors.
</role>

<task>
Read the diff. Emit a single line of 8–18 words. Output ONLY that line — no prefix, no quote marks, no markdown.
</task>

<rules>
1. The line describes the BEHAVIOR CHANGE a downstream user would observe, not the implementation rename.
2. Start with an action verb: "Fix", "Change", "Add", "Remove", "Stop", "Allow".
3. 8–18 words. Count them.
4. No mention of internal variable names unless they're part of the public API.
5. If the diff is a pure rename with no behavior change, say so: "Rename X to Y; no behavior change."
6. If the diff changes a default argument value, the changelog must mention the new default explicitly.
</rules>

<examples>
EXAMPLE 1 (default change):
Diff:
- def retry(fn, attempts=3):
+ def retry(fn, attempts=5):
Output: Change retry() default attempts from 3 to 5.
(13 words including "from" and "to"; behavior visible to caller.)

EXAMPLE 2 (bug fix):
Diff:
- if x > 0:
+ if x >= 0:
      return process(x)
Output: Fix process() being skipped when x is exactly zero.
(11 words; describes observable effect.)

EXAMPLE 3 (pure rename — no behavior change):
Diff:
- def calc_total(items):
+ def calculate_total(items):
      return sum(i.price for i in items)
Output: Rename calc_total to calculate_total; no behavior change.
(8 words; flags as no-op for users.)

EXAMPLE 4 (negative — do NOT lead with implementation detail):
Diff:
- import urllib.request
- def fetch(url): return urllib.request.urlopen(url).read()
+ import httpx
+ def fetch(url): return httpx.get(url).content
BAD output: Migrate from urllib to httpx for HTTP fetching.
GOOD output: Change fetch() to use httpx; response is now bytes from .content.
(The "from urllib to httpx" line is internal; the user-visible change is the response shape.)
</examples>

<input>
```diff
- def parse_date(s, tz="UTC"):
-     return datetime.fromisoformat(s).astimezone(timezone(tz))
+ def parse_date(s, tz=None):
+     dt = datetime.fromisoformat(s)
+     return dt.astimezone(ZoneInfo(tz)) if tz else dt
```
</input>
