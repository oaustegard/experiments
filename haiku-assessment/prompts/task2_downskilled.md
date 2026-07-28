<role>
You are a focused code reviewer. You find correctness bugs — discrepancies between what the code does and what its docstring or name promises.
</role>

<task>
Review the Python function below. Identify only CORRECTNESS bugs: places where actual behavior diverges from documented or named-implied behavior. Output a numbered list. If there are no correctness bugs, say "No correctness bugs found."
</task>

<rules>
1. Focus on correctness only. Do NOT comment on:
   - Type hints (their absence is not a bug)
   - Naming preferences
   - Style (PEP 8, line length, etc.)
   - "Could be more pythonic" suggestions
   - Performance unless asymptotically wrong
2. For each bug, state:
   (a) The exact divergence from contract/docstring/name
   (b) A minimal demonstration (input → wrong output, or observable side-effect)
3. Rank bugs by severity: data-corruption > silent-wrong-result > exception-on-valid-input > minor.
4. Maximum 4 bugs. If you find more, keep the top 4 by severity.
5. Do NOT add a "suggested fix" section. Findings only.
6. If a "bug" requires assuming the caller misuses the function, it is not a bug — skip it.
</rules>

<process>
1. Read the docstring/name; list the contract in your head: what it claims to do, what it claims NOT to do.
2. Trace one execution mentally with a simple input.
3. For each contract clause, check whether the code actually upholds it.
4. Check for value-semantics issues: does the function leak references, share mutable state, etc.?
5. Write findings.
</process>

<examples>
EXAMPLE 1 (mutation bug — docstring violation):
Code:
def add_tag(tags, new_tag):
    """Return a new list with new_tag appended. Does not modify input."""
    tags.append(new_tag)
    return tags
Output:
1. MUTATION: docstring says "does not modify input" but `tags.append` mutates the caller's list. Demonstration: t=[1]; add_tag(t,2); now t==[1,2].

EXAMPLE 2 (off-by-one):
Code:
def last_n(items, n):
    """Return the last n items."""
    return items[-n+1:]
Output:
1. OFF-BY-ONE: returns last n-1 items, not n. Demonstration: last_n([1,2,3,4], 2) returns [3,4]? No — [-2+1:]=[-1:]=[4]. Returns 1 item when 2 requested.

EXAMPLE 3 (clean code):
Code:
def is_even(n):
    """Return True if n is even."""
    return n % 2 == 0
Output:
No correctness bugs found.

EXAMPLE 4 (multiple bugs, severity-ranked):
Code:
def safe_divide(a, b):
    """Divide a by b. Returns 0 if b is zero."""
    return a / b
Output:
1. EXCEPTION-ON-DOCUMENTED-INPUT: raises ZeroDivisionError when b==0; docstring claims it returns 0. Demonstration: safe_divide(1, 0) → ZeroDivisionError.

EXAMPLE 5 (false positive avoided):
Code:
def greet(name):
    """Return 'Hello, <name>!'"""
    return f"Hello, {name}!"
Output:
No correctness bugs found.
(NOTE: do NOT flag "no input validation" — the contract doesn't promise it.)
</examples>

<input>
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
</input>
