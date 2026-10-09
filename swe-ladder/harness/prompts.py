"""Task prompts for rung 1 (issue only) and rung 2 (issue + rejected diff + hidden failures)."""
from __future__ import annotations

import re

from common import BIN

HINT = {
    "django/django": "targets are Django test labels, e.g. `swe-test utils_tests.test_text` "
                     "or `swe-test admin_views.tests.AdminViewBasicTest.test_changelist_view`",
    "sympy/sympy": "targets are test files, optionally with -k, e.g. "
                   "`swe-test sympy/core/tests/test_basic.py -k test_subs`",
}

RUNNING = """Running code: the checkout's code runs only through two commands, used from inside the checkout:
  {bin}/swe-test TARGET...    the repo's own test runner; {hint}
  {bin}/swe-run CMD ARGS...   anything else, e.g. `{bin}/swe-run python repro.py`
Both run without network access. Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted."""

# Inserted as a format value, so single braces; {out} is filled at dispatch.
FINISH = """When you are done, and as your very last action, write {out} containing {"summary": "<two sentences: what you changed and how you checked it>"}. Then reply with one line."""

RUNG1 = """Resolve this GitHub issue in the {repo} repository.

The repository is checked out at {path} (commit {short}). Edit the source code there so the issue is fixed. Hidden tests will check your change: fix the behaviour the issue describes, including closely related cases, without breaking existing behaviour. Test files you add or edit are not graded.

{running}

<issue>
{issue}
</issue>

{finish}
"""

RUNG2 = """Resolve this GitHub issue in the {repo} repository.

The repository is checked out at {path} (commit {short}). A previous attempt edited it; its changes are still in the checkout (`git diff` shows them, and they are reproduced below). That attempt was rejected: hidden tests that check this issue failed on it. Their names and output are below. You cannot run the hidden tests, but you can write your own. Change the source code so those hidden tests would pass, without breaking existing behaviour. Keep, rework or discard the previous changes as you judge best.

{running}

<issue>
{issue}
</issue>

<previous_attempt_diff>
{diff}
</previous_attempt_diff>

<hidden_test_failures>
Failing tests: {failing}

{excerpt}
</hidden_test_failures>

{finish}
"""


def _common(task: dict, path) -> dict:
    return dict(repo=task["repo"], path=path, short=task["base_commit"][:10],
                issue=task["problem_statement"].strip(),
                running=RUNNING.format(bin=BIN, hint=HINT[task["repo"]]), finish=FINISH)


def rung1(task: dict, path) -> str:
    return RUNG1.format(**_common(task, path))


def rung2(task: dict, path, diff: str, failing: list[str], excerpt: str) -> str:
    return RUNG2.format(**_common(task, path), diff=diff.strip()[:12000] or "(no changes)",
                        failing=", ".join(failing) or "(none named; see output)", excerpt=excerpt)


# ── failure excerpt from a grading log ──

_DJANGO_SEP = re.compile(r"^={20,}$", re.M)
_SYMPY_HEAD = re.compile(r"^_{5,} (.+?) _{5,}$", re.M)


def failure_excerpt(task: dict, log: str, failing: list[str], limit: int = 8000) -> str:
    """The traceback blocks for the failing graded tests; the log tail when none match."""
    blocks = []
    if task["repo"] == "django/django":
        for chunk in _DJANGO_SEP.split(log)[1:]:
            head = chunk.strip().split("\n", 1)[0]          # "FAIL: test_x (app.tests.T)"
            name = head.split(": ", 1)[-1].strip()
            if any(name == f or name.startswith(f) or f.startswith(name) for f in failing):
                blocks.append("=" * 70 + "\n" + chunk.strip().split("\n" + "-" * 70 + "\nRan ")[0])
    else:
        heads = list(_SYMPY_HEAD.finditer(log))
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(log)
            body = log[m.start():end]
            if any(f in m.group(1) for f in failing):
                blocks.append(re.split(r"\n=+ tests finished", body)[0].strip())
    text = "\n\n".join(blocks)
    if not text:
        text = "(no traceback for these tests in the output; last lines of the run follow)\n" + log[-3000:]
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more characters cut]"
    return text
