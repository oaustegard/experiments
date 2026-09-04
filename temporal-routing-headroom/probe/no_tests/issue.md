# Bug report

This repository ships no tests. A user reports that `cron_next` returns wrong results
when the day-of-month and the day-of-week fields are both constrained, and when either
one is constrained on its own.

There is no failing assertion to work from. Read the code, decide what the correct
behaviour is, and make the source implement it.

Constraints:
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: cron_next.
- A hidden suite grades this work. Convincing yourself the code is right is the task.
