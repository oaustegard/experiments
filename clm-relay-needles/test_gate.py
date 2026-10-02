"""The register edit gate goes red on the Phase 3 failure (14 counts reset) and green on a normal step."""
import re

from gen_tally import episode
from relay_tally import register_violation

ep = episode(96, 0)
A = ep["assets"]
table = lambda d: "| asset | holder | count |\n|---|---|---|\n" + "".join(f"| {a} | {h} | {c} |\n" for a, (h, c) in d.items())
prev = {a: ("Xu", 30) for a in A}
ok = dict(prev); ok[A[0]] = ("Sato", 31); ok[A[1]] = ("Novak", 32)
reset = dict(prev)
for a in A[:14]:
    reset[a] = ("Xu", 1)
dropped = {a: v for a, v in prev.items() if a != A[5]}
jump = {a: (h, c + 5) for a, (h, c) in prev.items()}
assert register_violation(table(prev), table(ok), A, 136) is None
assert "went down" in register_violation(table(prev), table(reset), A, 136)
assert "missing" in register_violation(table(prev), table(dropped), A, 136)
assert "rose by 200" in register_violation(table(prev), table(jump), A, 136)
print("gate tests pass")
