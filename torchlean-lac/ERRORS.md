# Errors

## Lean tactic written fluently and never compiled

The first draft of `score_lt_self` closed the positivity step with
`mul_pos_of_neg_of_neg |>.elim (fun _ => by positivity) (fun _ => by positivity)`,
which is not a valid use of that lemma and would not have elaborated. I caught it
by re-reading the file before the first `lake env lean` run, not by running it,
and replaced it with `mul_self_pos.mpr` plus `linarith`. Nothing shipped, but the
shape is worth logging: Lean tactic prose reads plausible at a glance and the only
check that means anything is the elaborator.

## Assumed a detached build survives a session suspend

`lake build NN NNCI NNExamples NNTests` was launched with `nohup ... &`. The CCotw
session suspended, and the build died at 3942/4166 with no sentinel written. A
relaunch, also detached, never emitted a line. Diagnosed by comparing the log's
mtime against `date` rather than reading the absent sentinel as "still running" —
16 minutes of no writes with no `lake` process is a dead build, not a slow one.
Cost about 20 minutes. Fixed by running the build in the foreground under
`timeout 580`; it completed clean, 4352 jobs.

## `#print axioms` reports a private name

`LACScratch/Core.lean` opens with `module` but no `@[expose] public section`, so
its declarations are private and the axiom audit prints
`_private.LACScratch.Core.0.LAC.score_lt_self`. The audit itself is valid — the
dependency list is the three standard axioms — but the file is a scratch target
compiled with `lake env lean`, not a Lake library, and a version intended for
import would need the `@[expose] public section` header TorchLean's own modules
use.
