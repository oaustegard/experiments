# Harness assumptions the decomp run broke (running log)

1. 09:30 doctor/ONE STAGE assumed no stage legitimately holds >90% of failures. Decomp starts with every
   function a `missing` stub and later nearly every failure is `match` (category lives in the reason).
   Fix: `tests.one_stage_ok` (swarming-failing-tests 0.2.0, claude-skills 9d3cebb) + selftest case.
2. 09:30 workers hold Bash and the answer is on the internet; no way to deny commands while allowing Bash.
   Fix: `worker.deny` -> claude --disallowedTools (swarming-failing-tests + dispatching-subagents 0.2.0).
3. 09:30 one-signature-per-cluster: at the start the only cluster per TU is "not written". Runner fix:
   stub reason bucketed by function size (small/medium/large) -> up to 3 parallel clusters per TU.
4. 09:46 preflight treats any tracked `.o` as a build artefact when `.o` is in paths.junk; target.o
   fixtures are tracked on purpose. Config fix (junk without `.o`).
5. 10:04 tests are not independent: a static function exists in the object only once a caller is written.
   Workers made unused statics appear with __attribute__((used)) / noinline / noipa (31 files in 17 min).
   Runner fix: reject gcc-steering attributes/pragmas; stripped them (14 of 216 matches lost);
   absent reason now says "needs its callers written". Event logged.
6. 10:05 a worker's result JSON had a raw newline in a string -> "output not JSON", finished work
   discarded. Fix: strict=False in fanout + read_json (0.2.1, claude-skills 2bfd9bb) + fanout test.
7. 10:58 the gate only merges a flip. In decomp a caller matches only once its static callee exists and a
   static callee only appears once a caller exists, so 125 "partial" workers (39% of spend, $9.60) were
   discarded; 103 notes named a missing callee, 47/92 quoted frac >= 0.8. Fix: partial-credit gate mode
   (`gate.partial_credit = "frac"`, swarming-failing-tests 0.3.0, claude-skills 1722149), 2 selftest cases.
   Budget re-split at equal $/function: neat cap $36, lua ~$23.
8. 10:58 known.json kept only status/reason/stage, so a per-test measure could not be compared at the gate;
   record() now keeps the partial-credit field (same release).
