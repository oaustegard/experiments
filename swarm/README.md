# swarm/ — Haiku worker orchestrator

Runs 1,000–3,000 headless Claude Haiku sessions, each fixing one cluster of
failing tests in its own git worktree, behind a serial merge gate.

```
python3 swarm/run.py run --concurrency 30 --max-usd 300 --model haiku   # main loop (resumable)
python3 swarm/run.py status          # merged/rejected/failed counts, spend, pass counts over time
python3 swarm/run.py queue --top 20  # what would be dispatched next
python3 swarm/run.py refresh         # full-suite baseline now (with bisect/revert)
python3 swarm/run.py run --fake ...  # whole machinery with swarm/fake_worker.py, no model calls
python3 swarm/queue.py R.json ...    # cluster any tests/run.py --json output
```
Key flags: `--max-workers N` (launches this invocation), `--concurrency`, `--max-usd`
(total ledger spend), `--per-task-usd` (2.0), `--timeout` (1500 s/worker), `--batch` (4
candidates per gate run), `--sample` (150), `--full-every` (25 merges), `--suites a,b`,
`--jobs` (gate test parallelism), `--root`/`--wt-root` (point at a scratch repo).

## Flow
1. **Baseline** (`merge.refresh`): every suite at main HEAD → `state/known.json`
   (latest status of every test), `state/baseline.json`, a row in `state/history.jsonl`.
2. **Queue** (`queue.py`): failing tests clustered by signature `"<stage>: <normalised
   reason>"` (numbers, paths, file:line:col, quoted identifiers masked; punctuation and
   C keywords kept). Clusters ≥20 tests are sharded (≤6 shards) so several workers can
   share a generic signature. Priority = 10·ladder level (smoke 6 > c-testsuite/chibicc 5
   > torture 4 > csmith 3 > sqlite/lua/quickjs 2 > doom/linux 1) + 3·log2(size) − penalty
   for failed attempts. Keys/tests in flight are skipped; a test that failed 3 attempts is parked.
3. **Launch** (`run.launch`): `git worktree add --detach /home/user/haiku-cc-wt/wNNNNN <main>`,
   then one `headless_fanout.py run` per worker with `--cwd` = that worktree, tools
   `Read,Grep,Glob,Edit,Write,Bash`, allow rules `Edit(//<worktree>/**)`, `Edit(//tmp/**)`,
   `Bash` (bare `Bash` verified to work under dontAsk), system prompt = `worker_prompt.md`.
   The task prompt lists 1–3 representative tests, other cluster members and notes from
   earlier failed attempts. The result goes to `/home/user/haiku-cc-wt/_results/<id>.json`
   and must satisfy `worker_result.schema.json`:
   `{status: fixed|partial|failed, test_ids_fixed, files_changed, root_cause, notes}`.
4. **Reap**: cost/turns/denials come from the runner's `ledger.jsonl`. Only `fixed` goes to
   the gate; `partial`/`failed` are recorded (their notes feed the next attempt).
5. **Gate** (`merge.merge_batch`, serial): diff the worktree vs its base (only `hcc/` and
   `docs/notes/`; any `tests/` or `swarm/` edit rejects), `git apply --3way` onto main in
   `/home/user/haiku-cc-wt/_gate`, commit (subject = root_cause, author Oskar Austegard).
   Up to `--batch` candidates are stacked and tested once: smoke + each candidate's targets
   (≤20 cluster members) + a fixed sample of known-passing tests (seeded by the baseline
   commit). Requires zero regressions vs `known.json` (regressions get one low-load
   re-run to filter timeouts) and ≥1 target newly PASS per candidate. A regressing batch is
   re-gated one by one. Then `git merge --ff-only` on main; if main moved, re-gate.
   Rejected diffs go to `swarm/rejected/<id>.diff` with the reason.
6. **Refresh** every `--full-every` merges: full suites; tests that were PASS in
   `known.json` but fail are bisected over swarm merge commits since the last baseline
   (only the regressed tests are run per step) and the culprit is `git revert`ed (its
   ledger row becomes `reverted`, its cluster is re-queued). Up to 3 culprits per refresh.

## State (`swarm/state/`, gitignored — delete it to start over)
| file | content |
|---|---|
| `ledger.jsonl` | append-only worker snapshots; last line per id wins. Fields: id, key, signature, tests, targets, base, status (launched→finished→merged/rejected, failed, reverted), cost_usd, turns, denials, result, merged_commit, reason |
| `known.json` | `{test_id: {status, reason, stage, commit}}` — what the gate compares against |
| `results/<commit>.json` | every result observed at that main commit |
| `baseline.json`, `history.jsonl` | last full run; pass counts per suite over time |
| `merges.jsonl`, `reverts.jsonl`, `counters.json` | merge order (for bisect), reverts, merge counters |
| `tasks/<id>/` | task.jsonl, runner.log, fanout/ (runner ledger + raw result), patch.diff |

Resume: just rerun `run`. Live workers from the previous process are adopted by pid,
dead ones are reaped, finished ones are merged.

## Modules
`common.py` (Ctx paths, git helpers, atomic JSON, ledger) · `testrun.py` (tests/run.py
adapter, results cache, sample) · `queue.py` · `merge.py` · `run.py` · `fake_worker.py`
(modes fix/noop/break/tests) · `_boot.py` (stops `swarm/queue.py` shadowing stdlib `queue`).

## Tuning notes
CPU (4 vCPU) is the bottleneck: workers run only their targets (`--jobs 1`) and smoke
(`--jobs 2`). If the gate falls behind (finished workers pile up), raise `--batch`, lower
`--sample`, or lower `--concurrency`. Watch `status` for rising `denials` or one dominant
failure reason: that is a prompt/permission bug, not bad luck — stop and fix.
