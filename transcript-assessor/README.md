# transcript-assessor

One Claude Haiku 5.5 call per transcript returns four assessments: compliance and security concerns, a multi-label task classification, the friction the person hit, and what the session achieved with an estimate of the manual hours it replaced. Code does everything a model does not need to: parsing, secret and PII scanning, timing, redaction, arithmetic and the review flag. Tested on 241 Claude Code sessions from the claude-workspace transcript archive. On a 16-session sample Haiku agreed with Opus 5.5 on the primary task label for 15 of 16 sessions and on the review flag for 14 of 16. Its manual-hours ranking matched Opus at Spearman 0.99. The whole corpus would cost about $0.24 through the Batches API.

Oskar: *"What I want to build is a number of classifiers and assessments/rankers that could capture a number of items from a transcript ... I want to make use of Haiku 5.5 for token economics - we would probably run it in a workflow, processing in batch, overnight. But let's test it here first"*

## Pipeline

```
source transcript ─► normalized events ─► signals (code) ─┐
                                        └► digest (code) ──┴─► Haiku 5.5, structured output ─► finalize (code)
```

1. **Load.** `load_events` reads three formats into one event list (role, kind, text, tool, error flag, timestamp, typed-by-a-person flag): Claude Code session JSONL, a Compliance API session transcript (`/v1/compliance/apps/sessions/{local,remote}/{id}/messages`), and a Compliance API chat (`/v1/compliance/apps/chats/{id}/messages`). All three are role plus `text` / `tool_use` / `tool_result` blocks, so one adapter per envelope is enough. The two Compliance API adapters were tested on the example responses in Anthropic's documentation, not on a live export.
2. **Signals.** Regexes for secrets (Anthropic, OpenAI, GitHub, AWS, Slack and Google keys, private keys, JWTs, bearer headers, `ENV_TOKEN=value` assignments) and PII (US SSNs, Luhn-valid card numbers in card-shaped formats), risky shell commands (`rm -rf`, force-push, `reset --hard`, `--no-verify`, `DROP TABLE`, `curl | sh`, `chmod 777`), tool-error rate, interruptions, permission denials, compactions, wall-clock minutes and an attention estimate: for each message the person typed, the gap since the previous event, capped at 10 minutes.
3. **Digest.** Every typed message in full (clipped at 4,000 characters), Claude's visible replies clipped at 900, each tool call as one line, failed tool results, and `[secret-scan]` and `[risky-cmd]` lines that show where each hit sits, even inside a successful tool result the digest would otherwise drop. Secrets are redacted before clipping, so a clip boundary cannot cut through a key the patterns would have caught. Over the budget, routine tool-call lines are dropped from the middle out. 395 MB of JSONL became 3.0 MB of digest, about 1.0M tokens for all 241 sessions. The median digest is about 1,900 tokens and the largest about 42,000, well under Haiku 5.5's 100K-token price break.
4. **Model.** `prompt.md` is the system prompt and holds the taxonomies, definitions, calibration rules and effort anchors. `SCHEMA` in `assess.py` is passed as `output_config.format`, so the API guarantees the shape. Effort is `low` and the system prompt is cached.
5. **Finalize.** Code sums the per-deliverable hour ranges, subtracts the person's attention time to get hours saved, and sets `flag_for_review` when the model's severity is `medium` or `high` or when the regex scan found a secret or PII.

## Taxonomies

| dimension | labels |
|---|---|
| task (multi-label + primary) | software_feature, bug_fix_debugging, refactor_migration, code_review, testing_qa, devops_infra, data_analysis, ml_ai_engineering, agent_prompt_tooling, automation_scripting, research_investigation, writing_documentation, communication_drafting, creative_content, planning_strategy, learning_explanation, admin_configuration, other; plus free-text `domain` and `autonomy` (interactive / delegated / unattended) |
| compliance | credential_exposure, sensitive_personal_data, confidential_data_egress, security_control_bypass, destructive_or_unreviewed_change, offensive_security, prompt_injection, legal_regulatory_hr, unapproved_service_or_account, other; each with severity, actor (user_request / assistant_action / third_party_content), turn, evidence and whether the session mitigated it |
| friction | misunderstood_request, wrong_approach, buggy_output, tool_or_env_failure, permission_or_access_block, repeated_correction, overreach_or_scope_creep, unverified_or_false_claim, slow_or_stalled, context_loss, format_or_verbosity, user_interrupted, other; each with turn and cost, plus a session level and end sentiment |
| success | outcome (achieved … not_achieved, unclear); deliverables, each with evidence turn, role and a low–high range of manual hours; confidence and rationale |

## Test

Sixteen sessions were chosen to span the archive: two unattended scheduled runs, one of them with a credential in its tool output, a one-question lookup, a 23-turn ML session and long build sessions with force-pushes and permission denials. Haiku 5.5 and Opus 5.5 each assessed all sixteen from identical prompt files, as subagents told to read the file and write the JSON. No API key exists in this environment, so this tested the prompt and digest, not the production request: the subagents ran without structured-output enforcement and with Claude Code's own system prompt around them. Opus is the reference here; nobody has hand-labelled these sessions.

Agreement of Haiku with Opus:

| measure | prompt v1 (n=13 valid pairs) | prompt v2 (n=16) |
|---|---|---|
| primary task label | 0.85 | 0.94 |
| task label set, mean Jaccard | 0.71 | 0.72 |
| review flag (severity ≥ medium or regex hit) | 0.77 | 0.88 |
| compliance max severity, exact | 0.69 | 0.56 |
| friction level, exact | 0.38 | 0.88 |
| friction level, within one step | 1.00 | 1.00 |
| outcome, exact | 0.69 | 0.69 |
| manual-hours rank (Spearman) | 0.98 | 0.99 |
| manual hours, total | 319.5 vs 281.2 | 297.1 vs 277.4 |

Haiku returned schema-valid JSON 16 of 16 times in each round. Opus left out a required field or added one the schema does not allow in 3 of 16 v1 runs; with structured outputs on the API neither failure is possible.

In v1, Haiku rated session friction higher than Opus in 6 of 13 sessions and lower in 2, and it reported compliance issues it could not see, such as a force-push known only from the signal count, or a pull request opened against the person's stated preference. Prompt v2 made three changes. Risky commands appear in the digest with the command text. The friction cost levels have operational definitions: `moderate` means the person had to step in. And the prompt states that a safeguard which fired and held is friction, not a compliance issue; it becomes a bypass only when the same action is then taken by another route. Friction agreement went from 0.38 to 0.88, with one session rated higher and one lower.

Compliance exact-severity agreement fell to 0.56 in v2 because Haiku still adds `low` items where Opus reports none. `low` does not set the review flag. Both review-flag disagreements are Haiku at `medium` where Opus said `low`, and in both the transcript shows content published to a public place. Every session Opus flagged, Haiku flagged.

Both models raised `medium` issues on real events in this archive, including a credential printed into tool output, which the regex scan also caught. The details stay out of this public writeup.

## Run over one week of sessions

Every session archived between 2026-10-01 and 2026-10-08 went through the pipeline: 69 sessions (26 interactive or delegated, 43 unattended scheduled runs), about 286K digest tokens from 167 MB of JSONL. Haiku 5.5 ran as subagents, as in the test, with prompt v2 unchanged. All 69 outputs were schema-valid, and the whole run took under ten minutes of wall-clock at 20 subagents at a time. Through the Batches API the same run would cost about $0.07.

| | interactive / delegated | unattended |
|---|---|---|
| outcome achieved / mostly / partial or unclear | 13 / 9 / 4 | 34 / 8 / 1 |
| friction none / low / moderate / high | 6 / 5 / 13 / 2 | 25 / 16 / 2 / 0 |
| manual-hours equivalent, median per session | 7.6 | 0.35 |

One session was flagged for review, at `medium`, and twelve `low` items in nine sessions stayed off the queue. The regex scan found no secrets or PII in the week. The top three interactive sessions account for 49% of the week's estimated manual hours, all of them large website or animation builds, so a total of hours saved is driven by a handful of estimates nobody has checked. The most common moderate-or-worse friction was a permission or access block (6), followed by tool or environment failures (5), misunderstood requests (4) and unverified claims (4). Per-session results stay out of this repository.

## Cost

Haiku 5.5 costs $0.10 per million input tokens and $0.50 per million output tokens for prompts up to 100K tokens; batch is half price; cached prompt reads are a tenth. With about 1.0M digest tokens, a 4K-token cached system prompt and about 3K output tokens per transcript including thinking, the 241 sessions come to about $0.47 synchronous, or $0.24 through the Batches API. That is about $0.001 per transcript. A nightly run over 10,000 enterprise transcripts of this size distribution would cost about $10. The output tokens are an estimate; check them against `usage` on the first real batch.

## Running it

```bash
python3 assess.py digest 'exports/*.json' --out digests/       # Compliance API JSON or Claude Code JSONL
python3 assess.py batch-submit digests/                         # prints the batch id; needs ANTHROPIC_API_KEY
python3 assess.py batch-collect <batch_id> --digests digests/ --out results.jsonl
python3 assess.py report results.jsonl
python3 assess.py compare results.jsonl reference.jsonl         # agreement against a labelled or stronger-model run
```

`batch-collect` lists every refusal, truncation or failed request on stderr instead of writing a partial record. Haiku 5.5 has no server-side refusal fallback, so a transcript about security work can be declined; rerun those on `claude-sonnet-5-5`. `emit-prompts` and `merge` are the no-API-key path used for the test above.

For an overnight job against the Compliance API: list chats with `order_by=updated_at` and keep the last cursor, so each run picks up new and changed chats; list sessions the same way. Fetch each transcript with `tool_use_input_max_bytes` and `tool_result_max_bytes` left at their 10,000-byte defaults, which is already more than the digest keeps. Skip chats with `deleted_at` set, and key results by chat or session id so a reappearing chat overwrites its old record.

## Limits

- **Hours saved has no ground truth.** Opus and Haiku agree with each other at rank 0.99, which shows they read the anchors the same way, not that the anchors are right. They put a website redesign at 95 and 81 manual hours. Anthropic's productivity research uses the same counterfactual method (a model estimates task time without AI) and notes it leaves out the time people spend checking the output. Before reporting hours to anyone, have the owners of 30–50 sessions estimate their own counterfactual hours and fit a correction.
- **The reference is a model.** Opus agreement measures consistency between models. Compliance needs a human-labelled set, weighted toward positives, before the review queue's recall can be stated.
- **Personal corpus.** One person, mostly agent tooling and research. Enterprise transcripts will shift the task distribution and add claude.ai chats with file attachments, which the digest records by file name only.
- **The Batches API path has not run.** `batch-submit` and `batch-collect` follow the SDK's documented request and result shapes; the first real batch is their test.
- **Results hold transcript text.** `evidence` and `summary` quote the transcript after redaction. Store results under the same access controls as the transcripts.

## Files

- `assess.py`: adapters, signals, digest, schema, Batches API submit and collect, merge, report, compare.
- `prompt.md`: the system prompt (v2).

Per-session results stay out of this public repo; the test outputs were kept in the session scratchpad.
