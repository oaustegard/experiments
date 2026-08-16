// Retry-round workflow for orchestrated-coding-pareto.
// Invoked with args = { round: 1|2, retry: ["task", ...], orch: ["task", ...] }
//   retry: tasks still failing in the haiku-retry arm (test-feedback only)
//   orch:  tasks still failing in the orch-haiku arm (opus diagnosis -> haiku fix)
// Reads prior code from data/solutions/<arm>/<task>.py and pytest output from
// data/failures/<arm>/<task>.txt (written by grade step between rounds).
export const meta = {
  name: 'ocp-retry-round',
  description: 'Retry round: haiku+test-feedback control vs opus-orchestrated haiku',
  phases: [
    { title: 'retry-feedback', detail: 'haiku retries with raw pytest output', model: 'haiku' },
    { title: 'orch-diagnose', detail: 'opus reviews failure, writes guidance', model: 'opus' },
    { title: 'orch-fix', detail: 'haiku retries with opus guidance', model: 'haiku' },
  ],
}

const BASE = '/workspace/experiments/orchestrated-coding-pareto'
const SCHEMA = {
  type: 'object',
  properties: { path: { type: 'string' }, code_chars: { type: 'number' } },
  required: ['path', 'code_chars'],
  additionalProperties: false,
}
const GUIDE_SCHEMA = {
  type: 'object',
  properties: { path: { type: 'string' }, root_causes: { type: 'number' } },
  required: ['path', 'root_causes'],
  additionalProperties: false,
}

const round = args.round
const marks = { start: budget.spent() }

function fixPrompt(task, arm, guidancePath) {
  const spec = `${BASE}/tasks/${task}/spec.md`
  const prev = `${BASE}/data/solutions/${arm}/${task}.py`
  const fail = `${BASE}/data/failures/${arm}/${task}.txt`
  const lines = [
    'Your previous implementation of a small Python module failed its hidden test suite. Fix it.',
    `1. Read these files and nothing else: the spec ${spec}, your previous attempt ${prev}, the pytest failure output ${fail}` + (guidancePath ? `, and a senior engineer's review of your failure: ${guidancePath}` : '') + '.',
    guidancePath
      ? '2. Follow the review guidance. Re-read the spec rules it points at. Produce a corrected, complete module.'
      : '2. Work out from the pytest output which spec rules your code violates and produce a corrected, complete module.',
    `3. Overwrite ${prev} with the corrected full module source using the Write tool.`,
    '4. Return structured output {path, code_chars}.',
    'Hard rules: only the listed Reads and that one Write. Do NOT run code or tests, do NOT use Bash. Standard library only.',
  ]
  return lines.join('\n')
}

function diagPrompt(task) {
  const spec = `${BASE}/tasks/${task}/spec.md`
  const prev = `${BASE}/data/solutions/orch-haiku/${task}.py`
  const fail = `${BASE}/data/failures/orch-haiku/${task}.txt`
  const out = `${BASE}/data/guidance/round${round}/${task}.md`
  return [
    'You are a senior engineer reviewing a junior developer\'s failing solution to a precisely-specified task.',
    `Read these files and nothing else: the spec ${spec}, the failing code ${prev}, the pytest failure output ${fail}.`,
    'Diagnose the ROOT CAUSES (not symptoms). For each: name the spec rule violated, point to the offending part of the code, and say concretely what a correct approach does. Be specific and complete - the junior will see your note, the spec, the code and the raw test output, and gets ONE shot.',
    'Do NOT write the corrected module yourself. Snippets of at most 3 lines are allowed to pin down an exact behavior.',
    `Write your review as markdown to ${out} using the Write tool, then return structured output {path, root_causes: <count>}.`,
    'Hard rules: only the listed Reads and that one Write. Do not run code or tests, do not use Bash.',
  ].join('\n')
}

// Phase 1: control arm - haiku with raw test feedback
phase('retry-feedback')
if (args.retry.length) {
  await parallel(args.retry.map(t => () =>
    agent(fixPrompt(t, 'haiku-retry', null),
      { label: `retry:${t}`, phase: 'retry-feedback', model: 'haiku', effort: 'medium', schema: SCHEMA })))
}
marks.retry_feedback = budget.spent()
log(`retry-feedback done; cumulative ${marks.retry_feedback}`)

// Phase 2: opus diagnoses each orch-arm failure
phase('orch-diagnose')
if (args.orch.length) {
  await parallel(args.orch.map(t => () =>
    agent(diagPrompt(t),
      { label: `diag:${t}`, phase: 'orch-diagnose', model: 'opus', effort: 'high', schema: GUIDE_SCHEMA })))
}
marks.orch_diagnose = budget.spent()
log(`orch-diagnose done; cumulative ${marks.orch_diagnose}`)

// Phase 3: haiku fixes with the guidance
phase('orch-fix')
if (args.orch.length) {
  await parallel(args.orch.map(t => () =>
    agent(fixPrompt(t, 'orch-haiku', `${BASE}/data/guidance/round${round}/${t}.md`),
      { label: `fix:${t}`, phase: 'orch-fix', model: 'haiku', effort: 'medium', schema: SCHEMA })))
}
marks.orch_fix = budget.spent()
log(`orch-fix done; cumulative ${marks.orch_fix}`)

return { round, marks, n_retry: args.retry.length, n_orch: args.orch.length }
