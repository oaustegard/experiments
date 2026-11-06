// The prompt the page offers for preparing a dataset with an AI agent. `decision` is appended when the user
// has drafted questions, so the agent refines it instead of starting over. The numbers are this experiment's
// (RESULTS.md: invoice workflow, library start, LoRA on the Mac): 0.629 at 16 cases, 0.727 at 64, 0.837 at 300.
export function agentPrompt(decision = null) {
  const draft = decision && Object.keys(decision.questions ?? {}).length
    ? `\n\nHere is the decision I have drafted so far. Refine it rather than starting over:\n\n${JSON.stringify(decision, null, 1)}\n`
    : "";
  return `You are helping me build a labelled dataset for a small decision model. The model (Ettin-17m, a 17-million-parameter text encoder) reads one case as text and answers a fixed set of questions about it, each by picking one option from a fixed list. I will train it in a browser page, which imports one JSON file in the format at the end. Work with me in the steps below and check with me before each next step.

1. The decision. Ask me what the cases are and what I want decided about each one. Turn that into 1 to 6 questions. Each question has a type:
   - "choice": one option from a named set, with 2 to 8 options.
   - "noul": a statement that is either true or false for the case.
   - "score": an ordered scale, given as a list of level descriptions from lowest to highest (the levels are named "0", "1", ...).
   Write "instructions" as the question itself. Give every option a one-sentence description of when it applies: the model reads these descriptions, so they carry meaning. Options must not overlap, and together they must cover every case; add an "other" or "unclear" option if real cases fall outside the rest.

2. The cases. A case's "state" is the text the model reads: the email, the ticket, the record, as text or JSON. Use real cases from me wherever possible. Generate synthetic cases only if I agree, and then make them look like the real ones: same fields, lengths, noise and mistakes. Keep each state under about 3,000 characters, because the model reads only about 900 tokens and cuts the rest; put the facts the answers depend on near the start. Never put the answer in the state (no "label" or "status: approved" field that gives it away).

3. The labels. Each case gets a "gold" answer for each question: an option name, or, where reasonable people would disagree, a distribution such as {"approve": 0.7, "hold": 0.3}. Before labelling, write down a one-line rule per question and apply it the same way every time. Flag the cases you are unsure about and ask me instead of guessing.

4. How many. 16 labelled cases teach the model little, 64 are a usable start, and 300 come close to what it can learn. On an invoice-review task it got 63% of decisions right from 16 cases, 73% from 64 and 84% from 300. Aim for at least 5 cases of every option of every question, and keep the mix close to what I will see in practice. Mark about 20% of the cases with "split": "test". The page never trains on them, so they give the honest score. Keep near-duplicates on the same side of the split.

5. Check before you hand the file over: every question has instructions and at least 2 options (a noul question may leave out its options); every gold value names an existing option, or is a distribution over existing options that sums to 1; every case id is unique and no state is empty. Then report how many cases each option of each question has, and flag any option with fewer than 5.

The output is one JSON object and nothing else:
{"decision": {"id": "snake_case_name", "description": "One sentence: what a case is and what gets decided.",
  "questions": {
    "category": {"type": "choice", "instructions": "Which kind of request is this?", "criteria": {"refund": "The customer wants money back.", "other": "Anything else."}},
    "urgent": {"type": "noul", "instructions": "The customer needs an answer today.", "criteria": {"true": "A deadline today or sooner is stated or implied.", "false": "No same-day deadline."}},
    "severity": {"type": "score", "instructions": "How serious is the problem?", "criteria": ["Cosmetic.", "Inconvenient.", "Blocks the customer's work."]}}},
 "cases": [{"id": "c001", "state": "...", "gold": {"category": "refund", "urgent": "false", "severity": "1"}},
           {"id": "c002", "state": "...", "gold": {"category": {"refund": 0.6, "other": 0.4}, "urgent": "true", "severity": "2"}, "split": "test"}]}
Leave "split" out for training cases. A score question's gold is its level as a string ("0", "1", ...), and a noul question's gold is "true" or "false".${draft}`;
}
