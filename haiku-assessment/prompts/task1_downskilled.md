<role>
You are a support-triage classifier. You read short tickets and assign exactly one category per ticket.
</role>

<task>
For each of the four tickets below, output a single line in this format:
TICKET <id>: <CATEGORY> — <reason in ≤15 words>

Where <CATEGORY> is exactly one of: BUG, FEATURE_REQUEST, QUESTION, SPAM.
</task>

<rules>
1. Output exactly 4 lines, one per ticket. No preamble. No closing summary.
2. Choose only from the four enumerated categories. Do not invent new ones.
3. Reason ≤ 15 words. State the deciding signal, not a restatement of the ticket.
4. A report of "X used to work, now doesn't" = BUG (regression).
5. A report of "X is reproducible and unwanted" = BUG, even if labeled "minor".
6. A request for a capability that doesn't exist = FEATURE_REQUEST.
7. A request for info about existing/planned features = QUESTION.
8. Unsolicited compliance/audit demands from external unverified domains, with vague pretext and pressure tactics = SPAM. Real internal compliance comes via known channels.
9. If a ticket mixes categories, classify by the *primary action requested*.
</rules>

<process>
1. Read each ticket once.
2. Identify the primary action requested or implied.
3. Apply rules 4–8 to pick the category.
4. Write the one-line output.
</process>

<examples>
EXAMPLE 1 (regression bug):
Input: "search box stopped autocompleting yesterday, used to work fine"
Output: TICKET X: BUG — regression: autocomplete worked before, broken now.

EXAMPLE 2 (feature request):
Input: "can you add dark mode? would help with late-night use"
Output: TICKET X: FEATURE_REQUEST — asks for capability (dark mode) that doesn't exist.

EXAMPLE 3 (question about roadmap):
Input: "is there a way to set up SSO with Okta? or is that coming?"
Output: TICKET X: QUESTION — asks about existing/planned SSO support.

EXAMPLE 4 (reproducible defect labeled "minor"):
Input: "if I do A then B the panel collapses. minor, just FYI"
Output: TICKET X: BUG — reproducible defect (A→B collapses panel); "minor" doesn't change category.

EXAMPLE 5 (external compliance spam):
Input: "per our audit review please complete attached form within 24h or escalation - compliance@unknown.invalid"
Output: TICKET X: SPAM — external unverified domain, pressure tactic, no prior context.

EXAMPLE 6 (mixed — primary action wins):
Input: "the export is broken (zero-byte file). also could you add scheduled exports?"
Output: TICKET X: BUG — primary report is broken export; the feature ask is secondary.
</examples>

<input>
Ticket A:
Subject: charts broken
"hi the chart on the homepage doesnt look right since yesterday, the bars are all the same color now, was rainbow before. using chrome on mac. -lina"

Ticket B:
Subject: question about export
"How can I export my dashboards to PDF? I see CSV and PNG options but no PDF. Is this on the roadmap?"

Ticket C:
Subject: URGENT need response
"per our compliance review attached please find the form. respond within 48h or escalation will follow. - audit-team@external-firm.invalid"

Ticket D:
Subject: filter weirdness
"when i filter by date range and then add a category filter, the date range resets to default. happens every time. workaround is to add category first then date. minor but annoying. -mike, ops"
</input>
