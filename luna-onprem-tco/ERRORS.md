# Error log — luna-onprem-tco

Every error in this experiment, how it was caught, and which way it pushed the
conclusion.

**Direction is the column to read.** Four errors. Three flattered self-hosting;
one flattered the electricity line, which is the thing the experiment was
*asked* about. The base rate matters here more than usual because this is a
procurement-shaped analysis with no expensive artifact to check it — nothing
crashes when a cost model is wrong, it just produces a confident number.

| # | error | caught by | direction |
|---|---|---|---|
| 1 | Took the requester's `$0.22/M input` as OpenAI's direct price. It is **AWS Bedrock**. Direct is `$0.20`, and a **batch tier at `$0.10`** existed that I never asked about and never modelled. | the requester, unprompted | **flattered self-hosting.** 10% on the interactive path, 120% on the batch path — and the batch path was the *only* route where the box could saturate. Pass 1 said "run the overnight batch locally"; at the real price that recommendation was worth half what I claimed. |
| 2 | Pass 1 sized the analysis on a **GB200 NVL72 rack** — 72 GPUs, 132 kW, $3.5 M — for what turned out to be 800 office seats. A ~40× over-provision. | the requester supplying seat count and duty cycle | **flattered the energy result.** Rack decode is **5.8× more energy-efficient per output token** than the 8-GPU node a real 800-seat deployment would buy ($0.014 vs $0.083/M), because MoE decode scales with NVLink-domain size. Quoting rack-scale energy for a single-node deployment overstates it by ~6×. |
| 3 | `model.py:power_kw` applied the **serving power floor** (idle + 35% of the load delta) to genuinely **parked** hours. The box is parked ~75% of the year. | recomputing in `recheck.py` and comparing against the throwaway script written during the conversation, which had it right | **inflated electricity by ~35%** (70.5 → 51.7 MWh/yr at scenario A). The one error pointing *away* from the conclusion: it made the electricity line look more important than it is, in the one experiment whose finding is that electricity is not important. Would have made the writeup *weaker*, not stronger — which is why it survived a read-through and only fell to arithmetic. |
| 4 | `model.py` returned **verdict: self-host** at 211% and 845% peak utilisation — comparing **one** node's cost against a token bill that needs three or nine. | the seat sweep, immediately after writing it; the 845% row was visibly absurd | **flattered self-hosting, badly.** Fixed structurally (`nodes_required = ceil(peak_utilisation)`, capex and batch capacity both scale) rather than as a caveat, because every instance of this class of error points the same way. `recheck.py` now asserts per-node peak ≤ 100% on every scenario. |

## The framing error, which is not in the table

Errors 1–4 are wrong numbers. The larger mistake was answering the literal
question well before establishing whether it was the deciding one. Pass 1
delivered a correct, sourced, carefully-bounded comparison of electricity
against token price — 63×, later 57× once error 1 was fixed — and that
comparison decides nothing, because electricity is 3% of the cost of owning
the hardware. Two turns of user-supplied constraints reframed it entirely.

I don't think the first pass should have been withheld; the requester asked a
specific question and got it answered. But it should have carried the sentence
that the second pass opens with, and it didn't: *this ratio is large and it is
not what the decision turns on.* The tell was available at the time — pass 1
computed "electricity is 15.7% of hardware TCO" and printed it as a footnote
under the headline it undercut.

## What did not go wrong

- The capability match held up. V4 Pro at AA index 53 vs Luna (max) 52 survived
  a deliberate attempt to find a cheaper open model that matched; V4 Flash was
  checked and rejected on three benchmarks rather than assumed.
- No number was carried from memory. `params.json` has a `source` on every row
  and `recheck.py` fails the build if one is missing.
- The prior-art checks were run before writing code, both came back clean, and
  the `ModuleNotFoundError` from `repo-index/ask.py` was treated as a missing
  dependency and installed — per METHODS.md — rather than as a broken check.
