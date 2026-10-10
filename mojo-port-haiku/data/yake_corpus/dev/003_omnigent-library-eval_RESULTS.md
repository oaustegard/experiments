# omnigent — library evaluation

**Question (from the session prompt):** "Is this anything you could make use
of?" — [omnigent-ai/omnigent](https://github.com/omnigent-ai/omnigent).

**Verdict up front:** Not as a *dependency* for claude-workspace. Omnigent is a
competing **meta-harness** — a server + CLI that wants to *own* agent
orchestration, model routing, and sandboxing. claude-workspace already gets all
three from Claude Code on the Web (CCotw), so adopting omnigent here would mean
running a parallel harness, not importing a library. It also can't run in this
container (no model API key by design; wants its own server, Node 22, tmux,
bwrap). **But** three things are worth borrowing — and one of them is free,
because omnigent reads the *same* `SKILL.md` format Muninn already uses.

Ground-truthed 2026-06-29 against a fresh tarball of `main` (`.spokes/omnigent`,
not committed — gitignored spoke clone).

---

## What it actually is

| Fact | Value |
|---|---|
| Repo | `omnigent-ai/omnigent`, Apache-2.0 |
| Pitch | "open-source meta-harness for all your AI agents" |
| Stars / forks | 5,308 / 674 |
| Age | created 2026-06-11 (**18 days old** at eval), pushed hours before eval |
| Maturity | **alpha**, PyPI `omnigent 0.3.0.dev0`, **379 open issues** |
| Language | Python ≥3.12 (60 MB repo, ~900 KB `uv.lock`, 390 KB `openapi.json`) |
| Shape | a **server** (`omnigent server`, localhost:6767) + CLI (`omnigent`/`omni`) + web UI + macOS desktop app; the `omnigent-client` Python SDK is a *client to that server*, **not** an importable orchestration lib |

**The one-line value prop:** a common orchestration layer over 11 agent vendor
harnesses — `antigravity codex claude cursor goose hermes kimi kiro opencode pi
qwen` — so you can mix/swap them in one session, govern them with policies,
sandbox them, and drive them from terminal / browser / phone / desktop.

Four pillars:

1. **Multi-vendor sub-agents.** One session can run Claude Code + Codex + Cursor
   + Pi together; ask one to review another's work. The shipped `examples/polly`
   (orchestrator) and `examples/debby` (debate) agents demonstrate it.
2. **Declarative YAML agent specs.** `omnigent run agent.yaml` — `executor`
   (harness/model/auth), `tools`, `policies`, `params`, `os_env`, `terminals`.
   (`docs/AGENT_YAML_SPEC.md`.)
3. **Policy engine.** ALLOW / DENY / ASK gates at three levels (server / agent /
   session), composed in order, DENY short-circuits. Builtins like
   `blast_radius`, `spawn_bounds`, `cost_budget`. (`docs/POLICIES.md`.)
4. **Sandboxing + cloud hosts.** `bwrap` (Linux) / `seatbelt` (macOS) + an L7
   egress proxy; disposable sessions on Modal / Daytona / E2B / Kubernetes /
   Databricks / etc.

---

## Fit against claude-workspace / Muninn

claude-workspace **is itself a bespoke single-harness setup** (CCotw = Claude
Code on the Web): layer-composed container, live skills, Turso memory, spoke
orchestration via the GitHub API, boot identity. So the comparison is harness
vs. harness, and most of omnigent's pillars **duplicate what CCotw already
provides**, not complement it:

| omnigent pillar | claude-workspace today | Overlap? |
|---|---|---|
| Multi-device (terminal/browser/phone/desktop) | CCotw runs from claude.ai web/mobile/desktop already | **Duplicate** |
| Cloud sandbox per session | CCotw *is* an ephemeral cloud container | **Duplicate** |
| Model routing / auth | Inference routes through the harness Agent tool; `ANTHROPIC_API_KEY` absent by design | **Conflict** — omnigent wants to own routing |
| Policy engine (ALLOW/DENY/ASK) | `.claude/settings.json` hook permissions + TDD hook | Partial — same goal, omnigent's is more structured |
| Cross-vendor **review** | already available — `gemini_generate()` via the Cloudflare AI Gateway (any provider the gateway fronts) | **Duplicate** — review is text-in/verdict-out, needs a different-*model*, not a tool-using agent |
| Cross-vendor **implementer agents** (Codex/Cursor/Pi as full coding harnesses in worktrees) | harness Agent tool sub-agents are all Claude | omnigent adds this — the one real gap, and a narrow one |
| Skills (`SKILL.md`) | `/mnt/skills/user/` | **Compatible** (see below) |

### Why it can't just run here
- **No model key.** CCotw deliberately omits `ANTHROPIC_API_KEY`; omnigent
  expects a first-party key, a CLI login, or a gateway to drive harnesses.
- **Heavy host deps.** Python 3.12+, `uv`, Node 22, `tmux`, `bubblewrap` — and
  it wants to *be* a running server. That's a parallel harness inside a harness,
  not a layer to compose.
- **Alpha + churning.** `0.3.0.dev0`, 18 days old, 379 open issues, pushed
  hourly. Wrong stability profile for a hub dependency.

---

## What's genuinely worth taking

Three borrowable things, in descending order of value:

### 1. Skill-format compatibility is free
`omnigent/spec/skill_sources.py` defines `_SKILL_FAMILIES = {claude, codex,
cursor, pi}` and discovers host skills in the **same `SKILL.md` format**
Muninn's skills already use (it maps `claude-sdk` / `claude-native` →
`claude` family and reads `~/.claude/skills`). **Muninn's entire skill library
would load into an omnigent agent with zero porting.** This is the lowest-cost
bridge if Muninn is ever run *under* omnigent (outside CCotw).

### 2. The Polly orchestration *pattern* (not the cross-vendor capability)
`examples/polly` is a coding orchestrator that writes no code itself: it
decomposes a task, dispatches `claude_code` / `codex` / `pi` sub-agents in
parallel git worktrees, then routes each diff to a reviewer **from a different
vendor than the implementer** (`claude → codex|pi`, `codex → claude|pi`). The
human merges.

**Correction (the original draft overclaimed this).** Independent
cross-vendor *review* is **already available to Muninn today**, no omnigent
required: review is a text-in / verdict-out task — hand the reviewer "diff +
contract," get a structured report — so it needs a different-vendor *model*,
not a tool-using agent. Muninn reaches Gemini (and any other provider the
gateway fronts) right now via `gemini_generate()` over the **Cloudflare AI
Gateway** (`experiments/phase-a-bridges/scripts/common.py`:
`gateway.ai.cloudflare.com/.../google-ai-studio/...`, already driving
`gemini-3.5-flash` with json-mode + thinking budgets). So the Polly
**cross-vendor-review discipline is directly runnable in CCotw today** — a
Claude implementer, a Gemini reviewer handed the diff.

The genuinely-residual gap omnigent fills is narrower: cross-vendor
*implementer agents* — running Codex / Cursor / Pi as **full tool-using coding
harnesses in their own worktrees**. *That* Muninn doesn't have (its agentic
sub-agents are all Claude via the Agent tool). It's a real but small and
debatable win — a different-vendor model reviewing a diff catches most of what
cross-vendor buys; a different-vendor *implementer* mainly diversifies how the
code gets written.

Still worth reading `examples/polly/config.yaml` +
`skills/cross-review/SKILL.md` as a reference for multi-agent orchestration
prompts — they're unusually careful about dropped turns and busy-poll
avoidance, and the diff+contract-only / never-point-the-reviewer-at-the-worktree
review recipe is portable as-is.

### 3. Declarative policies as a reference design
If Muninn ever formalizes its ad-hoc `settings.json` permission allowlist into
real governance, omnigent's ALLOW/DENY/ASK three-level model (`docs/POLICIES.md`)
and the function-policy form (`blast_radius` with `gate_pushes`, `spawn_bounds`
with `max_dispatches_per_turn`, `cost_budget` with `ask_thresholds_usd`) are a
clean prior-art template. Not adoptable as code (it's coupled to omnigent's
runner), but a good design to copy.

---

## Recommendation

- **Do not add omnigent as a claude-workspace dependency or container layer.**
  It overlaps CCotw on multi-device/sandbox/routing, conflicts on model auth,
  and is alpha-churning. There is no "import omnigent" win — the Python package
  is a *client to a server you'd have to run*.
- **Do keep it on the radar as a place to run Muninn-the-agent *outside*
  CCotw — but only for cross-vendor *implementer* agents.** Cross-vendor
  *review* Muninn can already do today (Gemini reviewer via the CF AI Gateway).
  The only thing omnigent uniquely adds is running Codex/Cursor/Pi as full
  agentic coding harnesses — and Muninn's skills would port to it for free.
- **Cheapest next step if curious (not runnable from this session):** author a
  `muninn.yaml` agent spec (`executor.harness: claude-sdk`, point `tools` at the
  existing skills) and `omnigent run` it on a host *with* a model key. Blocked
  here by design (no key, no local omnigent server) — would be a local-machine
  or non-CCotw experiment.

**One-liner:** A strong, fast-moving competitor to the CCotw harness — not a
library to absorb. Borrow the SKILL.md compatibility (free), the Polly
orchestration prompts (the cross-vendor-*review* discipline Muninn can already
run via the CF Gemini path), and the policy design (reference). The only thing
omnigent uniquely adds is cross-vendor *implementer* agents. Don't take the
dependency.

---

*Eval method: GitHub API metadata + a `main` tarball into `.spokes/omnigent`,
read `README.md`, `docs/{AGENT_YAML_SPEC,POLICIES}.md`, `examples/polly/`,
`omnigent/spec/skill_sources.py`, `omnigent/sandbox/`, `pyproject.toml`,
`sdks/python-client/`. No code executed (alpha server, no model key in CCotw).*
