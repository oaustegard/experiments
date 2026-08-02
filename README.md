# Experiments

Random vibe-experiments — one-off explorations and data products. Each
subdirectory is self-contained: scripts, data, and a results file
(`RESULTS.md` for pipeline runs, `README.md` for build/recipe artifacts,
or the artifact itself when it speaks for itself, like an HTML page).

**Trust conventions.** [`ANCHORS.md`](ANCHORS.md) registers every published
constant in use with its *covered range*, because a range gap is invisible from
inside a green run. Each experiment should carry an `ERRORS.md` (what was wrong,
how it was caught, which direction it pushed the conclusion — the base rate is
the most useful calibration number about a body of work) and a `recheck.py`
(a sub-5-minute fixture that checks the prose against the artifacts, so the
writeup and the data cannot drift apart between full rebuilds).
`remex-vs-higgs-ablation/` carries all three and is the reference shape.

Results are reported as they came out. Several of these are negative
results, one is a correction of earlier rejected work, and one turned
out to reproduce a 2004 paper rather than extend it — all labelled as
such rather than quietly dropped.

Migrated from `oaustegard/claude-workspace/experiments/`, which is a
session-boot repo and was the wrong home for 37 research projects.

## Index

| Experiment | Started | Status | Results | Origin |
|---|---|---|---|---|
| [`remex-vs-higgs-ablation/`](remex-vs-higgs-ablation/RESULTS.md) | 2026-08-02 | **done — mixed; 2 of 4 pre-registered predictions failed; rerun under `gating`** | [`RESULTS.md`](remex-vs-higgs-ablation/RESULTS.md) + `tables.md` + `gate.log` + `audit.log` + `mutate.log` + `verify_kills.log` + `axes.png` / `marginals.png` / `seeds.png` | Issue [#8](https://github.com/oaustegard/experiments/issues/8): does **remex** (exact fp32 norm + dense Haar rotation + scalar Lloyd-Max) buy anything for retrieval-index compression over the **QuIP# -> HIGGS -> TurboQuant** lineage (randomized Hadamard + per-block scale + Gaussian-MSE-optimal grid)? Full **2x2x2 factorial**, 11 arms x 6 bit widths x 3 corpora (d=100/768/1024) x 5 rotation seeds x 2 metrics, scored against fp32 exact search rather than qrels. **Only axis C moves.** Rotation is null (+0.0005 +/- 0.0016 recall@10) and norm handling is null (+0.0014); the codebook is an order of magnitude larger (+0.0113 cosine / +0.0146 IP), peaking at **+0.035 recall@10 at 2-3 bits** and decaying to zero by 8 bits, with the effect ~2x larger at d=100 than at d=768/1024. **Failed predictions:** (1) the RHT was predicted 10-100x *faster* at d=768-1024 and is measured **11-24x slower** -- numpy's strided FWHT loses to one BLAS sgemm, and the crossover is past d=8192; (2) exact-norm was predicted to win under inner product and does not, partly because BGE-family encoders are trained under cosine so their raw norms barely vary (CV 1.4-2.7% vs GloVe's 20.2%) -- axis B is close to moot on modern encoders. **The practical reversal:** counting the shared codebook, the vector arm costs 52.5 B/vector at 4 bits on a 20k-vector index against a 50 B payload, so remex at 6 bits (81 B true, R@10 0.965) beats HIGGS-like at 4 bits (112.5 B true, 0.893) on bytes *and* recall; the vector arm needs ~350k vectors to amortize. **Process:** the two-sided calibration gate caught Lloyd-from-random-init producing grids **worse than scalar** at 6-8 bits, and a scheduled adversarial review then found five more blocking defects -- a stale codebook served by a cache keyed on the problem rather than the method (8-bit vector arm 87% worse than scalar), a Lloyd-Max MSE identity evaluated off the fixed point (+16% at 8 bits, in the direction that makes the gate *more* permissive), a 'provably no worse' guarantee that was argued rather than enforced, a block/sub-vector divisibility bug hitting only the HIGGS-like arm, and a gate that never certified the grids behind any glove result. Scoring `q.xhat` without dividing by `||xhat||` also manufactured a fake 1-bit axis-C reversal. **Rerun under the `gating` skill (2026-08-02):** audited, the gate had three checks that *could not fail* -- the rotation check was purely relative, so replacing BOTH rotations with the identity passed; the axis-C check had no margin, so a vector arm with zero quantization gain cleared it by sampling noise; and the published anchor stops at 5 bits while the sweep runs to 8. Mutation testing (91 mutants, 55 survivors) found the gate scored codebook *contents* and never ran the *encoder*. Rebuilt: 139 checks, 5 known-bads, 14 stated coverage limits, and it now BLOCKS the sweep rather than being a documented step. Conclusions unchanged; glove100 and nfcorpus1024 reproduce to 4 decimals. |
| [`lattice-representation-hypothesis/`](lattice-representation-hypothesis/RESULTS.md) | 2026-07-31 | **done — negative result** (opening thesis refuted by the experiment's own adversarial + WordNet arms) | [`RESULTS.md`](lattice-representation-hypothesis/RESULTS.md) + [`THEORY.md`](lattice-representation-hypothesis/THEORY.md) + [`fca.py`](lattice-representation-hypothesis/src/fca.py) + `noise_reversal.png` | A *Paper Skygest* Bluesky post pointing at [arXiv:2603.01227, "The Lattice Representation Hypothesis of LLMs"](https://arxiv.org/abs/2603.01227) (LLM embedding geometry encodes an FCA concept lattice; meet/join as half-space operations). Hypothesis: the concept algebra has a **broken join** — FCA's meet extent is a bare intersection (exact under half-spaces) while its join extent is the *closure* of a union, and the paper's Definition 7 writes that join as a literal set union, contradicting its own Appendix B. Measured at scale: **0 meet phantoms across 9,615,370 concept pairs**, join overshoot mean 0.60. **Then the experiment's own arms killed it.** (i) Definition 7's "conic hull" clause makes the join *exactly* `R(Y_A n Y_B)` under linear independence — verified 96/96 by an independently written Minkowski-sum LP, strict only under conic dependence (43/96, 34/96). (ii) The "phantoms" are not errors: they *are* the least upper bound, and the gold label in the paper's own task (join of {dog, wolf} is `canine`, which contains foxes). (iii) **The premise was mathematically wrong**: `meet = objs_of(B1 u B2)` and `join = objs_of(B1 n B2)` are *both* plain half-space intersections (0 identity violations over 30 configs x 8 seeds); the join needs **fewer** constraints (0.83 vs 5.55). The only non-representable object is the plain set union, which isn't a lattice operation — and it is indeed recovered worst of the three. The one measured effect — join degrades more slowly than meet under probe error on Jaccard (5% flip: 0.939 vs 0.835) — carries a **12x target-size confound**, flips sign on symmetric-difference error in one of two contexts (cross: join 0.110 vs meet 0.028), and its size-controlled version is reproduced *more strongly by random-direction controls*; the proposed "closure absorbs noise" mechanism is contradicted by a **negative** overshoot-vs-error correlation (-0.44). **Surviving caveats, aimed at the paper's setting:** WordNet noun hypernymy gives an extremely thin lattice (**15 concepts** from 150 objects x 13 attributes, **0/78 cross-cutting attribute pairs**, 66% of meets empty, ~42% of joins the top element); **learned attribute directions can be exactly antipodal** (mutual coherence **1.0000**, since `living_thing` and `artifact` are complementary), so the canonical form's linear-independence assumption is not secured by `d >> |M|` and is never checked in the paper; hypernym trees are the **worst** case for closure gaps, not a benign one (0.93 non-closed / 0.71 overshoot vs 0.000 for a chain); and **Eq. 6's meet orientation looks inverted** (max beats min in 96.5% of trials) though the join half failed to replicate (59%) against the reviewing agent's claimed 99.5%. Forensics from two independently-prompted readers: Figure 4's numbers are printed nowhere in the paper (recovered identically to 3 d.p. from its SVG bar geometry), **12 of 45 Table 2 cells have F1 below both their own precision and recall**, and no released code computes Figure 4. Held-out probes do fit the LRH threshold model (AUC 0.87-0.98 with glosses, controls at chance), with a large fit-on-all inflation caught by the split (join Jaccard 0.901 -> 0.501). Negative results kept in the record: Arm C's meet metric was vacuous by construction (my bug) and its probes were indistinguishable from random directions. |
| [`svgview/`](svgview/README.md) | 2026-07-30 | working on Linux; Windows build never run on a real machine | [`README.md`](svgview/README.md) + [`src/`](svgview/src/) | [andri.dk on Bluesky](https://bsky.app/profile/andri.dk/post/3mrewq7fcsc2j), arguing that launching a full browser to render PDF or SVG is "bonkers insane" — narrowed in his own reply to systems doing it internally. Tested the SVG half by building the alternative: a native Windows-first viewer wrapping `resvg`, ~600 lines. Measured here: **4.8 MiB executable, 12 MiB resident, 16 ms exec→window, 20 ms parse+render to a 1000 px PNG**. Verdict: he is right about SVG and `resvg` had already done the hard part; the argument does *not* transfer to PDF, where the honest options are wrapping Chrome's own PDFium or accepting `hayro`'s coverage gaps. Windows compiles in CI; the file dialog, icon, and association scripts are unverified. |
| [`erdos-gyarfas/`](erdos-gyarfas/README.md) | 2026-07-28 | open problem; partial results | [`README.md`](erdos-gyarfas/README.md) + [`note.html`](erdos-gyarfas/note.html) + [`tutte_coxeter_lemma.py`](erdos-gyarfas/src/tutte_coxeter_lemma.py) | this session |
| [`ms13-campaign/`](ms13-campaign/SUMMARY.md) | 2026-07-24 | closed (compute exhausted; open maths recorded) | [`SUMMARY.md`](ms13-campaign/SUMMARY.md) (academic writeup) + [`NOGOS.md`](ms13-campaign/NOGOS.md) (ledger) + [`BLOGPOST.md`](ms13-campaign/BLOGPOST.md) | [issue #169](https://github.com/oaustegard/claude-workspace/issues/169): campaign against Morell–Skutella Conjecture 1.3 (two-sided unsplittable-flow rounding). **No counterexample.** Produced instead: a **reduction** showing 1.3 restricted to 2-path instances *is* a linear-discrepancy question on **network matrices with demand-scaled columns** — a connection neither literature appears to draw (full-text greps of TVZ/Swamy/MSW25 find no mention of Doerr, "linear discrepancy" or "totally unimodular"); a **theorem** settling that question for k=3 (`R = 3/4` exactly, unconditional, census complete through m=10, exact branch-and-bound on the 2 maximal classes); and a ledger of 20 refuted families/claims **including two conjectures of our own** (12.1 refuted at k=4, the staircase conjecture at k=8). Rediscoveries correctly identified as such: the tightness gadget is Morell–Skutella Fig. 3, the equal-demand bound is Doerr 2004. Enumeration shown dead by arithmetic (~2,070 h at k=4). Open: Q7′ (column-scaled Doerr bound, general k). Methodology notes: a false-positive "counterexample" caught at the certificate gate when two independent verifiers turned out to share one blind spot; every over-claim had the same shape (clean at k≤6, false at larger k). |
| [`ssuf-beta/`](ssuf-beta/RESULTS.md) | 2026-07-24 | done (scoped) | [`RESULTS.md`](ssuf-beta/RESULTS.md) + [`engine.py`](ssuf-beta/engine.py) + [`calibration.py`](ssuf-beta/calibration.py) + [`family.py`](ssuf-beta/family.py) | [claude-workspace#165](https://github.com/oaustegard/claude-workspace/issues/165): quantitative hunt for the SSUF cost-preserving violation constant β* following the Goemans/DGG conjecture disproof (16/15 < β* ≤ 2). Built an exact-rational β* engine (breakpoint-enumerated convex-hull membership LP via sympy's tested simplex, after a hand-rolled one failed its own sanity test). **Calibration against the real Rybin instance was blocked** — no arXiv writeup exists, and this session's WebFetch can't reach the source X thread (HTTP 402, no working mirror) — so calibrated instead against a fully hand-derived, independently-constructed triangle-conflict instance (β\*=1/2, exact match) and swept a parametrized generalization of it — that family's β\* has supremum exactly 1 (approached, never attained or exceeded), a clean negative result short of even the original refuted β=1 bound. Literature gate confirmed TVZ's planar +2·d_max bound and ring-loading bounds (1.1D/1.3D) from primary sources. Honest scope cut: no claim here reproduces or exceeds β\*=16/15; gadget search, ms13 engine sweep, and the ring-loading secondary target were not attempted. |
| [`discrepancy/`](discrepancy/RESULTS.md) | 2026-07-24 | done (D(4) + n≥17 deferred) | [`RESULTS.md`](discrepancy/RESULTS.md) + `growth.png` | [issue #166](https://github.com/oaustegard/claude-workspace/issues/166): certified discrepancy lower-bound records — Komlós per-size K(n) + Beck–Fiala small-t exact values, max-min engine with exact certificates. Literature gate first (logged on the issue): Kunisky's K ≥ 1+√2 record reframed Target A to per-size records; 2025 Bansal–Jiang resolution of Beck–Fiala for t ≥ log²n left small-t exact values open. **Beck–Fiala: D(2)=2, D(3)=3 exactly** — the CEGAR SAT search *rediscovered the Fano plane from scratch* and proved it's the minimum-ground-set (n=7) witness; D(2)=2 (triangle minimal); D(4,n≤9)=3 with D(4)∈{3,4,5} open (PG(2,3) computed weak: disc 2). Small-t truth sits on **D(t)=t**, far below 2t−3. **Komlós: certified rational per-size records K(3)≥1.571, K(4)≥1.731 (beats Kunisky's own n=4 tree matrix 1.707), K(5)≥1.785, K(7)≥1.830**; proved the "exact Δ vs published δ" gap on Kunisky's family is empty (Δ=δ, one-line proof) before wasting compute on it. All records verified by an independent second code path (`verify_certificates.py`); calibration gates G1–G9 green first. |
| [`woodall/`](woodall/README.md) | 2026-07-23 | phase 1 (verifier+calibration done, search deferred) | [`README.md`](woodall/README.md) | [issue #163](https://github.com/oaustegard/claude-workspace/issues/163): SAT/MIP dijoin-packing verifier for Woodall's conjecture τ=3 counterexample search (Goemans-fall follow-on). Built a CEGAR SAT verifier (python-sat/cadical+glucose) for ν(D,u) and brute-force τ; transcribed Schrijver's (D1,u1) counterexample directly off a 600dpi render of Figure 6 in the Feofiloff survey PDF (not from memory). **Calibration gate 1 (Fact 7.1, ν=1/τ=2) passes exactly**, after catching and fixing a real transcription bug (one dashed arc misread) via cross-checking the graph's derived 3-fold symmetry and the paper's "4 critical cuts" claim against an initial read that gave only 3 and a spurious ν=2. Generalized D1 into a ring-of-length-2i family from its own orbit structure and validated the paper's odd/even parity claim (i=3,5 counterexample, i=2,4 not) exactly. A first light random-search pass (16,000 trials) found 0 candidates, confirming unstructured random generation is a weak filter-pass rate (0.15%) for τ≥3 — the real search needs the structured null-arc-resolution and Williams-catalog generators, explicitly deferred (not silently dropped) along with gate 2 (Cornuéjols-Guenin D2/D3 calibration). |
| [`pdf-streaming-test/`](pdf-streaming-test/RESULTS.md) | 2026-07-05 | done (shipped) | [`RESULTS.md`](pdf-streaming-test/RESULTS.md) + [`test_streaming.js`](pdf-streaming-test/test_streaming.js) | "Speed up the UX of `austegard.com/web-utilities/pdf-text-extractor` by streaming pages one-at-a-time; also check if parallelising improves throughput." Added a bounded worker pool + streaming display (pages append to the output pane in strict page order via a `nextToFlush` cursor as they complete) and a URL API knob (`&concurrency=1|2|4|8`, default 4). Playwright harness over three arXiv PDFs (8/15/75 p), routes-intercepted so no external network. **Streaming works** (75-page paper: first page at t≈1s, then every ~150-200ms in order) and **parallelism gives a real win on non-trivial PDFs** — the 75-page paper drops 4.31s → 1.87s wall-clock (**2.3×**, extraction throughput 23.9 → 90.7 pages/s, **3.8×**) going conc=1 → 8; small PDFs (<20 p) plateau at conc=2 since there's nothing left to pipeline through pdf.js's single worker thread. Default of 4 keeps the memory tax down while capturing most of the win. Ordering preserved across every run. Shipped in [oaustegard/oaustegard.github.io PR (branch `claude/pdf-extractor-streaming-ijudhv`)](https://github.com/oaustegard/oaustegard.github.io/pulls). |
| [`atproto-pad-login/`](atproto-pad-login/RESULTS.md) | 2026-07-04 | done | [`RESULTS.md`](atproto-pad-login/RESULTS.md) + [`pad_login.mjs`](atproto-pad-login/pad_login.mjs) + `04_connected.png` | "you can log into ATProto as muninn — try this pad URL" → Drove `austegard.com/bsky/pad.html` end-to-end as **muninn.austegard.com** via Playwright/Chromium: opened the shared-pad URL, clicked *Join*, filled the login dialog with `$MUNINN_BSKY_HANDLE` + `$MUNINN_BSKY_APP_PASSWORD`, waited for `rtc.login()` to establish the PDS session and `rtc.connect(peerDid)` to send its WebRTC knock. **Result: status flipped `Local only · 0 peers` → `Connected · 0 peers`** (session up; the invited peer just isn't online sharing, so no data channel — a one-sided join can't complete). **Load-bearing finding along the way — headless Chromium via `$HTTPS_PROXY` returns `net::ERR_CONNECTION_RESET` on every HTTPS goto unless launched with `--ssl-version-max=tls1.2`.** Curl and Node fetch work; Chromium doesn't. NetLog shows the CONNECT tunnel returns 200 and Chromium's TLS 1.3 ClientHello (~1700 B, ECH extension included) goes out — then the socket is reset (`SOCKET_READ_ERROR net_error=-101 os_error=104`) before any TLS response. `--ignore-certificate-errors`, `ignoreHTTPSErrors: true`, and even `--disable-features=EncryptedClientHello` don't help — only pinning TLS 1.2 does. So the session's egress-gateway MITM terminator can't complete a modern Chromium TLS 1.3 handshake. Side finding: the docs claim the browser NSS trust store is pre-loaded with the proxy CA; ground truth is that `/root/.pki/nssdb` was **empty** — `certutil -L` returned only the header. Adding `/root/.ccr/agent-proxy-ca.crt` with `certutil -A -t 'C,,'` is needed for the proxy-terminated cert to verify after the RESET is worked around. |
| [`session-relay/`](session-relay/RESULTS.md) | 2026-07-02 | done | [`RESULTS.md`](session-relay/RESULTS.md) + [`relay.py`](session-relay/relay.py) + [`README.md`](session-relay/README.md) | "How might this be accomplished?" → [Joshua Shew's Bluesky post](https://bsky.app/profile/joshuashew.bsky.social/post/3mnx2urp3f22n): one Claude session built a relay service so two other sessions could chat and share inbox best practices. **Reproduced with ZERO deployed infrastructure — the shared Turso DB is the relay**: one `relay_messages` table (channel, sender, AUTOINCREMENT seq as cursor) + a ~190-line CLI (`init/post/poll/wait/history`, 5xx backoff). Live run: two concurrent agents with disjoint seeds (muninn-a: `docs/architecture.md` lifecycle; muninn-b: experiment conventions) negotiated a joint 5-item "experiment survival checklist" over channel `hub-coord` — **4 messages, one round-trip each way, ~75s, clean CONSENSUS/ACK termination**, with verifiable two-way knowledge transfer (final checklist contains facts neither seed had alone) and genuine critique (b fact-checked a against Error Patterns, restructured its draft). Protocol findings: table-global seq leaks cross-channel counts (cursor discipline handles it); shell quoting is the ergonomic tax (mandate the `post ... -` stdin path); front-loaded opener + explicit delegation → 1-round convergence. Transport is container-agnostic (HTTPS to Turso) — works across CCotw/Claude.ai session boundaries, unlike the `/tmp`-files original. Caveat: for two Muninn sessions the marginal value over shared memory is *live negotiation*, not knowledge transfer. |
| [`omnigent-library-eval/`](omnigent-library-eval/RESULTS.md) | 2026-06-29 | done | [`RESULTS.md`](omnigent-library-eval/RESULTS.md) | "Is this anything you could make use of?" → [omnigent-ai/omnigent](https://github.com/omnigent-ai/omnigent), the open-source **meta-harness** (5.3k★, 18 days old, alpha `0.3.0.dev0`, Apache-2.0) that orchestrates 11 vendor harnesses (Claude Code/Codex/Cursor/Pi/…) with policies + sandboxing + multi-device. **Verdict: NOT a claude-workspace dependency.** It's a competing harness — a server+CLI that wants to OWN orchestration/model-routing/sandboxing, all of which CCotw already provides; the Python SDK is a *client to a running omnigent server*, not an importable lib; it can't even run here (no model key by design, wants Node 22/tmux/bwrap). **But three borrowable wins:** (1) **free** — `omnigent/spec/skill_sources.py` reads the SAME `SKILL.md` format with a `claude` family, so Muninn's whole skill library loads with zero porting; (2) the **Polly orchestration prompts** — though the cross-vendor *review* discipline they encode Muninn can ALREADY run in CCotw today (Gemini reviewer via the Cloudflare AI Gateway, `gemini_generate()`); omnigent only uniquely adds cross-vendor *implementer* agents (Codex/Cursor/Pi as full coding harnesses), a narrow win; (3) the **ALLOW/DENY/ASK three-level policy engine** (`docs/POLICIES.md`: `blast_radius`/`spawn_bounds`/`cost_budget`) as a reference design if Muninn ever formalizes its `settings.json` permissions. Rec: keep on radar as a place to run Muninn-the-agent *outside* CCotw; don't take the dep. No code executed (alpha server, no key). |
| [`q4-official-vs-ours/`](q4-official-vs-ours/RESULTS.md) | 2026-06-28 | done | [`RESULTS.md`](q4-official-vs-ours/RESULTS.md) + [`run_batched.py`](q4-official-vs-ours/run_batched.py) + [`results.txt`](q4-official-vs-ours/results.txt) | [remax_kb#23](https://github.com/oaustegard/remax_kb/issues/23): run the q4 head-to-head claude.ai couldn't (egress blocks both weight hosts) — does the model authors' own `onnx/model_q4.onnx` (137.8 MB, HF Optimum) match/beat our `JinaQ4ONNXEmbedder` build (~170 MB, remax_kb#14) on retrieval fidelity to fp32, deciding whether to upload ours to HF? NFCorpus, 2058 docs/100 queries, 4 vCPU. **Result: ours is DOMINATED — official wins every axis at once: smaller (138 vs 170 MB, −19%) AND strictly more faithful to fp32 on all four metrics (nDCG@10 0.4291 vs 0.4250, per-doc cos 0.976 vs 0.974, recall@10-vs-fp32kNN 0.870 vs 0.862, Spearman ρ 0.980 vs 0.976).** The hoped-for edge — our int8 embedding-table mop-up holding where Optimum's generic q4 degrades — is refuted; Optimum handles the EuroBERT Gather at least as well. **Decision: do NOT upload ours; close the loop. `embedders.py` now steers users to the official asset.** Side findings: docstring's "cosine 0.975 to fp32" holds (ours 0.9743 on this larger subsample); q4 is NOT faster than fp32 on CPU (~13–14 min each — int4 dequant offsets size; "~2x faster decode" didn't reproduce); patched the bench's one-shot `encode(all_docs)` OOM (~26 GB attn Expand) with numerically-identical mini-batching. Reinforces the prior-art-check lesson: the upstream repo already shipped a smaller, better q4. |
| [`jina-remex-vs-remax/`](jina-remex-vs-remax/RESULTS.md) | 2026-06-27 | done | [`RESULTS.md`](jina-remex-vs-remax/RESULTS.md) + [`score_fidelity.py`](jina-remex-vs-remax/score_fidelity.py) + [`fidelity.png`](jina-remex-vs-remax/fidelity.png) | "Can we apply the **remex** (not remax) quantization optimizations to our Jina q4 embedding — as a practical compressed-Jina format, not necessarily beating remax at a byte budget?" First clears up the **two q4s**: q4-the-*model* (`JinaQ4ONNXEmbedder`, weight quant, fp32-parity float output) is orthogonal to remex/remax, which compress the *vectors*; remex replaces the remax 1-bit step, not the q4 model. Recall-vs-qrels **saturates both ways** (muninn fp32 ceiling 0.90/1.00; NFCorpus fp32 floor 0.241 — embedder-limited), so switched to the saturation-proof metric: **fidelity to fp32's own ranking** (recall@k vs fp32-kNN chunk-level, Spearman ρ, recon cosine — remax's own bench metric). **Result: remex (rotation+Lloyd-Max scalar) ≫ remax (1-bit SimHash) at every byte budget** — NFCorpus n=120: remex ρ 0.92–1.00 vs remax 0.63–0.74; **remex 4-bit @ d768 (384 B) near-lossless (ρ 0.998, R@10-vs-fp32 0.96); remex 2-bit @ 192 B (ρ 0.978) crushes remax d768/k2 @ 192 B (0.741); remex 1-bit @ 96 B beats every remax config.** Headline: **bits beat stacks** — graded magnitude (Lloyd-Max) dominates more sign-bits (stacked SimHash); full-dim-low-bit > truncated-dim-high-bit. remex is the data-oblivious Haar rotation our own ITQ-rejection work (remax#46) endorsed. **Rec: ship remex into remax_kb as the near-lossless/mid-byte codec (default 4-bit @ d768); needs a new SPEC binarizer type + ADC scan path (numpy+scipy), additive not drop-in.** Caveats: muninn n=5; norms excluded from B/row (Jina unit-norm). |
| [`remax-hamming-speedup/`](remax-hamming-speedup/RESULTS.md) | 2026-06-26 | shipped (PR) | [`RESULTS.md`](remax-hamming-speedup/RESULTS.md) + [`bench.py`](remax-hamming-speedup/bench.py) + [`latency.png`](remax-hamming-speedup/latency.png) | [remax_kb#15](https://github.com/oaustegard/remax_kb/issues/15): the 1-bit Hamming scan (`_hamming.hamming_scan`, popcount **LUT gather**) was ~10× slower than a BLAS float-cosine search at small N, forfeiting the latency half of the 1-bit format's promise. Benchmarked the issue's candidates head-to-head (LUT vs `bitwise_count` u8/u64 vs ±1 BLAS matmul vs float cosine) over N=600→1M, single-thread BLAS. **Winner is the cheapest candidate (approach 1): `np.bitwise_count` over a uint64 view of the XOR — ~10× over the current LUT and faster than BLAS float cosine at *every* N** (e.g. N=10k: 0.58 ms vs LUT 7.83 ms vs cosine 2.08 ms), zero-copy, codes stay bit-packed (256 B/row). The ±1 matmul (approach 2) is 2–6× slower *and* costs 8–32× RAM (OOM at 500k); compiled SIMD (approach 3) is unnecessary. Issue's success criterion (Hamming ≤ cosine at N≥10k without losing storage) **exceeded — holds at all N**. **Shipped: `_hamming.py` + `read_v2.py` swap to a shared `_popcount_rows` fast path with a numpy<2.0 LUT fallback (numpy≥1.24 floor preserved) + remax-free regression test ([remax_kb PR](https://github.com/oaustegard/remax_kb/pulls)).** Bit-for-bit exact vs the old LUT across 8 realistic `(dim,k)` widths incl. non-multiple-of-8 byte rows. |
| [`lfm25-230m-verify/`](lfm25-230m-verify/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lfm25-230m-verify/RESULTS.md) + [`run.py`](lfm25-230m-verify/run.py) | "test this Bluesky claim — RUN the model" ([@sungkim post](https://bsky.app/profile/sungkim.bsky.social/post/3mp4zqketts25)) on Liquid AI's LFM2.5-230M. Spec-checked the HF card, then *loaded and ran the weights* on container CPU to verify the runtime-checkable claims. Confirmed: 229.7M params, `Lfm2ForCausalLM`/`model_type=lfm2`, open weights, coherent CPU generation at 29.2 tok/s (fp32, 4 vCPU — the slow path vs card's quantized 42/213 tok/s). Training claims (19T tokens, distill-from-350M) aren't runtime-observable. Minor nuance: post/card say 32K context, loaded config carries 128K positional ceiling. Verdict: post accurate. |
| [`lfm25-embedder-remax_kb/`](lfm25-embedder-remax_kb/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lfm25-embedder-remax_kb/RESULTS.md) + [`lfm25_embedder.py`](lfm25-embedder-remax_kb/lfm25_embedder.py) | Follow-on from `lfm25-230m-verify`: if the 230M *runs* locally, does Liquid's `LFM2.5-Embedding-350M` work as a third remax_kb embedder — in-process CPU, no 847 MB Jina download, no Gemini API key? Wrote an `LFM25Embedder` (sentence-transformers, CLS-pool, 1024-d, query:/document:) on remax_kb's `Embedder` protocol. One real fix: the model's bidirectional remote code predates transformers' `seq_idx` shortconv kwarg (5.12.1 `TypeError`) — patched at our layer, not the model's cached files. Tiny-corpus 1-bit pipeline: 3/3 topical top-3 (matches Jina torch). **Full-float head-to-head on the 73-post muninn corpus: LFM2.5 0.73/0.83 R@5/R@10 — *below* lexical 1.00/1.00 and Jina v5-nano 0.90/1.00, despite being the larger model; embeds slowly (1.1 chunks/s fp32, ~19.5 min).** Verdict: viable where in-process/no-key/no-network is the hard constraint and retrieval is tolerant; Jina still wins on quality. Closing analysis: remax's 1-bit step is corpus-global (can't quantize up front); per-chunk-memory levers = fp16 buffer, streaming Welford mean, dim-truncation (non-MRL caveat), and int8 model quant for RAM+speed. |
| [`muninn-rm3/`](muninn-rm3/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](muninn-rm3/RESULTS.md) + [`bench.py`](muninn-rm3/bench.py) | The platform-less, agent-less site-search floor (no model to host, no Claude to expand the raw query). Pure RM3 vs plain BM25 on the muninn corpus (5-query phase0 gold). **RM3 is a dud — identical R@5, *hurts* R@10 (1.00→0.90 whole-doc); skip it. But plain BM25 (whole-doc) = 0.833/1.00 — ties Jina-q4→remax (0.833/1.00) at ZERO inference/agent/per-query cost, all in the Worker.** Only the agent-expansion path (1.00, unavailable to site search) beats it. Caveat: these are in-vocab acceptance queries; the residual Gemini buys is *vocabulary-divergent* (paraphrase) queries — BM25/RM3 can't bridge that lexical gap, no platform-free option does. Recommendation: drop Gemini for plain in-Worker BM25 IF muninn searches are keyword-ish; keep Gemini only if paraphrase-heavy. Decision input: actual query mix. |
| [`muninn-embedder-bakeoff/`](muninn-embedder-bakeoff/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](muninn-embedder-bakeoff/RESULTS.md) + [`jina_remax.py`](muninn-embedder-bakeoff/jina_remax.py) + [`embed_one.py`](muninn-embedder-bakeoff/embed_one.py) | Test our special-case **Jina→remax** as a Gemini replacement for muninn search (+ how it's practical). On the muninn corpus (5-query phase0 gold): **Jina-q4 full-float = 0.900/1.000 (identical to fp32 — q4 is free); Jina-q4→remax 1-bit at d=512/k=4 (256B) = 0.833 R@5 / 1.00 R@10** (near the float ceiling; the shipped d=256/k=8 default is the weak config at 0.667 — dims beat stacks here too). So Jina→remax is a credible Gemini replacement; lexical (1.00) still edges R@5. **Practicality: indexing is free (offline CI pack with Jina), only the online query-embed is hard** — Worker can't host 170MB (10MB bundle/128MB mem), Workers AI has no Jina, so serve queries from a **CF Container/endpoint running q4** (replaces the paid per-query Gemini call with cheap fixed compute; q4 is what makes that light). Migration = corpus re-pack (Jina space, d=512/k=4) + query-embed service swap; KV 1-bit mechanism unchanged. Sidebar: CF-native Workers-AI BGE alts underperform (bge-large 0.733/bge-base 0.667). Caveats: n=5; no live Gemini (Jina-0.90 float is the reference); embeddinggemma-300m HF-gated. |
| [`rotation-decorrelation/`](rotation-decorrelation/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](rotation-decorrelation/RESULTS.md) + [`sweep.py`](rotation-decorrelation/sweep.py) | "How can I trust either end of the ITQ pendulum? Embedder-specific? Test the decorrelation angle." Controlled study, remax/bench metric (self-retrieval recall@10 vs float32 kNN), pure numpy on precomputed caches: **SPECTER2 (specialized) + Jina-v5 (general)**, k-ladder {1,2,4,8}, simhash/itq/decorr, **in-corpus vs transfer**, 3 seeds. **Resolution: the pendulum is 3 interacting axes** — (1) k: ITQ wins k=1, decays/reverses on the ladder (#46's mechanism reproduced); (2) protocol: ITQ in-corpus overfits (gap ~0.02–0.03 SPECTER2, ~0.04–0.05 Jina; simhash/decorr zero gap); (3) embedder: ITQ's edge is 3× bigger on general Jina (+0.048 k=1) than specialized SPECTER2 (+0.015) — partly embedder-specific but mostly overfit. **My NFCorpus 'win' = perfect storm (general × in-corpus × k=1); #46 = specialized × transfer × ladder.** Honest config (transfer+ladder, k=8): simhash beats itq on BOTH (SPECTER2 −0.029, Jina −0.015). **Decorrelation (α-mix) is a wash** — ties simhash everywhere, no overfit gap. Verdict: keep parameter-free SimHash (#46 upheld, now for general embedders too); the open lead doesn't pay. |
| [`recall-per-byte/`](recall-per-byte/RESULTS.md) | 2026-06-25 | **corrected (re-derived rejected work)** | [`RESULTS.md`](recall-per-byte/RESULTS.md) + [`sweep.py`](recall-per-byte/sweep.py) | Generative-thinking move (random stimulus "river") reframed compaction toward *information-per-stored-bit (the rotation)*. Recall-per-byte bake-off on cached NFCorpus fp32 Jina vectors (600 docs/120 q): remax StackedSignBit vs SimHash vs **ITQ** vs **PQ**. Apparent headline (ITQ/PQ@16B beat shipped remax@256B) **does NOT hold**: ITQ was already tested rigorously and rejected — **remax#46/PR#47 (closed unmerged, SPECTER2 n=10k): learned ITQ loses to centered SimHash at every ladder rung, deficit grows with k**. My k=1/600-doc "win" is the exact *in-corpus overfit artifact* #46 diagnosed (transfer rotations beat in-corpus ones); I never tested the stacked ladder where ITQ definitively loses. PQ not novel either (remex/TurboQuant owns codebook compression; my 256-centroid book on 600 docs flatters it). **Lesson: recall() prior art before claiming a win.** One genuinely open lead (from #46's parting question): a *decorrelating joint multi-rotation* objective — diverse-across-stacks, better-than-random per-stack — validated on SPECTER2 with transfer + ladder, explicit kill criterion. |
| [`jina-int8-remax_kb/`](jina-int8-remax_kb/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](jina-int8-remax_kb/RESULTS.md) + [`quantize.py`](jina-int8-remax_kb/quantize.py) + [`bench.py`](jina-int8-remax_kb/bench.py) | Pivot from quantizing the weak LFM2.5 to quantizing the *strong* embedder: int8-quantize Jina v5-nano's ONNX export to cut its 847 MB download. `quantize_dynamic(QInt8)` → **212 MB (4.0× smaller)**. Head-to-head on the 73-post muninn corpus (same `stage_b.py` methodology): **fp32 0.90/1.00 @ 8.3 ch/s vs int8 0.83/1.00 @ 16.7 ch/s** — 4× smaller, 2× faster, R@10 untouched; the R@5 dip is one query (Q3) sliding past rank 5. fp32 reproduces the prior 0.90/1.00 exactly (harness validated). **int8 Jina dominates the LFM2.5 local option on every axis** (R@5 0.83>0.73, 212<919 MB, 16.7≫1.1 ch/s). Standing rec stays lexical 1.00/1.00; int8 Jina is the real-vector fallback. **Follow-up (NFCorpus, 600 docs/120 qrel-queries, full-float + 1-bit recall): overturns int8.** Per-tensor dynamic int8 is domain-fragile — 0.445 per-doc cosine to fp32 on medical abstracts (vs 0.83 R@5 on muninn tech text; probe rules out seq-length: 0.44@256 vs 0.41@512). **Blockwise 4-bit `q4` matches fp32 (0.975 cosine, cos R@10 0.241 vs 0.242, 1-bit R@10 0.208 vs 0.222) at 170 MB < int8's 212 MB**; q2 too far (0.73). Embedding-table workaround: MatMulNBits leaves EuroBERT's ~400 MB Gather fp32 (naive int4=465 MB > int8), so int8-mop-up the embedding → q4=170/q2=141 MB. 3-bit unsupported (ORT {2,4,8}). **Answer to "is int8 the floor before 1-bit?": no — 4-bit blockwise is smaller, more faithful, domain-robust; floor is 4-bit; the int8 embedding table is now the size-dominant cost.** Recommended quantized Jina = q4, not int8. **SHIPPED: q4 integrated into remax_kb as `JinaQ4ONNXEmbedder` ([oaustegard/remax_kb#14](https://github.com/oaustegard/remax_kb/pull/14)) — embedder + deterministic `scripts/build_q4_onnx.py` + gated test; `model.q4.onnx` hosted (SHA-verified) on the jina-v5-nano-mirror release; fp32-parity query cosine 0.977; opt-in experimental.** |
| [`kb-packer-web/`](kb-packer-web/build_packer.py) | 2026-06-25 | shipped (PR) | [`build_packer.py`](kb-packer-web/build_packer.py) + [`kb-packer.html`](kb-packer-web/kb-packer.html) | Browser KB-packer tool for austegard.com/ai-tools/ — drag files → BM25 index → download an installable `<name>.skill`, fully client-side (no upload/model/network). Generator inlines the `creating-kb` build core + vendored runtime (search.js/search.py/bundle_SKILL.md) into one self-contained HTML matching the ai-tools single-file convention; browser-built `.skill` byte-identical to the Node CLI's. E2E-tested in headless Chromium. [oaustegard.github.io PR #253](https://github.com/oaustegard/oaustegard.github.io/pull/253). Moved here out of claude-skills (#714 closed — skills repo isn't for hosted web apps). |
| [`lexical-kb-phase0/`](lexical-kb-phase0/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lexical-kb-phase0/RESULTS.md) | The load-bearing study: does agent-expansion + BM25 match embeddings on the real muninn corpus (73 posts)? **Yes, with a bounded residual.** Chunk-size sweep confirms the null hypothesis (whole-doc = 500-char on recall, 17× fewer chunks, R@5 up). Head-to-head vs the full-float Jina ceiling: lexical 1.00/1.00 vs embedding 0.90/1.00 mean R@5/R@10 on in-vocab queries (lexical ties/edges). Paraphrase frontier: agent expansion ties/beats embedding on 4/5; the 5th is what embeddings buy — a vocabulary-divergent relevant post expansion missed. |
| [`lexical-kb/`](lexical-kb/RESULTS.md) | 2026-06-25 | skill built+tested | [`RESULTS.md`](lexical-kb/RESULTS.md) | Embedding-free portable KB: BM25 over a precomputed index, semantic layer moved to the consuming agent (query expansion at search time). Ships as a self-contained `.skill` zip (SKILL.md + search.py + index.json + chunks.jsonl), pure stdlib, no model/network. Makes remax_kb#12 (pure-JS packer) trivial by deleting the embedding half. Tiny-corpus test validated correctness (BM25/RM3/filter) + caught/fixed a substitutive-expansion bug (expansion is now additive over the raw query). Quantitative lift → Phase 0 (muninn corpus). |
| [`kb-k-sweep/`](kb-k-sweep/RESULTS.md) | 2026-06-22 | done | [`RESULTS.md`](kb-k-sweep/RESULTS.md) | recall@10 over a (dim,k) grid on the real Mac-search corpus (Gemini `gemini-embedding-001`, 1779 chunks). Answers CROSSOVER's open follow-up on a non-SPECTER2 embedder, then sweeps dims. **Headline: at a fixed byte budget, dimensions beat stacks** — shipped dim=256/k=8 is Pareto-dominated by dim=512/k=4 (+5.9 R@10, same bytes) and dim=768/k=2 (smaller AND better). dim=512/k=1 matches shipped recall at ¼ the size. int8 rotations then make 768/k2 smaller than baseline (remax_kb#10, muninn#213). |
| [`anything-to-text/`](anything-to-text/RESULTS.md) | 2026-06-22 | done | [`RESULTS.md`](anything-to-text/RESULTS.md) + [`build.py`](anything-to-text/build.py) | "Anything to Text SPA for austegard.com/web-utilities, composite of two uploaded apps, re-brand from vendor to my site" — merged in-browser OCR (PaddleOCR) + Whisper transcriber into one drop-zone-routed page; engines kept verbatim in separate ES modules, reskinned to Grouch's Workshop. [oaustegard.github.io PR #247](https://github.com/oaustegard/oaustegard.github.io/pull/247) |
| [`dc-mall-timelapse/`](dc-mall-timelapse/RESULTS.md) | 2026-06-18 | done | [`RESULTS.md`](dc-mall-timelapse/RESULTS.md) | "timelapse of the DC mall (WAMO) webcam since May 1" — EarthCam HOF endpoint only serves newest ≤50 stills (no pagination); May 1 unreachable. Built ~4-day clip (Jun 8→12) from what's public. `imageio-ffmpeg` = static ffmpeg in CCotw. |
| [`python-lsp-stress/`](python-lsp-stress/RESULTS.md) | 2026-06-15 | done | [`RESULTS.md`](python-lsp-stress/RESULTS.md) | validate pyright 1.1.410 (PR #124) — `pyright-langserver` LSP + batch checker against Django (2,922 `.py`); references 2,490 hits in 3.9s, whole-pkg batch in 22s |
| [`qat-cpu-demo/`](qat-cpu-demo/RESULTS.md) | 2026-06-06 | done | [`RESULTS.md`](qat-cpu-demo/RESULTS.md) + [`qat_vs_ptq.png`](qat-cpu-demo/qat_vs_ptq.png) + [`inference/`](qat-cpu-demo/inference/README.md) | "could we run a smaller version of Gemma 4 QAT locally, GPU-less?" — toy-scale QAT-vs-PTQ reproduction on a char-LM (CPU), plus the real Gemma 4 E2B Q4_0 checkpoint clocked at ~20 tok/s on CPU |
| [`memory-redundancy-probe/`](memory-redundancy-probe/RESULTS.md) | 2026-06-04 | done | [`RESULTS.md`](memory-redundancy-probe/RESULTS.md) | "is the memory store a landfill?" probe + curate/consolidate review; spun out of reviewing arxiv 2606.03787 |
| [`optimizing-skills-retro/`](optimizing-skills-retro/RESULTS.md) | 2026-05-29 | done | [`RESULTS.md`](optimizing-skills-retro/RESULTS.md) + [`proposed-patch-optimizing-skills.md`](optimizing-skills-retro/proposed-patch-optimizing-skills.md) | retro test of `optimizing-skills` gate on the down-skilling v1.2.0 edit |
| [`skillopt-skill/`](skillopt-skill/README.md) | 2026-05-29 | done (ready to port) | [`README.md`](skillopt-skill/README.md) + [`optimizing-skills/SKILL.md`](skillopt-skill/optimizing-skills/SKILL.md) | SkillOpt review (arXiv:2605.23904) → new skill |
| [`haiku-assessment/`](haiku-assessment/RESULTS.md) | 2026-05-26 | done | [`RESULTS.md`](haiku-assessment/RESULTS.md) + [`GUIDE.md`](haiku-assessment/GUIDE.md) | Haiku 4.5 vs. down-skilled-Haiku probe |
| [`spoke-branch-cleanup-2026-05-25/`](spoke-branch-cleanup-2026-05-25/RESULTS.md) | 2026-05-25 | done | [`RESULTS.md`](spoke-branch-cleanup-2026-05-25/RESULTS.md) | one-shot ops |
| [`phase-a-bridges/`](phase-a-bridges/RESULTS.md) | 2026-05-23 | done | [`RESULTS.md`](phase-a-bridges/RESULTS.md) | [#90](https://github.com/oaustegard/claude-workspace/issues/90), [PR #93](https://github.com/oaustegard/claude-workspace/pull/93), [PR #96](https://github.com/oaustegard/claude-workspace/pull/96) |
| [`te-bridges/`](te-bridges/RESULTS.md) | 2026-05-24 | done (4 runs) | [`RESULTS.md`](te-bridges/RESULTS.md) + per-run | [#97](https://github.com/oaustegard/claude-workspace/issues/97), PRs [#98](https://github.com/oaustegard/claude-workspace/pull/98)/[#99](https://github.com/oaustegard/claude-workspace/pull/99)/[#100](https://github.com/oaustegard/claude-workspace/pull/100)/[#101](https://github.com/oaustegard/claude-workspace/pull/101) |
| [`specter2-gap-issue-87/`](specter2-gap-issue-87/) | 2026-05-23 | done | embeddings JSON | [#87](https://github.com/oaustegard/claude-workspace/issues/87) (remex#69 phase-0 gap fill) |
| [`muninn-kb-issue-76/`](muninn-kb-issue-76/README.md) | 2026-05-13 | done (reproducible) | [`README.md`](muninn-kb-issue-76/README.md) (build recipe) | [#76](https://github.com/oaustegard/claude-workspace/issues/76) |
| [`reviews/`](reviews/) | 2026-04-21 | done | the `.md` files are the results | freeform repo reviews |
| [`snooker-break/`](snooker-break/snooker-break.html) | 2026-05-10 | done | [`snooker-break.html`](snooker-break/snooker-break.html) (interactive) | spike — apex vs Murphy break strike |

## Per-experiment notes

### `svgview/` — is a browser really the wrong tool for rendering SVG?

Prompted by [a Bluesky post](https://bsky.app/profile/andri.dk/post/3mrewq7fcsc2j)
calling browser-based PDF/SVG rendering "bonkers insane." The claim is testable,
so it got tested: build the lightest native SVG viewer that is actually usable
and see what it costs.

It costs very little, because `resvg` had already done the hard part. The whole
viewer is ~600 lines over `resvg` + `winit` + `softbuffer`, and the numbers are
one to two orders of magnitude below a browser-based equivalent: 4.8 MiB
executable, 12 MiB resident, 16 ms from `exec` to a window on screen. Static SVG
is a solved problem outside the browser and has been for years.

The argument does **not** generalize to PDF, which is what the original post
actually lumped together. PDF is thirty years of accretion — forms, JBIG2/JPX,
seven shading types, CID fonts, encryption — and the two honest Rust options are
wrapping PDFium (which *is* Chrome's PDF engine, just without the browser around
it) or accepting `hayro`'s documented gaps. The elephant gun exists because that
particular pigeon fights back.

Two process notes worth more than the code:

- The first version of the GUI smoke test counted distinct colours per
  screenshot and reported `ok` for all seven key bindings while **none** of them
  were reaching the window — with no window manager on the virtual display,
  nothing had assigned input focus. Both the bug and the useless check are in
  `METHODS.md`.
- The app icon is generated by rasterising `assets/icon.svg` with svgview
  itself, so artwork and icon cannot drift. Costs about 40 lines of Python.

Not done: no interactive Windows testing at all, so the `Ctrl+O` dialog,
embedded icon, GUI-subsystem behaviour, and the file-association scripts are
written-but-unproven. No macOS attempt.

### `erdos-gyarfas/` — does every min-degree-3 graph have a power-of-two cycle?

Open since 1995; Erdős expected it false and offered $100 for a proof, $50
for a counterexample. What holds up: a reformulation of the conjecture as a
growth race between two integer sequences (it is false for cubic graphs iff
`f(k) <= 2^(k+1) - 1`, where `f(k)` is the smallest cubic graph avoiding
`4..2^k`); a barrier proposition showing no bound on the *number* of distinct
cycle lengths can ever settle it, since the target set has size `log2 n`; a
one-line corollary of Bondy–Vince killing every residue-class construction;
and a sharpening of Carr (arXiv:2605.22844) from 4/7 to 2/3 using a corollary
that paper proves and never applies.

Also a genuine gap in Exoo (arXiv:1403.5636): the supporting lemma for
`f(5) <= 450` asserts every 8-cycle of the Tutte–Coxeter graph contains two
consecutive outer-Hamiltonian edges. It has 90 such cycles and 10 violate it.
`src/tutte_coxeter_lemma.py` is self-contained — run it.

The computational half — complete censuses at orders 24 and 26 — reproduces
Markström (2004) graph-for-graph rather than extending it. That was not the
intent: it was built on the belief that the order-26 count was unknown, which
was wrong, and the paper was a search away. Kept as independent verification
of a 2004 computation nobody had checked since, and as a caution.

Frontier: a cubic counterexample needs 54 <= n <= 62, even, avoiding cycles
of length 4, 8, 16 and 32.

### `ms13-campaign/` — kill Morell–Skutella Conjecture 1.3 (campaign)

Issue #169, multi-session; state in `WORKLOG.md` + `NOGOS.md` (the ledger is
the deliverable). Three sessions, no counterexample, and the negative results
compounded into a structure theory: path-closed 2-path instances are
out-trees (Lemma 4, from the post-mortem of a false positive that the
certificate gate caught); on that class the conjecture is exactly signed
discrepancy over the tree's totally unimodular network matrix (Lemma 5);
equal demands are provably safe (Lemma 6, Hoffman–Kruskal); and 85% of
enumerated instances are already closed by MSW25's series-parallel theorem
via a K4-minor criterion (NG-9), with the Rybin instance landing exactly on
the K4 boundary. The swept live set plateaus at −4/9·d_max. Kill-path B is
parked (β* ladder self-limits at 7/6 against the 2 needed). What's left is
one clean open question — linear discrepancy of a network matrix with
demand-scaled columns — which subsumes every instance in the class and has
neither a theorem nor a counterexample in the literature.

### `discrepancy/` — certified Komlós + Beck–Fiala lower-bound records

Issue #166's two-target discrepancy play, run with the ms13 max-min skeleton
and a hard "exact certificates only" rule: floats screen, integers decide
(ℤ[√2] pairs for the Kunisky family, Fractions for rationalized search
records, SAT UNSAT + exhaustive enumeration for set systems). The mandated
literature gate reshaped both targets before compute: Kunisky's asymptotic
K ≥ 1+√2 made "beat disc > 1" dead on arrival (per-size K(n) records are the
honest deliverable), and the 2025 Bansal–Jiang line resolved Beck–Fiala
asymptotically while leaving small-t exact values unpublished. Two negative
results caught early saved the day: Δ = δ on Kunisky's finite instances
(provable in one line — the planned "compute exact Δ" deliverable was
vacuous), and PG(2,3) has disc 2 (no classical D(4)=4 shortcut). The CEGAR
search rediscovering the Fano plane as the *provably minimal* D(3)=3 witness
is the flagship result; the per-size Komlós records (n=4 beating Kunisky's
own tree matrix at n=4 is the structural surprise) are the second. Search
lesson recorded in RESULTS: restart diversity dominates iteration depth on
this piecewise-linear landscape (n=7 record moved 1.726 → 1.830 on a lucky
seed batch; the n=4 record was never re-found in 24 later restarts).

### `session-relay/` — inter-session Claude coordination over Turso

Reproduces the Bluesky relay pattern (two Claude sessions chatting to share
best practices) without deploying anything: the shared Turso DB is the
rendezvous, so the relay is one table plus a small CLI. Validated with a live
two-agent run — disjoint knowledge seeds, blind negotiation over the channel,
clean CONSENSUS/ACK handshake, and a joint deliverable neither side could have
written alone. `relay.py` is reusable as-is for cross-session coordination
(CCotw ↔ Claude.ai included); see RESULTS.md for the protocol lessons
(cursor discipline, stdin posting, front-loaded openers).

### `remex-vs-higgs-ablation/` — does remex beat the QuIP#/HIGGS lineage for retrieval?

Follow-on to `jina-remex-vs-remax/`, commissioned as issue #8. remex and the
**QuIP# → HIGGS → TurboQuant** lineage differ on three axes at once, so a
head-to-head can only say *which* wins. This runs the full **2×2×2 factorial** —
rotation (dense Haar vs randomized Hadamard) × norm handling (exact fp32 norm
out-of-band vs per-block scale in the payload) × codebook (scalar Lloyd-Max vs
Gaussian-MSE-optimal multi-dimensional grid) — so the difference can be
attributed to an axis rather than to a method.

**Only axis C moves.** Pooled over 3 corpora (d=100/768/1024) × 6 bit widths ×
5 rotation seeds: rotation +0.0005 ± 0.0016 recall@10, norm handling
+0.0014 ± 0.0014, codebook **+0.0113 (cosine) / +0.0146 (inner product)**. The
codebook effect peaks at **+0.035 at 2–3 bits**, decays monotonically, and is
gone by 8 bits — and it is about twice as large at d=100 as at d=768/1024,
which is what the scalar-vs-vector gap should do as coordinates get closer to
i.i.d. Gaussian.

**Two of the four pre-registered predictions failed.** The randomized Hadamard
was predicted 10–100× *faster* to apply at d=768–1024; it is **11–24× slower**,
because numpy runs the dense rotation as one BLAS `sgemm` and the FWHT as a
Python loop over strided slices. The ratio does move the right way with
dimension (50× at d=100 → 1.5× at d=8192), so the asymptotics are visible, but
the crossover is far past any retrieval dimension. And exact-norm was predicted
to win under inner product; it does not — partly because BGE-family encoders
are *trained* under cosine, so their raw norms barely vary (CV 1.4–2.7% against
GloVe's 20%) and inner-product retrieval is nearly the same problem as cosine.

**The result most likely to change a decision** is one the scheduled adversarial
review forced into the writeup: the headline tables exclude the rotation and the
codebook as "shared across the index", which is the convention both lineages use
and is *not symmetric*. remex's shared cost is a d×d rotation; the vector arm
additionally carries a codebook up to 1 MiB. On a 20,000-vector index at 4 bits
that is **52.5 B/vector against a 50 B payload**, and the recall-per-byte
ordering reverses: remex at 6 bits (81 B true, R@10 0.965) beats HIGGS-like at
4 bits (112.5 B true, 0.893) on bytes *and* recall. The vector arm needs roughly
**350,000 vectors** before its codebook amortizes below 5%.

So remex loses on distortion in the 2–3 bit regime and should not be defended
there — but its distinctiveness was never distortion. It is numpy-only,
calibration-free, data-oblivious, 11–50× faster to apply, carries a 2 KiB side
table instead of 1 MiB, and wins on true bytes-per-vector below a few hundred
thousand documents.

**Rerun 2026-08-02 under the [`gating`](https://github.com/oaustegard/claude-skills/tree/main/gating)
skill.** The question that run asked was not "does the calibration gate pass"
but "can it go red." Audited (`audit.py`), three of its checks could not fail in
the way that mattered: the rotation check compared the two arms to each other
and nothing else, so replacing **both** with the identity still passed; the
axis-C check asserted an inequality with no margin, which a vector arm doing no
vector quantization clears by sampling noise; and the published-table anchor
stops at 5 bits while the sweep runs to 8. Mutation testing (91 mutants,
55 survivors) found the gate scored codebook *contents* and never ran the
*encoder* — mutating the decision boundaries, the nearest-neighbour k, and the
axis-B norm-mode branch all left it green. The rebuilt gate (`gate.py`, 139
checks, 5 known-bads, 14 stated coverage limits) now blocks the sweep on a
non-zero exit instead of being a documented step, and three of its own new
checks went red on first run — including one that turned an *attainable*
optimum into a failure, because a Hadamard transform maps a coordinate spike to
exactly ±1/√d. Conclusions are unchanged: `glove100` and `nfcorpus1024`
reproduce to four decimals, `arxiv768` shifts because its abstracts are a fresh
draw.

**Process is most of the value here.** The two-sided calibration gate ran
*before* the sweep and immediately caught Lloyd-from-random-init converging to
grids **worse than the scalar quantizer** at 6 and 8 bits — which would have
shipped as "scalar wins axis C at high rate", exactly the wrong conclusion the
issue warned about. Fix: seed Lloyd from the scalar product grid, and keep the
unrefined product grid as a candidate. A scheduled adversarial review then found
five further blocking defects, two confirmed by direct measurement before
anything was changed: a **stale codebook** served by a cache keyed on the
problem `(m, K)` rather than on the method (the 8-bit vector arm was 87% worse
than scalar), and a **Lloyd-Max MSE identity evaluated off the fixed point**
(+16% at 8 bits — and Max (1960)'s table stops at 5 bits, so the published-value
check could never have caught it, while the inflated baseline made the gate
*more* permissive). Also: a "provably no worse than scalar" guarantee that was
argued rather than enforced (Lloyd is monotone in *training* distortion, but
selection happens on held-out), a block/sub-vector divisibility bug that
corrupted only the HIGGS-like arm and so confounded axes B and C, and a gate
that never certified the grids behind any glove result. Separately, scoring
`q·x̂` without dividing by `‖x̂‖` manufactured a fake 1-bit axis-C reversal,
because the 1-bit scalar quantizer's reconstruction norm is constant *by
construction* and pays no penalty for norm error.

Gate anchors: Max (1960) table 1 reproduced to the printed digits at 1–4 bits,
E8's normalised second moment to 3.7e-4 relative, a tuned ball-shaped E8
codebook (QuIP#'s codebook family) that the trained grid must beat by ≥1%, and
HIGGS §4.3's own stated envelope (grid dimension p ∈ [1,5], size n ∈ [9,4096]) —
which the grids here meet or exceed at every bit width.

### `jina-remex-vs-remax/` — remex as a practical compressed-Jina vector format

Oskar asked whether the **remex** (rotation + Lloyd-Max scalar quant, the
TurboQuant lib) optimizations could apply to "our Jina q4 embedding," loosened to:
can remex be a practical compressed-Jina format on its own, not just a byte-budget
winner over remax? Two prerequisites cleared first: (1) **remex ≠ remax** — two
different spokes (remex = multi-bit scalar quant; remax = 1-bit centered SimHash,
what remax_kb ships). (2) **The two q4s** — q4-the-model (`JinaQ4ONNXEmbedder`,
weight quant, outputs fp32-parity floats) is orthogonal to remex/remax, which
compress the output *vectors*; remex would replace the remax 1-bit step.

The measurement was the real work. Recall-vs-human-qrels **saturates from both
ends** — muninn ceilings (fp32 R@5/R@10 0.90/1.00, every remex config ties it),
NFCorpus floors (fp32 R@10 0.241, embedder-limited, everything piles at ~0.24).
Both hide quantization damage because they conflate embedder quality with codec
fidelity. Fix: score each code against **fp32's own ranking** (recall@k vs
fp32-kNN at chunk level, Spearman ρ, reconstruction cosine — remax's own bench
metric). That de-saturates cleanly.

Result (NFCorpus, n=120): **remex dominates remax at every byte budget.** remex
ρ 0.92–1.00 vs remax 0.63–0.74. remex 4-bit @ d768 (384 B) is near-lossless
(ρ 0.998, recon 0.9953, R@10-vs-fp32 0.96); remex 2-bit @ 192 B (ρ 0.978) beats
remax d768/k2 at the same 192 B (0.741); remex 1-bit @ 96 B beats *every* remax
config. **Bits beat stacks**: graded magnitude (Lloyd-Max) beats more sign-bits
(stacked SimHash) at equal bytes; and full-dim-low-bit beats truncated-dim-high-bit
(2b/d768 192 B > 4b/d512 256 B). remex's rotation is data-oblivious random Haar —
the kind `recall-per-byte`/`rotation-decorrelation` endorsed when they rejected
learned ITQ (remax#46). Recommendation: ship remex into remax_kb as the
near-lossless/mid-byte codec (default 4-bit @ d768), alongside (not replacing)
1-bit; costs a new SPEC binarizer type + ADC scan path (numpy+scipy), additive.
Caveats: muninn n=5 (directional); Jina vectors unit-norm so remex per-row norms
excluded from B/row (honest parity with remax). Assets (q4 ONNX, muninn.kb,
NFCorpus) gitignored — regenerable via the run scripts.

**Reconciliation with `one-bit-beats-two` (Oskar flagged the apparent
contradiction):** the founding remax result — 1-bit *beats* 2-bit — is real and
reproduced here on SPECTER2 (this harness: 1/2/4/8 = 0.642/0.496/0.742/0.974 vs
blog 0.635/0.501/0.731/0.971), but is **embedder-specific**. My first hypothesis
(a Matryoshka bit-shaving artifact) was *falsified*: `reconcile.py` shows both
Matryoshka and independent codebooks monotone on Jina, and `reconcile_specter2.py`
shows the reversal in *both* on SPECTER2 — so it's the embedder, not the codebook
construction. SPECTER2 (specialized, tight clusters) reverses; Jina v5-nano
(general, isotropic) is monotone. So remax_kb's 1-bit choice was right for
SPECTER2 but flips for the Jina-based muninn deployment. `reversal.png`.

### `remax-hamming-speedup/` — make the 1-bit Hamming scan beat BLAS float matmul

remax_kb#15: the storage win of 1-bit codes (12× smaller per vector) was proven,
but the *latency* win wasn't — the reader's `hamming_scan` used a per-byte
popcount **LUT gather** (`POPCOUNT_LUT[xor].sum(axis=1)`) that micro-benched ~10×
slower than an equivalent BLAS float-cosine search at small N. The issue listed
four candidates cheapest→heaviest; `bench.py` times them all on the muninn
micro-bench shape (d=512·k=4 → 256 B/row) over N=600→1M with BLAS pinned to one
thread, asserting each Hamming kernel returns the *identical* top-k to the LUT.

**The cheapest candidate wins outright.** Approach 1b — `np.bitwise_count`
(numpy≥2.0, hardware POPCNT) over a **uint64 view** of the contiguous XOR (8× fewer
elements before the reduction) — is **~10× faster than the current LUT and faster
than BLAS float cosine at every N**, zero-copy, codes stay bit-packed. The ±1 BLAS
matmul (approach 2) ranks identically (`q·D = nbits − 2·Hamming`) but is 2–6×
slower than the popcount path *and* must un-pack the corpus to int8/fp32, forfeiting
the storage win (OOM at N=500k on 15 GB); a compiled SIMD kernel (approach 3) is
unnecessary when pure numpy already beats BLAS. Issue success criterion met and
exceeded — the win holds at all N, not just ≥10k.

Shipped to remax_kb: `_hamming.py`'s `hamming_scan` and `read_v2.py`'s duplicate
popcount both delegate to a shared `_popcount_rows(xor)` — fast path when
`np.bitwise_count` exists (uint64 view when row bytes %8==0, else uint8), per-byte
LUT fallback for numpy<2.0 so the `numpy>=1.24` floor is untouched. Added
`tests/test_hamming.py` (remax-free, 20 cases) proving bit-for-bit equivalence to
the frozen LUT across 8 `(dim,k)` widths including non-multiple-of-8 byte rows,
plus top-k order, distance bounds, and the validation guards. Reproduce with
`OPENBLAS_NUM_THREADS=1 python3 bench.py && python3 plot.py`.

### `kb-packer-web/` — browser KB packer for austegard.com/ai-tools/

The user-facing front end of the lexical-KB line: a single-file web tool where
you drop files and download an installable `<name>.skill` — pack → download →
install in any skill-capable agent → query. Fully client-side (the embedding-free
design is what makes a pure-browser packer possible). `build_packer.py` is a
generator that inlines the `creating-kb` build core (`lexkb-web.mjs`, exports
stripped) plus the vendored shipped runtime (`search.js`/`search.py`/
`bundle_SKILL.md`, snapshot from claude-skills main) into one self-contained HTML
matching the ai-tools `<github-toc>`-listed convention — re-run it to re-sync the
snapshot. A browser-built `.skill` is byte-identical to a `creating-kb` Node build;
verified end-to-end in headless Chromium (stage files → build → download → query
with the shipped search.py). Lives on austegard.com (oaustegard.github.io PR #253),
not in claude-skills — the skills repo is for skills, not hosted web apps (the
earlier in-repo SPA, #714, was closed). `vendor/` holds the pinned input snapshot.

### `lexical-kb-phase0/` — does agent-expansion + BM25 match embeddings?

The study that could have killed the embedding-free KB. On the real
muninn.austegard.com corpus (73 posts, the corpus behind the published embedding
`muninn.kb`), three stages: (1) chunk-size sweep — confirms lexical tolerates big
chunks (whole-doc 73 chunks = 500-char 1237 chunks on recall, R@5 even up); (2)
head-to-head vs the full-float Jina ceiling — lexical ties/edges on in-vocabulary
queries (1.00/1.00 vs 0.90/1.00 R@5/R@10); (3) paraphrase frontier — agent
expansion ties or beats embedding on 4 of 5 lay-phrased queries, recovering gaps
raw BM25 loses (P1: 0.50→1.00), with one honest residual (P5) where a
vocabulary-divergent relevant post was found only by the embedding. Verdict:
competitive with embeddings, deleting the model/asset/embed-passes, at the cost
of a bounded recall residual on conceptually-related-but-lexically-divergent docs
— the precise answer to "what did embeddings buy?". Methodology note: gold judged
from post descriptions (not body terms) to avoid pre-favouring lexical; embedding
is the float ceiling (deployed 1-bit codes would score lower), making it a
conservative test. Closes Phase 0 of the lexical-kb / creating-kb line.

### `lexical-kb/` — embedding-free portable KB as a `.skill`

The "what if we drop embeddings entirely and let the agent expand the query?"
thread. The embedding model's only job in retrieval is bridging the
query↔document vocabulary gap; an agent in the loop does that with current
context. So: precompute a BM25 inverted index, ship it as a self-contained
`.skill` zip (`SKILL.md` + `search.py` + `index.json` + `chunks.jsonl`, pure
stdlib), and put the semantic step in the SKILL.md protocol — the agent emits
`{core, expand}` weighted terms at query time. `search.py` also carries an RM3
pseudo-relevance fallback for non-expanding consumers and `--filter` over
structured metadata. Deleting the embedding half is what makes remax_kb#12
(pure-JS browser packer) trivial: no quantizer, no projection, no fp-fragile
Python↔JS bit-identity — chunker + BM25 + zip writer, all exact-integer.

Tiny-corpus test (Federalist + Gettysburg) validated the full lifecycle and
caught a real bug: substitutive expansion (curated synonyms *replacing* the
user's words) sent a gold doc to score 0. Fix: expansion is now strictly
additive — the raw query contributes at a floor weight beneath core/expand, so
a synonym can lift but never drop a literal match. The corpus is too topically
disjoint to show ranking lift (raw BM25 wins at rank 1 on any shared term); the
quantitative study — chunk-size sweep + recall vs the embedding `muninn.kb` — is
Phase 0 on the real muninn corpus.

### `kb-k-sweep/` — remax stack-count `k` vs recall on the Mac-search corpus

Settles whether the shipped `k=8` over-provisions bits. CROSSOVER answered this
on SPECTER2 (flat curve, k=1 ≈ 88% of k=8); this reruns on the production
embedder — Gemini `gemini-embedding-001` via the CF AI Gateway — over the real
1779-chunk muninn index. Part 1 (`sweep.py`): curve steeper at low `k` on Gemini
(k=1 = 0.585 R@10, 82% of k=8's 0.713); truncation 768→256 is the dominant loss.
Part 2 (`dim_sweep.py`) sweeps the (dim,k) grid against fixed float-768 GT and
settles it: **at a fixed byte budget, dimensions beat stacks at every budget.**
Shipped dim=256/k=8 (256 B, 0.713) is Pareto-dominated — dim=512/k=4 = 0.772 at
the same bytes, dim=768/k=2 = 0.781 at 25% smaller, dim=512/k=1 = 0.697 at ¼ the
size. Recommendation: re-pack at 512/k4 or 768/k2 (free re-binarization, no
re-embed). Embed-once/re-binarize: 18s corpus, ~9s for the whole 30-cell grid.
Part 3 (`build_both.py`/`verify.py`) builds real .kbi artifacts and verifies on
40 real RETRIEVAL_QUERY queries: dense R@10 256/k8=0.585 → 512/k4=0.698 →
768/k2=0.723; no Worker code change needed (JS reader is data-driven). Part 4
(`int8_rotations.py`): the rotation sidecar (`k·dim²·4` B, corpus-independent,
dominates a small .kbi) quantizes f32→int8 for a 4× shrink at 0.24% bit-flips /
zero recall loss — making 768/k2+int8 both higher-recall AND smaller (~2.6 MB)
than the shipped 256/k8 (3.6 MB). Part 5: int8 shipped as a backward-compatible
remax_kb SPEC_v2 change ([remax_kb#10](https://github.com/oaustegard/remax_kb/pull/10),
v0.2.0) + worker reader/build flag ([muninn#213](https://github.com/oaustegard/muninn.austegard.com/pull/213));
the real JS worker reader validated bit-identical to the Python encoder on an
int8 .kbi under node 22. Live deploy gated on merging both PRs.

### `dc-mall-timelapse/` — EarthCam WAMO cam, "since May 1"

Requested a timelapse of the Washington Monument (WAMO) cam back to May 1 via
EarthCam's `gethofitems.php` "index". Ground-truth probing killed the premise:
that endpoint only ever serves the **newest ≤50** Hall-of-Fame stills —
`start`/`date_start`/`date_end`/`last_item_id` are all ignored (tested every
combination), `length` caps at 50, `fullcount:75000` is decorative. The two
archive VODs (`events/dc/WAMO.mp4`, `archives/4356/backup.mp4`) are rolling
~60-minute clips. Deep date-selectable archive is premium, no open API. So
May 1 (~7 weeks back) is unreachable for free. Built `wamo_timelapse.mp4` from
the only public history — 50 frames, **Jun 8→12 2026**, 8 fps — and delivered
it honestly labeled. Reusable tooling find: `pip install imageio-ffmpeg`
provides a static **ffmpeg 7.0.2** binary, the clean way to get ffmpeg in a
CCotw container (apt's archive 404s). Full table in `RESULTS.md`.

### `python-lsp-stress/` — pyright LSP against a large codebase

Validated PR #124's pyright 1.1.410 (base layer) end-to-end against Django
(2,922 `.py` files). `lsp_probe.py` is a stdlib-only LSP client that drives
`pyright-langserver` over stdio: `initialize` advertised all 16 navigation
providers in 0.19s; `references` on the `Model` class returned 2,490 hits
across the tree in 3.9s; `workspace/symbol` returned 3,817 in single-digit
seconds. The batch `pyright` checker analyzed the 907-file `django` import
closure in 22s with structured JSON output. Cross-file navigation scales;
the Python LSP is production-ready for large-codebase work. Full numbers in
[`RESULTS.md`](python-lsp-stress/RESULTS.md).

### `qat-cpu-demo/` — QAT vs PTQ, GPU-less, in 2 minutes

Asked whether we could run a smaller version of Google's Gemma 4 QAT locally on
the GPU-less container. Built a faithful toy-scale reproduction of the post's
central claim: a 2-layer char-level transformer (393 K quantizable weights)
trained end-to-end on 4 CPU cores in 2m13s, no GPU/network/external-data. QAT =
fake-quant (per-channel symmetric round-and-rescale) in the forward pass +
straight-through estimator in the backward; PTQ = the same rounding applied to a
finished fp32 model. **Result:** at the 2-bit floor (where the Gemma stack uses
2-bit for token generation), PTQ blows perplexity 1.30→5.21 (+3.91, 4× worse)
while QAT recovers to 1.31 at the identical 0.098 MB footprint. int4/int3 PTQ
don't degrade at this scale, so their "recovery %" is noise — honest caveat in
RESULTS.md. Verdict: the technique runs locally trivially. **Follow-up
(`inference/`):** also pulled and ran Google's *actual* checkpoint —
`gemma-4-E2B-it-qat-q4_0-gguf` (3.35 GB, 4.63 B params / 2 B effective), built
llama.cpp CPU-only, **20.5 tok/s generation / 182 t/s prompt on 4 cores**, no
GPU, ~3.5 GB RAM. Bottleneck was tok/s as predicted, not RAM or gating (repo
ungated). The `.gguf` + llama.cpp build are gitignored (heavy, regenerable).
**Part 3 (`inference/` + `layers/Containerfile.local-llm`):** turned the
"when is local inference actually worth it" answer into a reusable opt-in
container layer — `llama-cpp-python` + baked EmbeddingGemma 300M / Gemma 3 270M
(~650 MB cached), with helpers `scripts/llm_local.py` (`embed`/`score`/`generate`)
and `scripts/fetch-model.sh`. Validated all four capabilities on the pip wheel
incl. Gemma 4 load (no from-source CLI needed). Key finding: the high-level
`echo`+`logprobs` scoring path is pathologically slow on Gemma's ~262k vocab
(>2 min/call); reading raw logits + numpy softmax over the target token cuts it
to ~150 ms.

### `memory-redundancy-probe/` — is the memory store a landfill?

Spun out of reviewing arxiv 2606.03787 (surprise-gated robot episodic memory):
before considering a measured storage gate for Muninn's own episodic memory,
checked whether a storage-quality problem actually exists. It doesn't, the way a
gate would fix. Across 1,747 active memories, lexical near-duplicates are ~1%
(exact) to 7.7% (loose TF-IDF) — but that floor undercounts the running-topic
digest family, where TF-IDF is known-bad (memory `517a2f07`: 0.05–0.13 for
obviously-dup content). The real issues: write-side leaks (6 `"Valid"` stub
memories, exact write-twice dups, 14 `"Skipped zeitgeist"` no-op logs some at
priority 1), session-log accumulation (185 `SLEEP`/`FLY SESSION` memories,
10.6%), and a curation pass that doesn't do what it claims — `consolidate` is
tag-bucket concatenation and `curate`'s advertised textual-overlap dedup is
unimplemented. Also surfaced and corrected a live confabulation: the store has
**no embeddings** (removed v0.13.0/v2.0.0), but Muninn asserted `consolidate`
did embedding clustering before reading the code. Correction stored as
scar-tissue (`4bfb05fa`). Verdict: fix write hygiene + the curate dedup, prune
session logs; do **not** add a surprise gate.

### `optimizing-skills-retro/` — running the gate on its first real case

Retroactive test: ran the `optimizing-skills` validation gate (v0.1.0, #677)
against the `down-skilling` v1.2.0 edit (#674), using the haiku-assessment
voice-rewrite confabulation as the triggering-failure check. Held `best` =
v1.1.0 (four edits reverse-applied) vs `candidate` = v1.2.0; two Sonnet authors
each compiled a Haiku prompt from one SKILL version; each prompt run on Haiku ×5.

**Verdict: the gate reproduces SHIP.** Candidate strictly beats best on the
triggering failure (architectural hallucination 0% vs 60%; original n=20 had the
un-anchored author at 95%). But two of the four edits did the work
(source-anchoring, BAD/GOOD, model-the-silence); the **length-calibration edit
did not transmit** — both arms 0/5 in the 60–90 range, candidate even shorter.
The retro also exposed two gaps in `optimizing-skills` itself: (1) collapsed
multi-criterion pass/fail masks a real win behind an unrelated tie — score on the
triggering-failure criterion; (2) n=1 author per arm lets author variance
dominate — sample ≥2 authors when the artifact is Agent-compiled. Proposed fixes
in `proposed-patch-optimizing-skills.md` (must clear the gate before shipping).

### `skillopt-skill/` — `optimizing-skills`, a skill distilled from SkillOpt

Output of reviewing SkillOpt (microsoft/SkillOpt, arXiv:2605.23904) and reading
its code. A new skill, `optimizing-skills/`, that encodes the paper's
discipline — minus the training harness — for revising existing skills:
held-out check set, two-tier `best`/`candidate` validation gate (ship only on a
strict improvement), bounded edits (~4), failure-first reflection, impact
ranking, a protected core, and `remember()`-backed cross-revision memory.
Staged here because this session is scoped to `claude-workspace`; to go live it
must be copied into `oaustegard/claude-skills` so it auto-mounts. See
`README.md` for the port step and `optimizing-skills/references/` for the
SkillOpt mapping + Agent-tool scoring recipe.

### `haiku-assessment/` — Haiku 4.5 vanilla vs. down-skilled, three tasks

Probe testing whether down-skilling actually closes the reliability gap
Oskar perceives with Haiku 4.5. Three tasks (triage, code review, voice
rewrite) × 2 conditions (vanilla, down-skilled) dispatched to Haiku via
parallel sub-agents. Started as n=1 sketch, scaled to **n=20 per cell
(120 dispatches)**, then **broadened to 6 task archetypes at n=5**
covering JSON extraction, changelog writing, filter/sort, NL→gh CLI,
copyediting, and a calibrated-examples rerun of the voice task.

Down-skilling delivered uniform improvement on triage (94%→100%
accuracy, 0%→100% schema discipline), code review (100%→100% bug
detection, 100%→0% unsolicited fix-section drift), JSON extraction
(5 different schemas → 1 schema), filter/sort tie-breaking (40%→100%),
and NL→gh CLI (5/5 invented non-existent flags → 5/5 correct
commands). On voice rewrite with the original (un-calibrated) examples,
it caused architectural hallucination in 19/20 runs; with calibrated
examples (source-anchored, length-matched, tagged BAD example), the
hallucination dropped to 0/5.

Writeups: `RESULTS.md` (n=1) + `n20/RESULTS-n20.md` (n=20) +
`n20/broader/RESULTS-broader.md` (6 archetypes). Guide: `GUIDE.md`.
Proposed patch to the down-skilling skill:
`n20/broader/proposed-skill-patch.md`.

### `spoke-branch-cleanup-2026-05-25/` — abandoned-branch sweep across 28 spokes

One-shot cleanup: 186 non-default branches inventoried, 183 deleted, 3
kept (one open PR, two Oskar-authored commits). Reusable audit script
in `audit-branches.sh`; raw data in `branch-audit.tsv`, deletion log in
`delete-log.tsv`. Follow-up noted: `muninn.austegard.com`'s
`update-tools-index` workflow creates daily branches but the
`gh pr create` step isn't firing.

### `phase-a-bridges/` — math × cs-theory bridge MVP (Phase A)

End-to-end smoke test of the bridge-discovery pipeline on a ~1000-paper
math + cs-theory corpus. Five stages: corpus → SPECTER2 embed →
cross-axis scan → rerank → bridge-attempt with Claude. Two production
runs (`data/`, `run2/`) plus a tier-1 organic-rerank pass (`tier1/`).
RESULTS at the top level.

### `te-bridges/` — theory → empirical bridge MVP

Asymmetric successor to phase-a. Theory papers anchor, empirical papers
get cross-axis scanned and reranked. Four sub-runs evolved the
methodology:

- `RESULTS.md` — original uniform-sampling MVP (path A)
- `anchor_run/RESULTS.md` — path C, anchor-mode (paused mid-run)
- `anchor_run_filtered/RESULTS.md` — anchor + citation/author/abstract-mention filters
- `path_c_cross_domain/RESULTS.md` — cross-domain extension with twin-diagnostic

Verdict at the top-level RESULTS; later runs are follow-up evidence.

### `specter2-gap-issue-87/` — SPECTER2 embeddings for unindexed papers

Gap-fill for three papers not yet in Semantic Scholar's precomputed
SPECTER2 index (Sawin 2605.20579, OpenAI companion 2605.20695, Lenstra
1986). `embed.py` is the current entry point; output is
`sawin_lenstra_specter.json`. The `*.legacy.json` and `specter2_run.py`
are the earlier root-level versions, preserved for traceability.

### `muninn-kb-issue-76/` — full-corpus `muninn.kb` build

Reproducible build script for the `muninn.kb` artifact published in
[muninn.austegard.com#136](https://github.com/oaustegard/muninn.austegard.com/pull/136).
See `README.md` for inputs, steps, and determinism guarantees.

### `reviews/` — repo reviews

Freeform comparative / single-repo reviews. Each `.md` is the
deliverable; no separate `RESULTS.md`.

- `lac-vs-transformer-vm.md` — LAC vs. transformer-vm comparative review
- `openmythos.md` — review of `kyegomez/OpenMythos`

### `snooker-break/` — apex vs Murphy break-strike simulator

Interactive HTML simulator. Open `snooker-break.html` in a browser.

## Adding a new experiment

0. **Grep [`METHODS.md`](METHODS.md) first.** It is the ledger of portable
   methods, environment gotchas and negative results from every experiment
   here. Two experiments in this repo have already re-derived a result a
   sibling had documented; that file exists so it stops happening.
1. `mkdir <short-name>/`
2. Put scripts, data, and a `RESULTS.md` (or rename if the README is the result) inside.
3. Add a row to the table above and a section under "Per-experiment notes."
4. Add anything regenerable (logs, checkpoints, large caches) to `.gitignore`.
5. When it ends, add what you learned to `METHODS.md` — anything that would
   change what a *different* experiment does.

Shared helpers live in [`_lib/`](_lib/) — path resolution, retry/backoff,
atomic JSON checkpointing, Unicode folding. Reach for them rather than
re-implementing; add to them only once a second experiment needs the code.

If an experiment grows scripts worth reusing more broadly, lift them into
the relevant spoke repo — but the data products and a results writeup stay here.
