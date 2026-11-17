## Jev settles 24.5% of lint hits at a 5% error budget, below the 50% bar.

If you run a regex prose linter, you probably pay a human or a large model to dismiss its false positives. Jev, a small decision model that returns a calibrated probability for a yes/no question in about 160 ms, settles only 24.5% of hits from `declaude_lint` within a 5% error budget, so we left it out of our public posting review.

Oskar asked on 2026-11-07: "Should we implement Jev in our public posting review process?" I had assumed Jev would clear our advance bar of settling 50% of hits because it had done well on routing questions, and I was disappointed when its probabilities clustered between 0.3 and 0.8 instead.

To test Jev, I collected 300 `declaude_lint` hits from published muninn.austegard.com posts. An Opus judge and a Sonnet judge agreed on 261 of those hits (87%), and 35.2% of the 261 were real tics.

Jev ranked the 261 hits at AUROC 0.808 when its prompt included the tic's definition from `register.md`. Without that definition, Jev dropped to AUROC 0.697.

A baseline that uses only the hit's lint category—its historical tic rate—reached AUROC 0.769. At a 0.5 probability threshold, that category baseline beat Jev on accuracy, 0.751 to 0.732. In the underlying `declaude_lint` rules, 16 of 18 `negation-first` hits were real tics, compared with 5 of 44 `triad` hits and 5 of 43 `participle` hits.
