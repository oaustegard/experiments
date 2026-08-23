# Sample provenance

Every sample except one arrived as a comment on
[claude-workspace#244](https://github.com/oaustegard/claude-workspace/issues/244),
posted by the writing session itself. `fetch_samples.py` regenerates those from
the issue and names them in comment order, so the `-a`/`-b` suffixes are stable.

`opus-5-b.md` is the exception. That session finished the post and then refused
to deliver it, so Oskar pasted the text into the parent session on 2026-08-23
and it was written to disk by hand. It is not on issue #244 and
`fetch_samples.py` will not reproduce it. The text is verbatim as pasted.

The refusal is itself data, and it is the third distinct delivery refusal from
the second round: Opus 5 asked for GitHub connector authorization, Opus 4.8
declined a "proxy workaround", and Sonnet 5 judged the task to be prompt
injection and held out for a human after a provenance poke. None of the three
had any trouble writing the post.
