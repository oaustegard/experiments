# Errors

What was wrong on the way to `RESULTS.md`, in the order it was caught, with the
direction each one pushed the conclusion.

**1. Read a transposed argument list as a library failure.** The first version of
`test_every_supported_width_round_trips` called `unpack(packed, 997, bits)`. The real
signature is `unpack(packed, bits, n_values)`, so `997` arrived as the bit width and five
of six parametrized cases failed with `Bit width 997 is not supported`. For one reading of
the output this looked like the `SUPPORTED_BITS` refactor having broken `unpack`. Caught by
reading the error text rather than the failure count. Direction: would have manufactured a
regression that did not exist and blamed the refactor for it.

**2. Framed the `"none"` rotation as uncovered before checking.** The meta-oracle correctly
reported that `test_pq_and_npz_round_trip_rotation` parametrizes `["haar", "rht"]` and never
sees `"none"`. The first draft of the finding said `"none"` had no round-trip coverage. A
perturbation of `save_pq`'s rotation byte then failed
`tests/test_scalar_mode.py::TestSerialization::test_pq_round_trip`, which does cover it,
because scalar mode uses `rotation="none"`. Direction: would have overstated the tool's
find. `RESULTS.md` now says coverage by coincidence rather than by enumeration, and the
perturbation table marks that row co-detected.

**3. Took the third perturbation as load-bearing when it is not.** Adding `5` to
`SUPPORTED_BITS` fails the new totality oracle, but it also fails the pre-existing
`test_bits_validation`, which asserts that 5 is refused. Only two of the four perturbations
are cases where the whole suite stays green. Recorded as such in both `RESULTS.md` and
`remex/lib.spec.md`.

**4. Wrote a parity oracle the analyzer would not accept, and briefly read that as the
analyzer being right.** The first version looped `sorted(persistable)`, an alias for the
declared domain, and coherence refused the claim. The oracle was complete and correct;
the analyzer requires the loop to name the declared domain literally. Direction: nearly
recorded a false failure as a legitimate catch, which would have made the tool look more
precise than it is.

**5. Ran the coherence test suite with a 120-second foreground timeout.** It takes 3m04s,
so the call was moved to the background by the harness. No harm, but it is the shape the
hub `CLAUDE.md` warns about; the recovery was a single non-looping `tail` on the output
file rather than a poll loop.

## Base rate

Five errors over one session's work. Four of five were caught by reading output rather than
by a second tool, and three of the five pushed toward overstating the tool's value, which is
the direction to watch when the task is "assess this thing."
