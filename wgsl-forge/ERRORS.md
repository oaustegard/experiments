# Errors

What was wrong, how it was caught, and which way it pushed the conclusion. Seven errors in
one build: four were caught by a check written for this experiment (#1, #2, #6, #7), one by a
real-world input (#3) and two by naga (#4, #5).

1. **Curves flattened at the SVG's own size.** Round 1 turned curves into line segments
   within 0.2 pixels at the SVG's nominal size (24 px for an icon). Rendered at 512 px, the
   chords cut inside every convex curve: an ink ratio of 0.982–0.984 that
   more resolution could not fix. `tests/scale_check.py` caught it, by measuring ink at several sizes. It
   had made the compiler look worse than it is (median PSNR 28.93 dB, against 34.74 dB after
   the switch to quadratic segments within 0.02 nominal px). The new bound is still tied to
   the SVG's size, so it holds up to about 50× magnification, not at any zoom.
2. **Missing parentheses around an animated value.** `Circle(r=44 + 3*sin(T*2))` compiled to
   `length(q - c) - 44+3*sin(u.t*2)`. Frames at t=0 and t=4.7 matched closely, because `sin`
   was near zero there; t=1.3 diverged visibly (27.9 dB, a smaller sun). `forge.scene.wg` now
   parenthesises any compound expression. Lesson: test animated output at times where the
   animated term is large.
3. **Packed SVG arc flags.** `a8.77 8.77 0 014.907-2.77` means flags 0 and 1, then x=4.907.
   The first parser split a packed token only when its second character was `0`, `1` or `.`.
   `014.907` passed that test, giving flag 0 and leaving `14.907`, whose second character
   `4` failed it, so the second flag was read as the number 14 and the coordinates shifted. Caught
   by two Simple Icons logos (Rust, Firefox) that failed to parse; no hand-written test SVG packed flags that way.
4. **Type aliases for generic constructors.** The first alias pass aliased `vec2`, `array`
   and `mat2x2` (constructors that infer their template), which are not types on their own.
   naga rejected 9 of the 86 outputs. Fixed by aliasing only complete types.
5. **naga's alias-in-template quirk.** After fix 4, naga 30.0.1 rejected
   `var<private> x: array<i32, 2> = array<e, 2>(...)`, where `e` aliases `i32`, with
   "expected `array<i32, 2>`, but got `array<i32, 2>`". Tint accepts it. The alias pass now
   leaves template arguments alone. Recorded in METHODS.md.
6. **Test bug in the template test.** The filter `t.text in "<>"` also matched the empty EOF
   token, so a correct lexer failed the test. Fixed the test.
7. **Baseline exits 0 on failure.** `wgsl-minifier` prints its error and exits 0 without
   writing the output file; the first comparison script crashed reading a file that did not
   exist. The benchmark now counts a missing output as a rejection, which is how 36 of 86
   are classified.

Two of the seven (#1, #2) would have changed a reported number had they shipped. Both were
caught only by comparing against an independent renderer at an extra operating point:
another size for #1, a third time for #2.
