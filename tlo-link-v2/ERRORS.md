# Errors

What was wrong during this experiment, how it was caught, and which way it pushed the conclusion.

1. **The first dictionary trainer was worse than v1's hand-made 1.6 KB dictionary.** My own substring scorer (`bench.py`, `train_dict`) made 2 to 16 KB dictionaries that lost 3 to 9% against v1. The benchmark table caught it. Using zstd's COVER trainer for the content fixed it. Direction: before the fix, the run said a bigger dictionary does not help, which was wrong.
2. **The reader at first trimmed letters glued to the end of a link until the deflate stream ended cleanly.** `test_roundtrip.py` caught 30 of 229 links decoding to the wrong document when a word was glued on. Raw deflate has no checksum, so a trimmed body can also end cleanly. The trimming was removed. Direction: the robustness claim got smaller, and the CRC-16 option came out of it.
3. **The baseline uses zlib, not fflate.** On the README's ten links, zlib was 1 to 8 characters shorter than the site's own output. This was not corrected. Direction: v1 looks slightly better than it is, so the measured gain understates v2's slightly.
