The draft was already clean, so I made two edits. The `--bits` row said "Bits per dimension", but "per dimension" isn't in the facts, so I changed it to "Bit width used for the search". Confirm the original wording if it's accurate. In the rescoring paragraph, the last sentence restated the one before it, so I rewrote it to tell the reader what to do: build the index with `--keep-float` to get rescoring.

I kept the "Searching an index" header. The task names it, and the voice signature says headers in docs stay noun-phrase labels, so the linter's flag on it doesn't apply. The draft also correctly leaves out the changelog material a first-time reader doesn't need: the removed `--rerank` flag, the old 2-bit default and TSV output, and the trailing-comma bug fix. It describes each as it works now. I found no contradictions in the facts.

The structural review, newcomer read and diff check weren't available, so I did their checks by eye. No independent reader saw the text.
