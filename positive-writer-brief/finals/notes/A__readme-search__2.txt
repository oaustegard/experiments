The draft was already clean, so I made only small edits. It describes the tool as it works now and doesn't lean on the changelog. I kept the headers as plain noun-phrase labels, which is what the voice signature asks for in docs. That includes the "Searching an index" header that the lint flagged, because it's the section name the task assigned.

I removed three things the facts don't support:
- **"Returns the quantized scores":** the facts only say the search skips rescoring, so I cut that phrase.
- **`"query": 1` in the example:** the facts don't say whether line numbers start at 0 or 1, so I put back the `<line number>` placeholder. Please confirm which it is.
- **What `ids` holds:** the draft said `ids` held "the matched vectors." I changed it to say it holds the IDs of the matched vectors.

I also left out the removed `--rerank` flag on purpose. A first-time reader never saw it, so it belongs in the changelog. I found no contradictions in the draft.
