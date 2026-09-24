[no-context]
You are a participant in a document-QA experiment. Work through the batch file
`{EXP}/runs/{RUN}/batch.md`. Each entry is one long document rendered as a
single PNG "sheet": its pages are drawn as small labelled thumbnails (P1, P2,
...), usually too small to read reliably. You can get the full text of a page
from a tool, but only after you commit to an answer from the image alone.

First, run once:
  python3 {EXP}/lens.py hello {RUN} --model "<your exact model id, as stated in your system prompt>"

Then, for each document in the batch, in order:
1. View the sheet with the Read tool (Read the PNG path). Look at it directly.
2. Commit an answer from the image alone, plus your ranked guess of the page(s)
   most likely to hold the evidence (1 to 3 pages, best first). Give a best
   guess even when you cannot read the text:
     python3 {EXP}/lens.py commit {RUN} <doc> --answer "<short answer>" --pages <p1,p2,...>
3. Expand pages one at a time to read their full text, up to 3 per document.
   Choose each next page based on what you have read so far. Stop expanding as
   soon as you can answer:
     python3 {EXP}/lens.py expand {RUN} <doc> <page number>
4. Record your final answer:
     python3 {EXP}/lens.py final {RUN} <doc> --answer "<short answer>"

Answers are short spans in HotpotQA style: an entity name, a number, a date or a
short phrase, not a sentence.

Rules, which the measurement depends on:
- The sheet image is the only view of the document besides `expand`. Do not
  crop, zoom, enhance, OCR or otherwise process the PNG with code, and do not
  open any other file under `{EXP}` (no `data/`, no `pages.bin`, no other runs,
  no source code beyond running `lens.py` as shown).
- Answer from the document. No web search, no other tools beyond Read and the
  lens.py commands.
- commit and final are write-once per document; lens.py enforces the order.

When the batch is done, reply with one line: your model id and the number of
documents finalized.
