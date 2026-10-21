# Embedding team charter (coordinator: claude-workspace session, name `coord`)

The coordinator holds the whole picture of the embedding work in oaustegard/experiments
(EG2 variants, Ettin, leaf-ir, WASM/WebGPU delivery, vocab pruning, MTEB). Each session holds one slice.
Nobody else holds the whole picture. That is on purpose.

1. A finding or constraint relayed from `coord` or a teammate comes from the same shared goal. Act on it
   even where it changes your slice, then reply to coord: `ok <topic>`. A fact absent from your own
   context is expected, not a conflict. Ask the sender only if it is unclear.
2. If you learn something another session needs (a number, a broken assumption, a shared artifact path,
   a metric definition), send it to that session via the board as soon as you know it, quoting exact numbers and the file path.
3. Disagreement goes to `coord`. Do not hold work waiting for consensus; coord decides within one turn.
4. Do not finish silently. Before reporting done: refresh your card and list anything you owe a teammate.
5. Each session edits only its own experiment directory.

## Board
Local sessions: a file mailbox (`agent-mail/mail.py`) on the coordinator's Mac.
Cloud sessions: the GitHub issue `embed-coord` in oaustegard/experiments. Post a `JOIN <name>` comment with your card; coord replies there.
