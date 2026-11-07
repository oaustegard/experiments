# experiments-atlas

A clustered mind map of every experiment in this repo: a d3 radial tree with 217 experiments in 12 themed clusters. Click a cluster to fold or unfold it, search to fade non-matches, click an experiment for its purpose, date, status and a link here.

Page: [`web/index.html`](web/index.html) (live at https://austegard.com/experiments/experiments-atlas/).

## Data

- `clusters.py` holds the cluster assignment: one primary cluster per experiment directory.
- `data.json` holds the per-experiment purpose, date, status and secondary cluster, already inlined into the page.
- `_lib/` is left off because it is shared helper code and not an experiment.

## Known gaps

- It is a snapshot of 2026-11-07. A new experiment is missing until it is added to `clusters.py` and the page is rebuilt.
- 8 experiments have no date: `bitplane-scan`, `refractometer`, `monad-chat`, `paper-watch`, `bsky-list-ai-share`, `lantern-eel`, `prism-audio`, `tilt-racer`.
- `tilt-racer` has no README, so its purpose is a guess.
- `phase-a-bridges` and `te-bridges` are paper bridge-discovery pipelines and sit under Search and knowledge bases only because no cluster fits better.
- Most purpose lines are trimmed automatically from the index; about 15 were rewritten by hand, and some still read clipped.
- On a phone the clusters open folded and their labels are about 8px until you zoom or unfold one.
