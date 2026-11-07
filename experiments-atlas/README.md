# experiments-atlas

A clustered mind map of every experiment in this repo: a d3 radial tree with 217 experiments in 12 themed clusters. Click a cluster to fold or unfold it, search to fade non-matches, click an experiment for its purpose, date, status and a link to its directory on GitHub.

Page: [`web/index.html`](web/index.html) (live at https://austegard.com/experiments/experiments-atlas/).

## Data

- `clusters.py` holds the cluster assignment: one primary cluster per experiment directory.
- `data.json` holds the per-experiment purpose, date, status and secondary cluster.
- `tpl.html` is the page with a `__DATA__` placeholder; `python3 build.py` inlines `data.json` into it and writes `web/index.html`. To add or move an experiment, edit `clusters.py` and `data.json`, then run `build.py`.
- `_lib/` is left off because it is shared helper code and not an experiment.

## Known gaps

- It is a snapshot of 2026-11-07. A new experiment is missing until it is added to `clusters.py` and `data.json` and `build.py` is run.
- 11 experiments are missing from that README index (`bitplane-scan`, `bsky-image-search`, `bsky-list-ai-share`, `embed-coord`, `lantern-eel`, `leaf-wasm`, `monad-chat`, `paper-watch`, `prism-audio`, `refractometer`, `tilt-racer`); their purposes come from each folder's own README or RESULTS file.
- 8 experiments have no date: `bitplane-scan`, `refractometer`, `monad-chat`, `paper-watch`, `bsky-list-ai-share`, `lantern-eel`, `prism-audio`, `tilt-racer`.
- `tilt-racer` has no README, so its purpose is a guess.
- `phase-a-bridges` and `te-bridges` are paper bridge-discovery pipelines and sit under Search and knowledge bases because no cluster fits them well.
- Most purpose lines are trimmed automatically from the repo's top-level `README.md` index; about 15 were rewritten by hand, and some read clipped.
- On a phone the clusters open folded and their labels are about 8px until you zoom or unfold one.
