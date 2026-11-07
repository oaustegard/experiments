# experiments-atlas

A clustered mind map of every experiment in this repo: a d3 radial tree with 217 experiments in 12 themed clusters. Click a cluster to fold or unfold it, search to fade non-matches, click an experiment for its purpose, date, status and a link to its directory on GitHub.

Page: [`web/index.html`](web/index.html) (live at https://austegard.com/experiments/experiments-atlas/web/).

## Data

- `clusters.py` holds the cluster assignment: one primary cluster per experiment directory.
- `data.json` holds the per-experiment purpose, date, status and secondary cluster.
- `tpl.html` is the page with a `__DATA__` placeholder; `python3 build.py` inlines `data.json` into it and writes `web/index.html`. To add or move an experiment, edit `clusters.py` and `data.json`, then run `build.py`.
- `web/og.png` is the 1200x630 link-preview image, rendered from the page by `make_og.js` (needs Playwright). Rerun it after a visible change to the map.
- `_lib/` is left off because it is shared helper code and not an experiment.

## Known gaps

- It is a snapshot of 2026-11-07. A new experiment is missing until it is added to `clusters.py` and `data.json` and `build.py` is run.
- Dates are each directory's first commit date, which can differ from when the work was started.
- `tilt-racer` has no README, so its purpose is a guess.
- `phase-a-bridges` and `te-bridges` are paper bridge-discovery pipelines and sit under Search and knowledge bases because no cluster fits them well.
- Most purpose lines are trimmed automatically from the repo's top-level `README.md` index (or from the folder's own README or RESULTS file where the index has no row); about 15 were rewritten by hand, and some read clipped.
- On a phone the map opens with every cluster folded. Tapping a cluster shows only that cluster, with labels at about 9px; pinch to zoom further. The experiment card appears at the bottom of the screen after a tap and closes with ×.
