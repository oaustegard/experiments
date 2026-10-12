# Finalizing the numbers

The writeup's numbers (experiments/haiku-cc/RESULTS.md) were taken while the hub swarm was still running
(`runs/swarm-r19.log`, finishing reject-gcc-dg and the re-opened tail). Once it stops
(`pgrep -f 'swarm/run.py run'` prints nothing), refresh them with:

    /opt/anavenv/bin/python analysis/stats.py

It fetches the sibling ledgers (`fed` hcc/stats/s1..s3) and the escalation results
(hcc/exp/escalation{,2,3}), then rewrites `analysis/out/` (metrics.json, comparison.md, five PNGs).
`metrics.json` carries `as_of` and `hub_running`; when `hub_running` is false the wall clock ends at
the hub's last ledger or merge event, not at the time of the refresh.

Then copy `analysis/out/*.png` to experiments/haiku-cc/figures/, replace the "as of" numbers in
RESULTS.md (Answer, Findings 1, the comparison and node tables) from `comparison.md`, and refresh the
snapshot as experiments/haiku-cc/SNAPSHOT.md describes.
