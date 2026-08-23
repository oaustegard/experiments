# Why we removed our cache

In March we put a read-through Redis cache in front of the hottest query path in `ledger-api`. In July we took it out. Median latency improved both times we touched it — that was the trap.

The numbers, at a steady 6,200 rps:

| | p50 | p95 | p99 |
|---|---|---|---|
| Before cache | 5.2 ms | 14 ms | 38 ms |
| With cache (87% hit rate) | 3.9 ms | 21 ms | 96 ms |
| After removal | 5.4 ms | 15 ms | 41 ms |

We shaved 25% off the median and paid for it with a 2.5x worse tail. Here's the mechanism, because it's not specific to us.

## A cache doesn't make things faster. It makes some things faster and everything else slower.

Our hit path was a Redis `GET` plus JSON decode: ~1.3 ms. Our miss path was `GET` (0.4 ms) → Postgres (5.2 ms) → JSON encode (0.9 ms) → `SETEX` (0.4 ms) ≈ 6.9 ms. So a miss cost about 33% more than the uncached query it replaced.

Now do the percentile arithmetic. With an 87% hit rate, every request above the 87th percentile is a miss. Your p99 is no longer the 99th percentile of *all* requests — it's roughly the 92nd percentile of the *miss* distribution, and the miss distribution is strictly worse than the uncached one.

That generalizes to a rule worth writing on a wall: **a read-through cache can only improve your tail if it also makes misses cheaper.** It doesn't. It makes them more expensive by exactly the cost of the cache round trips. Read-through caching is a p50 optimization sold as a latency optimization.

## We took away Postgres's cache to build our own

`ledger-primary` had 24 GB of `shared_buffers`. The reason the uncached p50 was 5.2 ms and not 50 ms is that the hot rows were already in memory. Postgres had a perfectly good cache; we just couldn't see it on a dashboard.

Then we started absorbing 87% of reads. A buffer pool is a popularity ranking, and we had removed the popular queries from Postgres's view. Its buffer pool filled with cold, one-off scans. Cache hit ratio drifted from 99.4% to 96.1% over four months, and the Postgres component of our miss path went from 5.2 ms to about 9 ms.

That drift is why we didn't blame the cache. The regression didn't ship with the deploy; it arrived over a quarter, looking like organic growth.

## Two systems in series means two sets of tails

Redis is single-threaded, and ours was doing `BGSAVE` on a 30 GB instance. The fork stalled the event loop for 300–400 ms a few times an hour. That's a rounding error on availability and a catastrophe for p99.

Worse, our client timeout was 250 ms, and on timeout we fell through to Postgres. So a slow Redis didn't remove latency from the request — it *added* 250 ms to it. Failure modes are latency, not just availability, and almost nobody budgets them that way.

And then stampedes: a hot key expiring under a 5-minute TTL produced ~900 concurrent misses on one query, which saturated pgbouncer's 120 slots, which queued everything else. We added single-flight and it helped. But we'd now written a distributed lock to protect a database that hadn't needed protecting.

## What we should have measured first

**Where the tail actually comes from.** We never profiled it. When we finally did, 60% of our p99 was pgbouncer queue wait during write bursts, not query execution — a cache doesn't touch that. The real fix was an N+1 on one endpoint fanning out 40 sequential lookups. Batching it took p99 from 41 ms to 18 ms, which is more than the cache ever promised.

**The backing store's own hit ratio.** If Postgres is serving 99% from `shared_buffers`, Redis is a second copy of memory you already own, one network hop further away.

**A miss-path budget, on paper, before writing code.** Three numbers: hit latency, miss latency, expected hit rate. If `miss_latency > uncached_latency` — and for read-through it always is — then you can compute your new p99 before you build anything. We could have known this in an afternoon.

**What the cache does to the thing behind it.** Model the buffer-pool effect. Caches don't sit beside your database; they change its workload into the worst possible mix.

## We didn't remove every cache

We kept the one in front of our address-validation vendor: 300 ms p50, rate-limited, occasionally down. That cache is worth its complexity because the origin is slow, expensive, and unreliable.

That's the actual heuristic. Cache when the backing store is slow, costly, or flaky — not when it's fast and you'd like it faster. A well-indexed Postgres query served out of RAM is not an expensive origin. Putting a network hop in front of it doesn't buy speed; it buys a second system that can be slow in new ways, and your users only ever feel the slow ways.
