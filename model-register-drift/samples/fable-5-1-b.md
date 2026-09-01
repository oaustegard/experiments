# Why we removed our cache

Four months ago we put a Redis read-through cache in front of the Postgres database that serves our product catalog API. Last week we deleted it. Median latency was better with the cache. p99 latency was worse, and p99 is the number our customers actually feel. This is a write-up of what happened, because the failure mode was not obvious to us going in and I suspect it is not obvious to a lot of teams.

## What we built

The catalog service handles roughly 4,000 requests per second at peak. The hot path is a single query: fetch a product record, joined with its current pricing and inventory, by ID. Before the cache, that query ran against a `db.r6g.2xlarge` Postgres primary with a working set that fit comfortably in `shared_buffers`. Latency at the API layer looked like this:

- p50: 3.1 ms
- p95: 9 ms
- p99: 18 ms

We added a read-through layer: check Redis by product ID, on miss query Postgres, write the result back with a 5-minute TTL. Cache invalidation happened on writes via a `LISTEN/NOTIFY` consumer that issued `DEL`s. Redis was a single `cache.r6g.large` node in the same AZ. Hit rate settled at about 91%.

## What the dashboards said

Two weeks in, p50 had dropped to 0.9 ms and we congratulated ourselves. Then we started getting tickets about intermittent slow page loads that we couldn't reproduce. When we finally pulled the full latency histogram instead of staring at the median, it looked like this:

- p50: 0.9 ms
- p95: 11 ms
- p99: 41 ms
- p99.9: 130 ms

p99 had more than doubled. The distribution had gone bimodal: a tight cluster of fast hits and a long, fat tail of misses that were slower than the pre-cache baseline.

## Why the tail got worse

**The miss path is strictly more expensive than the old path.** A miss is a Redis round trip (~0.4 ms), then the Postgres query (~3 ms), then serialization and a Redis `SET` (~0.6 ms with our payload sizes). Every miss paid about 1 ms of overhead on top of the original query. That alone doesn't explain a 41 ms p99, but it means the cache could only ever be a net win if the miss rate was low *and* the misses were spread evenly. Neither held.

**Misses were correlated, not random.** Our TTL was a flat 300 seconds. After every deploy, and after every Redis failover during maintenance, the cache was cold and refilled in a burst. Worse, popular products that got cached at roughly the same time expired at roughly the same time. We'd see 30 to 60 requests for the same SKU arrive within a few milliseconds of expiry, all miss, all hit Postgres, all issue the same `SET`. Postgres handled the load fine in aggregate, but the connection pool (pgbouncer, 40 server connections) briefly queued, and queued requests are where the 100 ms+ outliers came from.

**Redis is single-threaded and our values were big.** A fully hydrated product record with variants and localized descriptions serialized to 40–90 KB of JSON. A handful of catalog-wide products were over 400 KB. Redis handled these fine in isolation, but a `SET` of a 400 KB value takes real time on the event loop, and every other request on that node waits behind it. We measured Redis-side p99 at 12 ms during peak, entirely driven by a few dozen large keys. The cache was adding latency to *hits*, not just misses.

**Postgres was already a cache.** This is the part that stung. Our working set was about 6 GB. `shared_buffers` was 8 GB and the OS page cache had another 40 GB. Buffer hit ratio was 99.7% before we ever touched Redis. We were putting a network cache in front of a database that was already serving from memory. The 3.1 ms median wasn't disk time; it was planning, the join, and row serialization. Redis couldn't beat that by enough to justify its own tail.

## What we did instead

We deleted the Redis client, the invalidation consumer, and about 600 lines of code. Then we spent a week on the query itself:

- Added a covering index so the pricing lookup became an index-only scan.
- Switched to prepared statements through pgbouncer to cut planning time.
- Trimmed the response payload; we'd been shipping 30 KB of description text on an endpoint that mostly fed thumbnail grids.

Current numbers:

- p50: 2.2 ms
- p95: 6 ms
- p99: 12 ms
- p99.9: 28 ms

p50 is slower than with the cache. p99 is a third of what it was, and nobody has filed a "sometimes it's slow" ticket in three weeks.

## What we'd tell ourselves four months ago

Look at the whole histogram before and after. A cache trades a better median for a worse tail almost by construction, because misses are more expensive than the uncached path. That trade is only worth it if the tail is small, uncorrelated, and cheap. Ours was none of those.

Check whether your database is already serving from RAM. If it is, the win you're imagining is mostly gone before you start.

If you do cache, jitter your TTLs, cap value sizes, and put a per-key lock or coalescing layer on the miss path so a stampede costs one query, not sixty.

And be willing to remove infrastructure. The cache wasn't wrong for the problem we thought we had. It was wrong for the problem we actually had, and it took us four months to notice because we were watching the wrong percentile.
