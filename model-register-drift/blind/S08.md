# Why we removed our cache.

Last spring we put a Redis read-through cache in front of the Postgres queries behind our `accounts-api`. Four months later we took it out. The reason was simple and a little embarrassing: p99 latency was *worse* with the cache than without it. Median latency looked great the whole time, which is exactly why it took us four months to notice.

Here's how that happens, and what we should have measured before we wrote a line of caching code.

## The setup

`accounts-api` serves profile and entitlement lookups — the kind of read-heavy endpoint everyone reaches for a cache on. Requests fanned out to three or four indexed Postgres queries against our primary. Under normal load, the endpoint ran about 6 ms at p50 and 34 ms at p99.

We added a standard read-through layer in Redis: check the cache by key, return on hit, and on a miss query Postgres, populate the key with a 5-minute TTL, and return. Textbook. We shipped it, watched p50 drop to 3 ms, declared victory, and moved on.

## What the average hid

The mean and the median both improved, because most requests hit the cache and a cache hit is genuinely faster than four Postgres round trips. But averages are the wrong instrument for a cache. A cache doesn't make every request faster — it makes *some* requests much faster and *every other* request a little slower. The tail is where the "little slower" requests pile up.

Three things were happening at p99, and none of them showed up in the average.

**Misses pay for the lookup twice.** A cache miss isn't free — it's the full Postgres query *plus* a Redis round trip you didn't have before. Our hit rate was about 78%, which sounds fine until you look at *which* requests missed. Our key distribution had a long tail: a large population of accounts that get read once a day. Those requests essentially never hit the cache, so for them we'd added a guaranteed extra network hop to a workload that was already the slowest. The p99 request is, almost by definition, a tail-of-the-distribution request — the exact request most likely to miss.

**Redis has its own tail.** A single Redis instance is fast at p50 and lumpy at p99. Cross-AZ network jitter, a slow `MGET` behind a big pipeline, a fork for `BGSAVE`, a brief failover — any of these adds a few milliseconds, and now they're on the critical path of *every* request, hits included. We had taken Postgres's tail and *added* Redis's tail on top of it instead of replacing it.

**Stampedes.** When a popular key expired, dozens of concurrent requests missed simultaneously and all rushed Postgres at once. That burst drove up Postgres latency for everything, including requests that had nothing to do with the expired key. We had no request coalescing, so every TTL expiry was a tiny thundering herd.

Put together: hits got faster, but the slow requests got slower and more numerous, and the slow requests are what p99 measures. p99 drifted from 34 ms to 51 ms. We were serving a better median on top of a worse experience for the customers most likely to be annoyed.

## What we should have measured first

The uncomfortable finding, once we pulled the cache and re-benchmarked, was that Postgres alone did fine. The working set fit comfortably in `shared_buffers`, so the queries we were "protecting" were already served from Postgres's own memory. We had cached data that was effectively already cached — and paid a network hop and a tail-latency penalty to do it.

Before adding a cache, these are the numbers that would have told us not to:

- **Is Postgres actually the bottleneck?** Check the buffer cache hit ratio and `pg_stat_statements` for the target queries. If they're already served from memory in single-digit milliseconds, a cache mostly adds a hop.
- **What's the realistic hit rate for *your* key distribution?** Not the aggregate — the hit rate weighted by how requests actually spread across keys. A long tail of rarely-read keys can make the miss path dominate the experience even at a "good" overall hit rate.
- **What does a miss cost versus a hit?** If a miss is meaningfully slower than the un-cached path, model the blended p99, not the p50.
- **Measure p99 and p99.9, before and after — on the same traffic.** A cache is a tail-latency bet. Judge it on the tail.
- **Do you need stampede protection?** If a single key can be requested concurrently, you need coalescing or early recomputation, or every expiry is a small outage.

Caches are wonderful when the backing store is genuinely the bottleneck and your access pattern is genuinely skewed toward hot keys. Ours wasn't, on either count. The lesson wasn't "caches are bad." It was: measure the tail, measure the miss, and prove the thing you're caching isn't already in memory — before you add a second system to the critical path of every request.
