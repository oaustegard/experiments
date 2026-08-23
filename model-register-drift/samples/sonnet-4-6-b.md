# Why we removed our cache.

In March, we shipped a Redis read-through cache in front of our primary Postgres queries for Orion, our order-management service. By July, we had ripped it out. Our p99 latency was 340ms with the cache and 180ms without it.

Here’s what happened, and what we should have measured before we started.

## The setup

Orion handles roughly 4,000 requests per minute at peak. Most of those requests hit a handful of “hot” queries: active orders by customer, order status by order_id, and a product-catalog join we fondly call “the monster join.” Our Postgres p99 was sitting at around 180ms, and under load testing we’d seen it climb to 400ms. The instinct was obvious: cache the reads.

We stood up a Redis cluster (two primary nodes, two replicas) and wrapped the three hot query paths in a read-through cache with a 60-second TTL. Straightforward stuff. Cache miss → query Postgres → write result to Redis → return. Cache hit → return from Redis.

In staging everything looked great. Hit rates above 90% on the monster join. Redis round-trip at 2ms. We shipped it.

## What we saw in production

Four months later, our SLOs were quietly getting worse. P50 stayed roughly the same — around 40ms. P95 was about the same — around 120ms. But p99 had crept up from 180ms to 340ms, almost double.

We didn’t notice immediately because the median looked fine. It took a new engineer on the team running a latency percentile breakdown during an unrelated incident to surface it.

## What was actually happening

Four things conspired to make p99 worse:

**1. Cache stampedes on TTL expiry.** With a fixed 60-second TTL and a high request rate, a large cohort of keys expired nearly simultaneously. When a popular order’s cache entry expired, dozens of concurrent requests would all miss, all race to query Postgres, and all attempt to write back to Redis. Postgres got hammered in synchronized bursts. Those bursts showed up as p99 spikes.

**2. Hot-key saturation in Redis.** A handful of our highest-traffic customers had order records that were read thousands of times per minute. A single Redis primary was fielding the bulk of that traffic. At peak, one key was responsible for ~8% of reads on its primary node. Redis is single-threaded per core; when that node’s CPU hit 85%, all requests waiting on it stacked up. P99 climbed.

**3. Serialization and deserialization cost.** The monster join returned large nested objects — sometimes 80KB of JSON. Deserializing 80KB of JSON on every cache hit took 12–15ms in our Python service. For p50 requests hitting warm, non-contended keys, that was fine. For p99 requests hitting a slightly slow deserialization pass while also waiting on a busy Redis connection pool, it stacked.

**4. Retry amplification on Redis timeouts.** We had a 10ms timeout on Redis gets with two retries. When Redis was under load, those timeouts fired, and the retry logic added up to 30ms before falling through to Postgres. We had essentially introduced a worst-case floor of 30ms + Postgres latency on any request that hit a struggling Redis node.

The 90th percentile of users never saw any of this. The 99th percentile saw all of it, all at once.

## What we should have measured first

Before adding a cache, we should have answered these questions with data:

**Is Postgres actually the bottleneck?** We assumed the database was slow. We hadn’t instrumented query latency separately from service latency. Turns out a chunk of our p99 was coming from connection pool exhaustion, not Postgres execution time. Caching query results doesn’t help a saturated connection pool.

**What does our key distribution look like?** We had no idea that 5% of our customers represented 60% of our read volume, concentrated on a tiny set of keys. Hot-key analysis before we shipped would have predicted the Redis saturation problem.

**What’s our actual cache miss pattern?** We assumed a smooth, distributed TTL expiry. We never modeled synchronized expiry under our actual request patterns. Jittering the TTL by ±20% would have been a one-line fix that we didn’t think to add.

**What’s the p99 cost of the cache path, not just the p50?** We benchmarked average cache hit latency (2ms). We never measured the p99 of the Redis call under production-like load with hot keys and concurrent writes. That number was closer to 45ms.

**What happens when the cache is degraded?** We had a fallback to Postgres on cache miss but not on timeout. So a slow Redis was worse than no Redis.

## What we did instead

We removed the cache and fixed the actual problems. We added connection pool monitoring and right-sized the pool. We added query-level caching at the database layer using materialized views with incremental refresh for the monster join. P99 is now at 95ms — better than before we ever added Redis.

Caching is not free. Every cache is a distributed system with its own failure modes, and those failure modes appear in your tail latency. Measure your tail, not your mean, before you reach for Redis.
