# Why We Removed Our Cache

Four months ago we put a Redis read-through cache in front of the Postgres queries backing `orders-api`, our order-lookup service. Last week we ripped it out. The cache wasn't broken — it was doing exactly what we asked it to do. The problem is that what we asked it to do wasn't what we actually needed, and nobody checked p99 latency until it was too late to pretend otherwise.

## The setup

`orders-api` serves about 4,000 requests per second at peak, mostly `GET /orders/{id}` and `GET /customers/{id}/orders`. The Postgres primary was healthy — median query time around 3ms, p99 around 40ms — but our on-call load kept climbing, and someone did the math: at 4,000 RPS, even a 90% cache hit rate would cut primary read load by 3,600 queries per second. That felt like an easy win, so we added a Redis read-through layer: check cache, on miss hit Postgres, populate cache, return.

We shipped it, watched throughput on the primary drop as predicted, and moved on. Cache hit rate settled around 87%. Everyone was happy for about six weeks.

## What we didn't measure

We measured hit rate and primary query volume. We did not measure the latency distribution of cache misses separately from hits, and we did not measure what happens to a request when Redis itself is slow or unreachable.

That second gap is what got us. A read-through cache doesn't remove the database dependency — it adds a new dependency in front of it. Every single request, hit or miss, now pays the cost of a network round trip to Redis before anything else happens. On the happy path that round trip is fast: our Redis cluster (`cache-orders-01`, three nodes, `cache.r6g.large`) averaged 0.4ms per GET. Fine.

But averages hide the story p99 tells. Three things were compounding under the average:

**1. Cache misses got slower, not faster.** A miss now meant: query Redis, miss, query Postgres, then write back to Redis — synchronously, on our client's default settings. A miss went from "one 3ms Postgres query" to "Redis round trip + Postgres query + Redis write." With an 87% hit rate, 13% of requests — 520 per second — were now slower than before, not faster.

**2. Redis p99 was much worse than Redis average.** GC pauses in the client and, mainly, key eviction contention during nightly batch jobs that wrote large blobs into the same cluster for an unrelated service. Redis p50 was 0.4ms. Redis p99 was 22ms — most of our entire latency budget, spent on the thing that was supposed to make requests faster.

**3. Failure mode was correlated, not independent.** When Redis had a bad thirty seconds — a failover, a slow node — every request stalled waiting on it, including ones that would have been served by Postgres in 3ms if we'd just asked Postgres directly. We'd taken a system where Postgres p99 was 40ms and independent per-query, and wired it in series with a second system whose own p99 could spike to hundreds of milliseconds during incidents. Serial dependencies multiply your tail risk; they don't average it.

## The number that made the decision

We finally pulled `orders-api` p99 latency before and after the cache, plotted over the same weeks of traffic, after a customer complaint about intermittent slow page loads sent us digging. Before the cache: p99 41ms. After the cache, steady state: p99 68ms. During Redis's periodic rough patches (about 90 minutes a week, mostly correlated with the nightly batch jobs): p99 spiked past 300ms.

We had optimized for a metric — primary DB load — that nobody downstream actually felt, at the cost of a metric — p99 latency — that customers absolutely felt. Median latency did improve slightly, from 4ms to 3ms, because cache hits really are fast and most requests are hits. But nobody complains about median latency. They complain about the slow request, and we'd made the slow requests slower and more frequent.

We removed the cache. p99 went back to 41ms. Primary CPU utilization went back up, from 22% to 61% at peak — comfortably under the 80% threshold we alert on, so it turned out we had headroom we didn't need to protect in the first place.

## What we'd measure next time

Before adding a cache in front of a database that's already fast, we'd now insist on:

- **The database's actual p99, not its average**, and whether it has headroom before you add infrastructure to protect it.
- **The cache's own p99 under realistic contention**, not a clean benchmark — including what shares the cluster and what its failover costs in milliseconds, not just availability.
- **End-to-end latency broken out by hit and miss**, before shipping, so a miss penalty is a known number instead of a surprise.
- **Whether the thing being protected is actually under threat.** "This could reduce database load by 90%" is not the same claim as "this database is at risk of falling over." We had the first and treated it as the second.

A cache adds a dependency. Dependencies added in series with something fast and reliable can only help your median — and only hurt your tail, unless you've specifically engineered the cache to be faster and more reliable than the thing it's shielding. Ours wasn't. So we took it out.
