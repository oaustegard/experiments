# Why we removed our cache

When we added Redis in front of our user service queries four months ago, every metric looked good. Cache hit rates climbed to 92%. Average latency dropped by 200ms. We shipped it to production expecting wins across the board.

Then we looked at p99 latency.

Our p99 had gotten worse—not by a little, but by 800ms. We were hitting 2.4 seconds at the tail when we used to hit 1.6 seconds. We spent the next six weeks pulling it out and learned some hard lessons about what we should have measured in the first place.

## How a cache makes tail latency worse

Here's the thing nobody tells you: a cache adds complexity in the request path, and complexity compounds at high percentiles.

Our setup looked textbook. Requests came in, checked Redis first, fell back to Postgres on miss. On paper, that's straightforward. In practice, we had three failure modes that only showed up at p99:

**Thundering herd on eviction.** Our cache key scheme included user ID. When a user had more than 500 active cache entries, Redis evicted older keys using LRU. If eviction happened during a traffic spike, dozens of workers would miss simultaneously, all hammering Postgres at once. Postgres query time spiked to 400ms. Without the cache, those requests spread naturally across time; with it, we synchronized them.

**Connection pool exhaustion.** Our cache client (Predis) wasn't pooling connections efficiently. We had 40 application servers, each maintaining a persistent connection to Redis. Whenever Redis garbage-collected or restarted, all 40 clients would reconnect at once—a 15-second window where we'd exhaust the connection limit and start queuing. Those queued requests wouldn't time out; they'd wait, pushing p99 higher and higher. Before the cache, we had one Postgres connection pool; after, we had Redis *and* Postgres, and both could saturate.

**Serialization cost at scale.** Cache hits were fast (3ms), but misses had an invisible tax. We were serializing/deserializing JSON to/from Redis. For a user with a complex permission structure, that was 80KB of data. At p99 traffic (3x our median), network overhead and parsing made cache misses actually slower than direct Postgres queries (which at least had the query optimizer working in our favor).

## What we measured (and what we didn't)

We tracked the wrong things.

**What we measured:**
- Hit rate (92%, excellent)
- Average latency (down 200ms, shipped)
- Cache eviction rate (normal levels)

**What we didn't measure:**
- Hit rate stratified by percentile
- Latency correlation with traffic
- Full request path latency vs. component latency
- Connection behavior during cache storms
- Eviction impact on downstream queries
- Tail behavior of the "slowest 5%" of cache misses

If we'd looked at our p99 hit rate, we would have found it was actually 60%. High-percentile traffic hits the cache less often (more users, larger datasets, colder keys), so we were getting the worst of both worlds: complexity tax plus miss penalties.

A simple test would have caught it: we should have load-tested with request distributions that matched production traffic—including the bursty patterns that drive p99. Instead, we tested with uniform request rates. The synthetic load never triggered the eviction storms or connection exhaustion we saw live.

## The math that doesn't scale

Here's the arithmetic we should have done before shipping.

Our baseline Postgres query had a p99 of 200ms and a p50 of 10ms. Adding the cache:

- **Cache hit path:** 3ms (Redis lookup) + 1ms (serialization) = 4ms. Wins 196ms.
- **Cache miss path:** 15ms (Redis lookup) + 200ms (Postgres) + 1ms (serialization) + 10ms (network/queue) = 226ms. Loses 26ms.

At 92% hit rate in aggregate, that looks like +142ms improvement. But at p99, with 60% hit rate:
- 60% of requests: 4ms
- 40% of requests: 226ms

Our effective p99 latency: we're sorting the 99th percentile tail. When 40% of requests are slow, the 99th percentile captures them. The few requests that escape under 200ms in the miss case? Buried.

That math completely changed when we factored in: eviction storms adding 400ms, connection queueing adding 50ms, and the fact that without the cache, Postgres query variance was ±50ms, but our cache miss variance was ±200ms. Tail risk exploded.

## What we do now

We removed the cache and bought better Postgres hardware instead. For the 20% of our queries that truly benefit from caching (bulk reports that run hourly), we pre-warm a separate Redis instance that we don't evict from. We use it as a write-through cache, not read-through, so misses go to application memory, not back to Postgres.

Before adding *any* cache now, we measure:

1. **Latency at every percentile**, not just average. (p50, p95, p99, p99.9)
2. **Hit rate stratified by load level.** Is it 92% when idle and 40% under spike?
3. **The miss path in isolation.** How slow is a cache miss compared to a direct query?
4. **Connection and queue behavior.** What happens when the cache restarts? When it evicts? Under sustained load?
5. **Customer-visible latency**, not component latency. The only metric that matters is whether the user's request is slower or faster end-to-end.

The cache was a textbook premature optimization. Average latency was never the bottleneck; we optimized a metric that didn't matter while breaking the metric that did.

If you're thinking about adding a cache, run the load test first. And look at p99, not average.
