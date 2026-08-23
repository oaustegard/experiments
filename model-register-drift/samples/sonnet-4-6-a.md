# Why We Removed Our Cache

Four months ago we shipped Redis in front of our primary Postgres read path. Last Tuesday we ripped it out. p99 latency improved by 340ms the moment we did.

Here's what went wrong, and what we wish we'd measured first.

## The Setup

Our service is **Fetchr**, an internal fulfillment query API used by warehouse operators. The hot path is a `GET /orders/{id}` endpoint backed by a five-table Postgres join. At peak—roughly 9 AM and 2 PM warehouse shifts—we were seeing p50 around 28ms and p99 around 190ms. Nothing alarming, but the team wanted headroom before the holiday push.

The proposal was straightforward: add a Redis read-through cache with a 60-second TTL on order lookups. Order status changes maybe once every few minutes on average. The hit rate should be high. Should be a slam dunk.

We deployed it on week 14. Postgres CPU dropped noticeably. Redis hit rate climbed to 82% within two days. Everyone felt good.

Then we pulled the p99 numbers at the end of month one: **530ms**. Up from 190ms.

## What Actually Happened

### The tail isn't where you think it is

Our 82% hit rate looked healthy. But think about what that 18% miss rate means at our traffic volume. We handled roughly 4,200 requests per minute at peak. That's **756 cache misses per minute**—all of which go to Postgres, most of them clustered around cache expiry windows, and all of them competing for connection pool slots that were now sized for a lighter Postgres load.

The hit rate is an average. It obscures the *shape* of misses. Our 60-second TTL meant every popular order ID expired in a synchronized wave. When the warehouse team refreshes their boards at shift start, they all hit the same order IDs, all expired, all at once. We had introduced a thundering herd we didn't have before.

### We added latency to the fast path

Before the cache, a request either hit Postgres in 28ms or waited in a connection queue. After the cache, every request paid the cost of:

1. Redis round-trip: ~4ms
2. Deserializing the cached payload: ~2ms
3. **Then** Postgres if it missed

On a hit, we went from 28ms to 6ms—great. But on a miss, we went from 28ms to 34ms before even touching Postgres. During miss storms, Postgres was now also contending with a reduced connection pool (we'd shrunk it when Redis came online), so queuing latency spiked.

The p50 improved. The p99 got demolished.

### The cache made failures compound

Redis sat on a shared cluster. Twice during those four months we saw transient Redis latency spikes—one from a keyspace notification misconfiguration, one from a noisy neighbor during a compaction. Each time, Fetchr's latency spiked with it. Before the cache, Redis didn't exist in our failure surface. Now it did.

## What We Should Have Measured First

**1. Miss rate under realistic traffic, not average traffic.**

We modeled hit rate on overall request volume. We should have modeled it against *unique key request rate* during the worst five-minute window of a shift. That would have shown us that our 60-second TTL was almost exactly aligned with how often operators refreshed their UIs.

**2. Tail latency of the cache itself.**

Redis p99 on our shared cluster was 18ms, not 4ms. We'd been looking at mean latency in our Redis dashboards. An 18ms p99 overhead on a 28ms baseline is significant.

**3. Connection pool behavior under partial hit load.**

We reduced Postgres pool size from 40 to 20 connections because we expected fewer Postgres queries. We should have done the math: at 18% miss rate with 4,200 RPS, we still needed capacity for ~756 QPS to Postgres, all bursty. Twenty connections wasn't enough.

**4. Failure mode analysis.**

We never asked: what happens to every Fetchr request if Redis has a bad minute? The answer turned out to be "they all get slower," which is worse than the Postgres-only baseline during Redis incidents.

## The Decision

We tried tuning first—jitter on the TTL, a larger connection pool, cache stampede protection with a probabilistic early reexpiry. Jitter helped. The fundamental problem didn't go away: our query was already fast, our data changed frequently enough to limit hit rates, and we'd added operational complexity for latency wins that only showed up at the median, not the tail.

So we removed it. The Postgres queries are fine. They were always fine. We have better indexes now as a side effect of the investigation, and we spent a day adding query-level timing histograms we should have had from the start.

The cache was solving a problem we assumed we had. The metrics we needed to justify it were right there—we just didn't pull them before we shipped.
