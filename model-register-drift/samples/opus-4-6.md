# Why We Removed Our Cache

Last October, we added a Redis read-through cache in front of the Postgres queries that power our order-history service, Ledger. By February, we'd ripped it out. Our p99 latency dropped from 410ms to 230ms the day we removed it.

This is the story of how caching made our system slower, and what we should have measured before we ever wrote the first `SETEX`.

## The Setup

Ledger serves order history for roughly 12,000 requests per second at peak. The primary read path hits three Postgres tables behind a join: `orders`, `line_items`, and `fulfillments`. Postgres was handling this fine—p50 of 8ms, p99 of 45ms—but during a capacity-planning review, someone noticed the database connection pool was at 70% utilization during peak. The worry was that the next traffic milestone would push us into connection queuing. A read-through cache seemed obvious: cache the rendered order response in Redis, keyed by `order:{id}`, with a 120-second TTL.

We shipped it in a week. The cache-hit path returned in 2ms. The team celebrated and moved on.

## What Went Wrong

Four months later, an SRE flagged something during an incident review: Ledger's p99 had crept from 45ms to 410ms. Not the median—the median looked great at 3ms, down from 8ms. But the tail was nine times worse than before the cache existed.

We spent two days tracing it, and the explanation turned out to be straightforward once we looked at the right graphs.

**Problem 1: Cache misses got slower.** Before the cache, every request went to Postgres, and the connection pool was tuned for that load. After the cache, 94% of requests were served from Redis, so we "right-sized" the Postgres connection pool down from 200 connections to 40. When a cache miss happened, it now had to contend for one of those 40 connections. During any burst of misses—a deploy that flushed a key prefix, a spike in first-time lookups, a Redis node restart—requests queued behind the shrunken pool. A miss that used to take 45ms now took 200-600ms waiting for a connection before the query even ran.

**Problem 2: The bimodal distribution hid the damage.** With a 94% hit rate, our p50 and p75 looked stellar. Dashboards were green. Alerts were based on p50. Nobody was watching p99 until the incident review. The cache didn't eliminate slow requests—it moved them from "every request is moderately fast" to "most requests are very fast and a few are very slow." The aggregate average improved while the worst-case experience got dramatically worse.

**Problem 3: Thundering herd on TTL expiry.** Our 120-second TTL meant that popular orders expired simultaneously. An order viewed 500 times per minute would be served from cache 499 times—and then all viewers in that expiry window would simultaneously miss, all hitting Postgres at once. We'd effectively converted steady database load into periodic spikes. A jittered TTL would have helped, but we hadn't thought to add one.

**Problem 4: Cold-start amplification.** Every Redis restart or failover event dumped 100% of traffic onto Postgres for a few seconds. Before the cache, Postgres was provisioned for full traffic. After the cache, it wasn't—because we'd scaled it down to match the reduced load. The cache had become load-bearing infrastructure while still being treated as an optimization.

## What We Should Have Measured

Before adding the cache, we should have asked three questions and gotten numbers for each:

**1. What does the miss penalty look like under realistic conditions?** Not "what's the query latency on an idle database," but "what happens to query latency when the connection pool is sized for a 90%+ hit rate and then 20% of requests miss simultaneously?" We never load-tested the post-cache Postgres configuration under cache-miss scenarios.

**2. What is the actual shape of our latency distribution, and what are we optimizing for?** Our p50 was 8ms. Our p99 was 45ms. The business requirement was actually about p99—customers care when their order page takes 400ms, not when the average is 3ms. A cache that improves p50 by 5ms but worsens p99 by 350ms is a bad trade for user-facing latency.

**3. How does the system behave when the cache is unavailable?** If the answer is "catastrophic degradation," then the cache isn't a cache—it's a new dependency. We should have load-tested Ledger with the cache disabled against the *new* (downsized) Postgres configuration before calling the project done.

## What We Did Instead

We removed the cache. We scaled the Postgres connection pool back to 200 connections. We added PgBouncer in front of the database to handle connection multiplexing more efficiently. p99 went back to 48ms. We added a connection-pool utilization alert at 60%.

The total compute cost went up by about $400/month from the larger connection pool. We also stopped paying $1,200/month for the Redis cluster. Net savings: $800/month and one fewer system to operate.

## The Lesson

A cache doesn't remove work. It moves work in time—from "now" to "later, all at once." If your system isn't provisioned for that "all at once," you've traded steady load for spiky load and called it an improvement because the dashboard shows a better average. Measure the tail. Test the failure mode. And ask whether the problem you're solving is actually a problem before you add infrastructure to solve it.
