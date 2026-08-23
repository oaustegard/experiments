# Issue archive: claude-workspace#244

Verbatim copy of [oaustegard/claude-workspace#244](https://github.com/oaustegard/claude-workspace/issues/244),
the collection surface for the writing samples in `samples/`. That repository is
private, so the issue is reproduced here for readers of `RESULTS.md`.

Fetched 2026-08-23. Issue opened 2026-08-23 by @oaustegard; 13 comments, state `open` at fetch time.
Bodies are unedited, including the `MODEL:` lines and the nested code fences
`fetch_samples.py` parses. Comment bodies are the delegates' own text; only the
headings between them are mine.

---

## Issue body

**Register comparison across model generations — writing sample collection** — @oaustegard, 2026-08-23T17:44:35Z

Collection point for a controlled register experiment.

Six models get an identical writing assignment with no voice steering from the
parent session or the user. Each spawned session posts its finished Markdown as
a comment on this issue. The parent then scores every sample against the
`declauding` skill's 42-entry Claudism register (`references/register.md`),
via `scripts/declaude_lint.py` plus a structural pass.

Arms: Opus 5, Opus 4.8, Opus 4.6, Sonnet 5, Sonnet 4.6, Haiku 4.5.

Comment format each delegate is asked for:

```
MODEL: <model id>
---
<the post, verbatim Markdown>
```

This issue is a data-collection surface, not a discussion thread.

---

## Comments

### Comment 1 — @claude[bot], 2026-08-23T17:47:17Z

[`5387540662`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387540662)

MODEL: claude-sonnet-5

````markdown
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
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 2 — @claude[bot], 2026-08-23T17:47:44Z

[`5387542829`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387542829)

MODEL: claude-opus-4-6

````markdown
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
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 3 — @claude[bot], 2026-08-23T18:35:53Z

[`5387767685`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387767685)

MODEL: claude-opus-5

````markdown
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
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 4 — @claude[bot], 2026-08-23T18:37:42Z

[`5387776291`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387776291)

MODEL: claude-haiku-4-5-20251001

````markdown
# Why We Removed Our Cache

Four months ago, the UserService team at Acme deployed a Redis read-through cache in front of our primary Postgres queries. We expected a 40% reduction in database load and faster response times across the board. After running it in production for 16 weeks, we ripped it out. Our p99 latency is now better, not worse. Here's what we learned about measuring before we build.

## The Setup

Our service handles user profile queries: fetching account details, preferences, and subscription status. Before the cache, our Postgres instance was handling ~8,000 queries per second during peak hours, with p50 latencies around 12ms and p99 around 45ms. The database was healthy—CPU was around 60% at peak, and we had a solid B-tree index strategy. But we saw an opportunity: if we could reduce database load, we could handle traffic spikes without scaling, and we'd have room for growth.

We deployed a Redis cluster (3 nodes, 32GB each) with a 15-minute TTL on user profiles. The implementation was straightforward: check Redis first, cache miss → query Postgres, write to Redis, serve the response.

## The Mirage

The first three months looked good on the charts that mattered to us. Redis hit rates stabilized around 87%, and database query volume dropped to about 2,800 queries per second. We congratulated ourselves at retro.

But we were measuring the wrong things.

Nobody had explicitly measured p99 latency before the cache went live. We had monitoring dashboards, but they focused on throughput, error rates, and CPU. So when p99 latency actually started climbing in week 4, we didn't notice until a customer started complaining about slow profile loads during their lunch hour.

When we finally pulled the data, the numbers were sobering:

- **Without cache:** p99 = 45ms (consistent month to month)
- **With cache:** p99 = 78ms (and climbing into week 8)
- **p95 actually improved** (18ms vs 22ms)
- **p50 was roughly the same** (11ms vs 12ms)

The cache was making the tail worse.

## Why This Happened

The culprit was tail latency serialization. Here's the sequence:

1. Request arrives, hits Redis (network round trip: ~1ms)
2. Redis misses (15-20% of requests after churn)
3. Code serializes to Postgres (another ~1-2ms network overhead)
4. Postgres serves the query (12-15ms)
5. Write result back to Redis (~0.5ms)

On a cache miss, you now have **3-4ms of network overhead** layered on top of database latency. Without the cache, you skip steps 1 and 5 and go straight to step 3-4. The database latency is latency; the network overhead is pure loss.

But that doesn't fully explain the spike in p99. The real problem was **thundering herd under uneven load**.

At peak times, user profile queries came in waves: marketing campaign sends emails → hundreds of users open them within seconds → all hit their profile page → most of those user IDs expire from cache within the same 5-minute window → cache fills with misses → every request takes the slow path.

During these waves, our Redis cluster wasn't the bottleneck—our application servers were. They were making sequential network calls (Redis, then Postgres) instead of direct calls to Postgres. Under high concurrency, this added queuing latency.

The p95 improved because for the majority of requests (that hit Redis cleanly), we saved 12-15ms of database latency. But the 5% of requests that missed cache or arrived during thundering herd windows now waited for two network round trips under load, pushing them into the 70-80ms range.

We also had connection pool exhaustion on the Postgres side. With the cache, fewer connections were needed, so we'd tuned down our pool size. When cache misses spiked, we'd run out of available connections and start queuing queries, making p99 even worse.

## What We Should Have Measured First

Before deploying any caching layer, measure these baselines:

1. **Latency percentiles, not averages.** p50, p95, p99, p99.9. The cache only needs to help one metric to seem worthwhile, but tail latency is what users feel.

2. **Cache hit rate required to be worthwhile.** Calculate the math: if a cache hit saves N milliseconds but a miss costs +M milliseconds (network + invalidation overhead), you need a hit rate above (M / (N + M)). For us, that was 85%, and we were right at it. There's no margin.

3. **Latency under cache misses specifically.** Don't just measure average cache miss latency; measure p99 latency when you deliberately poison the cache and force misses. This is your realistic worst case.

4. **Behavior during load spikes.** Use load testing to replay the patterns you actually see: waves of similar queries, thundering herd scenarios, cache expiry happening during peak traffic.

5. **Connection pool size under miss scenarios.** Verify you have enough database connections to handle peak throughput with cache misses. This isn't obvious when the cache looks good.

## What We Did After

We removed the cache entirely and invested the effort elsewhere: improved our Postgres indexes (shaved another 3ms off p99), increased connection pool size, and added a query-specific TTL to reduce redundant requests at the application level (a simple in-memory dedup, not a distributed cache).

Our new p99 is 38ms. Not earth-shattering, but we know it's real and stable.

The lesson: premature caching is expensive because it's invisible. You cache to make systems faster, but if you don't measure tail latency carefully, you'll make them slower and not notice until your customers complain. Measure first, always. And if you're going to add a cache, know exactly how much latency you'll pay for a miss, and make sure your hit rate is high enough to justify it.

Caches are still valuable—just not for systems that are already fast.
````


---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 5 — @claude[bot], 2026-08-23T18:40:16Z

[`5387787794`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387787794)

MODEL: claude-sonnet-4-6

````markdown
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
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 6 — @claude[bot], 2026-08-23T18:49:57Z

[`5387845208`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387845208)

MODEL: claude-opus-4-8

````markdown
# Why we removed our cache

Four months ago we put a Redis read-through cache in front of the Postgres queries powering our product catalog service, `catalog-api`. Last week we took it out. Not because Redis failed, and not because of a cache-invalidation horror story. We removed it because our p99 latency was *worse* with the cache than without it — 180ms with Redis, 120ms without. Here is how that happens, and what we should have measured before writing a line of caching code.

## The setup

`catalog-api` serves product detail lookups by ID. Each request ran one indexed primary-key query against a well-warmed Postgres replica. Median latency was around 4ms, p99 around 110ms. The p99 was dominated by a handful of large products whose rows joined out to big variant and pricing tables.

Someone looked at the median, saw 4ms of "database time" on every request, and reasoned that caching would eliminate it. We stood up Redis, wrote a read-through wrapper — check Redis, on a miss query Postgres and populate the key with a 10-minute TTL — and shipped it. The dashboards showed an 84% hit rate. Everyone moved on.

## Why the tail got worse

The mistake was reasoning about the mean and shipping to affect the tail. Caching changes the *shape* of your latency distribution, and the tail is where it bites back.

**A cache adds a fixed tax to every request.** Every single call now does a Redis round trip first. In our network that was a real 1.5ms at median but a fat-tailed 40ms+ at p99 — Redis has its own tail, driven by connection-pool contention and the occasional slow `GET` when a big serialized value comes back over the wire. On a hit, great, we skip Postgres. On a miss, we pay Redis *and* Postgres. So the miss path is strictly slower than the no-cache path ever was.

**Your tail lives in the misses, and misses cluster.** An 84% hit rate sounds like the tail should shrink by 84%. It doesn't. p99 is defined by your slowest 1% of requests, and the slowest requests are exactly the ones least likely to be cached: rarely-requested products with big, expensive joins. Popular products — the cache hits — were already fast in Postgres. We were caching the cheap queries and still paying full price, plus Redis tax, on the expensive ones.

**Serialization isn't free.** Those big product rows had to be serialized to JSON, stored, fetched, and deserialized. For the largest products, JSON encode/decode plus moving ~200KB through Redis cost more than the original indexed query. We had turned a 90ms database read into a 130ms cache miss.

**Cache stampedes concentrate the pain.** Every 10 minutes a hot key expired, and the concurrent requests that missed simultaneously all fell through to Postgres at once. That synchronized load spiked replica latency for *everyone*, cached or not, right at the moment the TTL rolled over. Our p99 had a sawtooth in it that mapped exactly to the TTL boundary.

Put together: the median improved slightly (hits are fast), but p99 got dragged up by the miss penalty, the serialization cost on big rows, and the stampede spikes. We optimized the case that was already fine and taxed the case we cared about.

## What we should have measured first

None of this required four months to discover. It required asking the right questions before adding a dependency.

**Where does the tail actually come from?** We should have pulled a p99 trace breakdown before touching Redis. It would have shown that our tail was join and row-size bound, not round-trip bound — a problem a cache can't fix and can make worse. A denormalized column or a covering index would have addressed the actual slow queries.

**What is the cost of a miss, not just the benefit of a hit?** The right model isn't "hit rate × savings." It's `hit_rate × hit_latency + miss_rate × (redis_latency + db_latency)`, evaluated *per percentile*, not on the mean. Run those numbers with our real distribution and the p99 regression was predictable on paper.

**What is Redis's own latency distribution under our load?** We treated the cache as free and instant. We never load-tested its p99, never sized the connection pool for our concurrency, never accounted for large-value serialization.

**Does the workload even suit a cache?** Read-through caching pays off with a skewed read distribution, small values, and queries that are genuinely expensive to recompute. Ours had cheap hits and expensive misses — close to the worst-case shape.

## The takeaway

A cache is a bet that recomputation is more expensive than a network round trip and a serialization pass, for the requests that matter. For our workload that bet was wrong, and averages hid it. We replaced Redis with one covering index and a denormalized price column. p99 is now 70ms, we deleted a stateful dependency, and on-call got quieter.

Measure your tail before you cache it. If you can't explain, per percentile, where your latency comes from, a cache is just as likely to move the problem as to solve it.

````


---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 7 — @claude[bot], 2026-08-23T18:58:00Z

[`5387883203`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387883203)

MODEL: claude-haiku-4-5-20251001

````markdown
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
````


---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 8 — @claude[bot], 2026-08-23T18:59:22Z

[`5387889561`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387889561)

MODEL: claude-sonnet-4-6

````markdown
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
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 9 — @oaustegard, 2026-08-23T19:16:47Z

[`5387970884`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387970884)

# Why We Removed Our Cache

Last quarter, we ripped out the Redis cache that sat in front of our primary Postgres database. The cache had been running for four months. Our p99 latency was **worse** with it than without it.

This is the story of how that happened and what we should have measured before we ever added the cache.

## The Setup

Our service, `order-ledger`, handles order lookups for an internal fulfillment pipeline. It serves about 12,000 requests per minute, almost all of them reads against an `orders` table with roughly 40 million rows. The table has proper indexes. Postgres was returning results in 4–8ms at p50 and 18–22ms at p99.

That p99 bothered us. Someone proposed a Redis read-through cache: on a miss, query Postgres, write the result to Redis with a 120-second TTL, and serve subsequent reads from memory. Simple. We estimated a 70% hit rate based on access-pattern sampling, and the math looked great on a whiteboard.

## What Happened

After deploying, our p50 dropped from 6ms to 1.2ms. We high-fived. But two weeks in, on-call started noticing something: alert pages were coming more often, not less. We dug into the p99 numbers.

Before the cache, p99 was a stable 20ms. After the cache, p99 had climbed to 35ms and kept spiking to 80ms during peak hours.

Three things were going wrong.

### 1. Cache Stampedes on Popular Keys

Our TTL was uniform: 120 seconds for every key. When a hot key expired, dozens of concurrent requests would miss simultaneously and all hit Postgres at once. Postgres would handle it, but the query time for that burst would jump to 40–60ms — each request waiting on the same lock contention in the connection pool. The irony: the most popular data, the data the cache was "for," created the worst latency spikes.

### 2. Bimodal Latency Distribution

With the cache, our latency distribution became bimodal. Hits returned in under 2ms. Misses took the original 6ms from Postgres *plus* 1.5ms to write back to Redis. That writeback penalty was small in isolation, but it shifted the miss path from 6ms to roughly 8ms — and misses still made up 30% of traffic because our actual hit rate was lower than projected.

We had modeled our hit rate from access logs, but those logs didn't account for the long tail of order lookups that happened exactly once (fulfillment workers checking an order, then never looking at it again). These one-shot reads always missed the cache, always paid the writeback cost, and contributed nothing to future hits. They were pure overhead.

### 3. The GC and Serialization Tax

We were using a Java service with Jackson for serialization. Every cache hit required deserializing a JSON blob back into an `OrderSummary` object. At 8,400 cache hits per minute, this created significant GC pressure. Our young-gen pauses went from 5ms to 12ms, and a full GC — which now happened every 90 seconds instead of every 5 minutes — would stall the event loop for 40–80ms. These pauses showed up directly in our tail latency.

## What We Should Have Measured

Before adding the cache, we should have answered four questions:

**What is the actual hit-rate ceiling?** Not the theoretical one from access logs. We should have run a trace of real traffic through an LRU simulator to see how many requests would actually benefit. Our real hit rate was 68%, but only 45% of *unique keys* were ever accessed more than once within a TTL window.

**What does the miss path cost, end to end?** Adding a cache doesn't make misses free. It makes them slower. You now have a Redis round-trip that fails, a Postgres query, and a Redis write. If your miss rate is non-trivial, you need to account for that extra cost.

**What is the variance of the uncached path?** Our Postgres p99 was 20ms. That's not bad. Caching is highest-value when the uncached path is both slow and variable — a 200ms query with spikes to 2 seconds. When the uncached path is fast and consistent, the headroom for improvement is small, and the new failure modes you introduce can easily eat it.

**What happens when hot keys expire simultaneously?** We should have simulated cache stampedes under realistic concurrency. Probabilistic early expiration or lock-based recomputation would have mitigated the issue, but the better question was whether we needed the cache at all.

## The Removal

We pulled Redis out on a Tuesday. p50 went from 1.2ms back to 6ms. p99 went from 35ms back to 19ms. Alert pages dropped to zero. We deleted the `CacheConfig` class, removed the Redis dependency, and closed the Terraform module.

Total lines of code removed: 1,400. Total operational alerts eliminated: the Redis cluster monitoring, the hit-rate dashboards, the eviction-pressure alarms. Total on-call incidents in the four months the cache was live: eleven. Total incidents in the four months after removal: two, both unrelated.

## The Lesson

A cache is not a performance improvement. It is a trade: you are trading a predictable, slightly slower path for a bimodal distribution with a fast path and a potentially worse slow path. If your slow path was already acceptable, you may have just traded a solved problem for an unsolved one.

Measure the tail, not the median. The median will always look better with a cache. That's the trap.

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 10 — @claude[bot], 2026-08-23T19:17:05Z

[`5387972240`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387972240)

MODEL: claude-opus-4-6

````markdown
# Why We Removed Our Cache

Last quarter, we ripped out the Redis cache that sat in front of our primary Postgres database. The cache had been running for four months. Our p99 latency was **worse** with it than without it.

This is the story of how that happened and what we should have measured before we ever added the cache.

## The Setup

Our service, `order-ledger`, handles order lookups for an internal fulfillment pipeline. It serves about 12,000 requests per minute, almost all of them reads against an `orders` table with roughly 40 million rows. The table has proper indexes. Postgres was returning results in 4–8ms at p50 and 18–22ms at p99.

That p99 bothered us. Someone proposed a Redis read-through cache: on a miss, query Postgres, write the result to Redis with a 120-second TTL, and serve subsequent reads from memory. Simple. We estimated a 70% hit rate based on access-pattern sampling, and the math looked great on a whiteboard.

## What Happened

After deploying, our p50 dropped from 6ms to 1.2ms. We high-fived. But two weeks in, on-call started noticing something: alert pages were coming more often, not less. We dug into the p99 numbers.

Before the cache, p99 was a stable 20ms. After the cache, p99 had climbed to 35ms and kept spiking to 80ms during peak hours.

Three things were going wrong.

### 1. Cache Stampedes on Popular Keys

Our TTL was uniform: 120 seconds for every key. When a hot key expired, dozens of concurrent requests would miss simultaneously and all hit Postgres at once. Postgres would handle it, but the query time for that burst would jump to 40–60ms — each request waiting on the same lock contention in the connection pool. The irony: the most popular data, the data the cache was "for," created the worst latency spikes.

### 2. Bimodal Latency Distribution

With the cache, our latency distribution became bimodal. Hits returned in under 2ms. Misses took the original 6ms from Postgres *plus* 1.5ms to write back to Redis. That writeback penalty was small in isolation, but it shifted the miss path from 6ms to roughly 8ms — and misses still made up 30% of traffic because our actual hit rate was lower than projected.

We had modeled our hit rate from access logs, but those logs didn't account for the long tail of order lookups that happened exactly once (fulfillment workers checking an order, then never looking at it again). These one-shot reads always missed the cache, always paid the writeback cost, and contributed nothing to future hits. They were pure overhead.

### 3. The GC and Serialization Tax

We were using a Java service with Jackson for serialization. Every cache hit required deserializing a JSON blob back into an `OrderSummary` object. At 8,400 cache hits per minute, this created significant GC pressure. Our young-gen pauses went from 5ms to 12ms, and a full GC — which now happened every 90 seconds instead of every 5 minutes — would stall the event loop for 40–80ms. These pauses showed up directly in our tail latency.

## What We Should Have Measured

Before adding the cache, we should have answered four questions:

**What is the actual hit-rate ceiling?** Not the theoretical one from access logs. We should have run a trace of real traffic through an LRU simulator to see how many requests would actually benefit. Our real hit rate was 68%, but only 45% of *unique keys* were ever accessed more than once within a TTL window.

**What does the miss path cost, end to end?** Adding a cache doesn't make misses free. It makes them slower. You now have a Redis round-trip that fails, a Postgres query, and a Redis write. If your miss rate is non-trivial, you need to account for that extra cost.

**What is the variance of the uncached path?** Our Postgres p99 was 20ms. That's not bad. Caching is highest-value when the uncached path is both slow and variable — a 200ms query with spikes to 2 seconds. When the uncached path is fast and consistent, the headroom for improvement is small, and the new failure modes you introduce can easily eat it.

**What happens when hot keys expire simultaneously?** We should have simulated cache stampedes under realistic concurrency. Probabilistic early expiration or lock-based recomputation would have mitigated the issue, but the better question was whether we needed the cache at all.

## The Removal

We pulled Redis out on a Tuesday. p50 went from 1.2ms back to 6ms. p99 went from 35ms back to 19ms. Alert pages dropped to zero. We deleted the `CacheConfig` class, removed the Redis dependency, and closed the Terraform module.

Total lines of code removed: 1,400. Total operational alerts eliminated: the Redis cluster monitoring, the hit-rate dashboards, the eviction-pressure alarms. Total on-call incidents in the four months the cache was live: eleven. Total incidents in the four months after removal: two, both unrelated.

## The Lesson

A cache is not a performance improvement. It is a trade: you are trading a predictable, slightly slower path for a bimodal distribution with a fast path and a potentially worse slow path. If your slow path was already acceptable, you may have just traded a solved problem for an unsolved one.

Measure the tail, not the median. The median will always look better with a cache. That's the trap.
````


---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 11 — @claude[bot], 2026-08-23T19:21:34Z

[`5387992423`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5387992423)

MODEL: claude-opus-4-8

````markdown
# Why we removed our cache

Four months ago we put a Redis read-through cache in front of the hottest read path in our `catalog-service`: a set of Postgres queries that resolve a product ID into a fully hydrated product record. Last week we took it out. Not because it broke, and not because Redis fell over. We removed it because our p99 latency was *worse* with the cache than without it, and once we understood why, keeping it made no sense.

This is the story of how a cache can make your tail latency worse, and the measurements we skipped that would have told us not to bother.

## The setup

The read path was simple. A request comes in, we look up a product by ID, and we return it. The query hit our primary Postgres instance and, on a warm database, ran in about 3ms at the median and 40ms at p99. That p99 bugged someone, so we did the obvious thing: read-through cache in Redis, 10-minute TTL, look in Redis first and fall back to Postgres on a miss.

The rollout looked great in the dashboards that we were watching. Median latency dropped from 3ms to about 0.9ms. Cache hit rate settled around 82%. Everyone moved on.

## What actually happened at the tail

The problem is that median latency was never the thing we cared about. We added the cache because of p99, and p99 is where a read-through cache quietly betrays you.

Think about what a cache does to the *distribution* of latencies, not the average. On a hit, you pay one Redis round-trip: fast, ~0.4ms. On a miss, you pay the Redis round-trip **and then** the Postgres query. The miss doesn't replace the database call; it happens *before* it. Every cache miss is now strictly slower than it would have been with no cache at all.

So the question that decides whether the cache helps your tail is: *what is sitting in the slow part of the distribution?* And the answer, for us, was exactly the requests that miss.

Our access pattern was heavily long-tailed. A small number of popular products got hit constantly and lived comfortably in cache. But the p99 requests were, almost by definition, the unpopular ones — the products nobody had asked for in the last ten minutes. Those requests missed, and now each of them paid an extra Redis hop on top of the same Postgres query they used to run directly. Our 82% hit rate was real, but it was concentrated entirely in the traffic that was already fast. The cache made the fast requests slightly faster and the slow requests slightly slower. p99 went from 40ms to roughly 65ms.

It got worse under load. Redis is single-threaded, and we were storing fairly large serialized product blobs. During traffic spikes the Redis round-trip itself picked up jitter, and TTL expiry on popular keys produced small stampedes: a hot key expires, a few hundred requests miss simultaneously, and they all hit Postgres at once. We had added a second network dependency, a serialization step, and a new failure mode — all to speed up queries that Postgres was already serving from its own `shared_buffers` in a couple of milliseconds.

That last point is the one that stung. Postgres already has a cache. Our working set fit in memory, so the database wasn't doing physical I/O for hot rows anyway. We had essentially bolted a slower, less reliable cache in front of a faster one.

## What we should have measured first

None of this required four months to discover. It required about a day of measurement that we did *after* the fact instead of before. If you are about to put a cache in front of a database, measure these first:

- **The latency distribution, not the average.** Pull p50, p95, p99, p999 for the query as it is today. If your pain is at the tail, a cache only helps if the tail requests are the ones that will hit.
- **The realistic hit rate *for the slow requests specifically.*** An overall hit rate is close to useless here. Bucket your requests by latency and ask what fraction of your *slow* requests would have hit the cache. If cold, unpopular keys dominate your tail, the cache can't touch it.
- **The miss penalty.** Measure Redis round-trip time under production load and add it to your current query time. That sum is your new latency floor on every miss.
- **Whether the database is actually the bottleneck.** Check `shared_buffers` hit ratio and whether you're doing physical reads. If Postgres is serving from memory, you're competing against RAM, and Redis-over-the-network will lose.
- **The write and invalidation cost.** How often does the underlying data change, and what does a stampede on expiry do to the database you were trying to protect?

## The result

We pulled the cache, deleted the invalidation code, and removed a dependency from the on-call runbook. p99 went back to 40ms, p50 rose from 0.9ms to 3ms, and nobody noticed — no SLO was written against the median.

Caching is not free performance. It's a bet that your hits will outnumber your misses *where it matters* — and "where it matters" is almost never the average. Measure the tail before you reach for Redis. Sometimes the fastest cache is the one your database already has.

````

---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 12 — @claude[bot], 2026-08-23T19:23:52Z

[`5388002641`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5388002641)

MODEL: claude-opus-5

````markdown
# Why we removed our cache

In March we put a Redis read-through cache in front of the hottest query in `ledger-api`, the service that answers "what is this account's current balance?" In July we took it out. In between, p50 dropped from 11 ms to 3 ms and p99 rose from 42 ms to 61 ms.

We had an SLO on p99. We shipped a change that made p50 look great on the dashboard everyone screenshots, and made the number we were actually judged on worse. Here is the mechanism, because none of it is specific to us.

## p99 lives entirely inside your misses

The read-through path is three steps on a miss: `GET` from Redis, query Postgres, `SETEX` the result back. On a hit it is one step. A cache does not make requests faster — it splits your latency distribution into two populations, a fast one and a *slower-than-before* one. The miss path costs the uncached query plus two network round trips plus serialization.

We ran a 94% hit rate. That sounds excellent. Now consider what it means for the tail: 6% of requests are misses, so the slowest 6% of the distribution is, roughly, the miss population. p99 sits at the 99th percentile of everything, which is comfortably inside that top 6%. The hits — all 94% of them, all beautifully fast — are below p94. They are invisible to p99.

That is the whole thing in one sentence: **a cache cannot improve p99 unless the miss rate is below 1%.** Under that threshold you are cutting into the tail itself. Over it, every point of p99 is a miss, and you have made misses more expensive than they were. 94% is not close to good enough. 99.5% would have been.

## The miss path got worse over time

Misses also didn't stay at their day-one cost. Before the cache, Postgres was serving balance lookups almost entirely from `shared_buffers` — the buffer cache hit ratio on that table was 99.7%. Ordinary read traffic kept the hot pages resident.

Then Redis absorbed 94% of the reads. Postgres stopped seeing the traffic that kept those pages warm, and the remaining 6% arrived as a thin, scattered stream across a 40-million-row table. Buffer hit ratio on that table fell to 88% over the next six weeks. Our misses started hitting disk.

This is the ugly feedback loop: the cache starves the layer beneath it of the access pattern that made that layer fast, so the miss path degrades *in proportion to how well the cache works*. We traded one warm cache for two lukewarm ones and put a network hop between them.

## Redis is a shared, single-threaded resource

Two more contributions to the tail. Redis processes commands on one thread, so a slow command blocks everything behind it. Our balance payloads were JSON, averaging 4 KB, and a nightly reconciliation job ran large `MGET`s against the same cluster. Redis RTT was 0.6 ms at p50 and 9 ms at p99 — a fifteenfold spread on a component we had been modeling as a constant.

Evictions brought stampedes. The TTL was 300 s with no jitter, so keys written together expired together, and a burst of concurrent misses for one key all went to Postgres at once. We eventually added a per-key lock, which fixed the stampede and added another Redis round trip to the miss path.

## What we should have measured first

None of this needed four months in production to discover.

**The latency distribution of the query, not its mean.** We knew balance lookups "averaged about 14 ms." We did not know the distribution was already bimodal, or what was in the tail. It turned out most of our p99 was connection-pool wait under burst, not query execution. A cache does nothing about that and never could have.

**The achievable hit rate, offline.** Take a day of production keys, the intended cache size, and the intended TTL, and simulate. Key cardinality and request skew determine your hit rate, and you can compute it from a log file in an afternoon. If the answer isn't ≥99%, a cache will not move p99.

**The database's own cache hit ratio.** `pg_statio_user_tables` tells you whether Postgres is already serving from memory. If it is, Redis isn't saving you a disk read — it's adding a network hop in front of something that was already RAM-speed.

**Redis RTT under production concurrency**, at p99, not p50, and not from `redis-benchmark` against an idle cluster.

**Which number you are actually trying to move.** This is the one we skipped, and it subsumes the rest. Caching is excellent at shedding *load*: ours took 94% of read volume off the primary, which was real and valuable. It is much worse at improving *tail latency*. Those are different goals with different success criteria, and we wanted the second while measuring neither.

## Where we landed

We removed the cache, raised `work_mem`, fixed an index that was producing a bitmap heap scan on about 3% of lookups, and moved balance reads to a replica. p99 is 24 ms now and p50 is 9 ms — worse at the median than the cached system, better everywhere that mattered.

If we need to shed read load from the primary again, we'll put a cache back. We'll just be honest that that's what it's for.
````


---
_Generated by [Claude Code](https://claude.ai/code)_

---

### Comment 13 — @claude[bot], 2026-08-23T19:24:48Z

[`5388006772`](https://github.com/oaustegard/claude-workspace/issues/244#issuecomment-5388006772)

MODEL: claude-sonnet-5

````markdown
# Why we removed our cache

Four months ago we put a Redis read-through cache in front of the hottest query in our Payments Platform: `GetAccountBalance`, the call our checkout service makes on every order to confirm a buyer's account is in good standing. Last week we ripped it out. This is the postmortem, minus the org-chart drama.

## The setup

The query hits `ledger-pg-01`, our primary Postgres instance (RDS, `db.r6g.4xlarge`), and it's not a complicated query — an indexed lookup by account ID joined against a small rollup table. Before caching, it ran at a steady 12ms p50 and 45ms p99 under normal load, and closer to 60ms p99 during our nightly settlement batch, which competes for the same connection pool.

At 40k requests/second peak, even a "fast" query adds up, and the on-call rotation was tired of settlement-window latency pages. So we added a read-through cache: `ledger-cache`, a three-shard Redis 7 cluster on `r6g.xlarge` nodes, keyed by account ID, 30-second TTL, synchronous invalidation on writes.

## What the dashboards said, and what they hid

The average-latency graph looked great immediately. p50 dropped from 12ms to 3ms. Hit rate settled around 91%. We shipped it, wrote the wiki page, moved on.

p99 did not drop. It went from 45ms to 210ms, and it got worse specifically during the traffic windows we'd built the cache to help. We didn't notice for weeks because our alerting was on p50 and error rate, not tail latency — the exact blind spot the cache was supposed to fix.

Three things were happening at once:

**Hot-key skew overloaded a single shard.** Our traffic isn't remotely uniform — the top 50 accounts (marketplace sellers with high order volume) generated roughly 8x the lookup rate of a median account. Consistent hashing put a disproportionate share of those keys on one shard, which meant that shard's CPU spiked well above the cluster average during peak, even though the *cluster-wide* hit rate and load looked fine on the rollup dashboard.

**Miss-plus-fallback cost more than no cache at all.** Our client timeout on Redis was 50ms, with a fallback to Postgres on timeout or miss. That's the right instinct, but it meant that whenever the hot shard got slow, requests paid the full 50ms waiting on Redis *and then* paid the normal Postgres query time on top of it, sequentially. A request that would have cost 12ms with no cache now cost 60-90ms. This happened exactly when the shard was under load — which was exactly when traffic was highest — so the worst-case path and the highest-traffic path were the same path.

**Connection pool exhaustion turned a Redis problem into an application problem.** The Redis client used a blocking pool of 20 connections shared across roughly 200 app server threads. When Redis latency crept up even 10-15ms, threads started queuing for a pool connection rather than failing fast, and that queuing delay showed up as request latency completely unrelated to anything Redis was actually doing wrong.

Underneath all of that, synchronous cache invalidation on writes added a Redis round-trip to the write path, and during shard slowness that showed up as write latency too — a query optimization had leaked into the write path's tail.

## Six weeks of tuning, then the call to remove it

We tried rebalancing shards with a better hash function, splitting the connection pool by criticality, shortening the timeout, adding jittered backoff. Each change bought back a little tail latency and cost a little complexity. After six weeks, p99 was still worse than the pre-cache baseline, and we had a new distributed system to operate, monitor, and explain to new hires. We removed the cache and added a Postgres read replica for the balance-lookup path instead. p99 went back to 48ms. Nobody misses the cache.

## What we should have measured first

- **The full latency distribution of the uncached path, not the average.** p50 of 12ms hid a p99 that already spiked to 60ms during settlement. A cache that helps the median can still make the tail worse if it adds a failure mode that correlates with your worst existing conditions.
- **Key access skew.** A 91% hit rate is a cluster-wide average; it says nothing about whether that traffic is spread evenly across shards. We should have pulled the actual production key-frequency distribution before choosing a partitioning scheme, not after.
- **The true cost of a miss.** If a cache miss means "pay for the cache lookup, then pay for the query anyway," the miss path must never be allowed to run sequentially under load — either it needs to race the two, or the timeout needs to be short enough that the combined worst case still beats no-cache. We never modeled combined worst-case latency, only best-case hit latency.
- **Resource contention between the cache client and the app runtime.** Connection pool sizing versus thread count is capacity planning, not an afterthought, and it deserves the same load testing as the database itself.
- **A load test with fault injection, not just steady state.** Every benchmark we ran was clean-cluster, clean-network. We never ran one where a shard was artificially slowed down while under peak synthetic load, which is exactly the scenario that hurt us in production.

The honest summary: the cache made the common case faster and the bad case much worse, and we were only watching the metric that showed the common case. If your p99 matters — and for a payments path, it always does — benchmark the cache against its failure modes before you benchmark it against its best case.
````

---
_Generated by [Claude Code](https://claude.ai/code)_

---
