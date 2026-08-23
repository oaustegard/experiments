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
