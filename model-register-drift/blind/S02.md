# Why we removed our cache

Last March we put a Redis read-through cache in front of the Postgres queries backing `ledger-api`, our account and balance service. Last July we took it out. In between, our p99 latency got worse — not dramatically, but consistently, and in a way that took us an embarrassingly long time to attribute to the cache itself.

This is the writeup we wish we'd read before we started.

## What we built

`ledger-api` serves account lookups and balance reads: roughly 12k rps at peak, backed by a single Postgres primary with two read replicas. The hot path was a pair of indexed point queries against `accounts` and `balance_snapshots`.

The cache was the obvious shape. On read, check Redis; on miss, query Postgres, serialize the row to JSON, `SETEX` with a 60-second TTL. Shared ElastiCache cluster, three nodes, TLS on, already used by four other services.

## What happened

Mean latency improved. p50 went from 11ms to 8ms. Everyone was happy for about six weeks, which is how long it took for a p99 dashboard to become someone's problem.

| | before | with cache |
|---|---|---|
| p50 | 11ms | 8ms |
| p99 | 190ms | 240ms |
| p99.9 | 480ms | 1.2s |

Three things were going on.

**The tail is the miss path.** Our hit rate was 61%. That means the slowest 1% of requests — the ones p99 actually describes — were essentially all misses, and a miss is now *strictly more work* than the uncached query ever was: a Redis round trip, then the Postgres query, then serialization and a write-back. We measured the miss penalty at 1.6ms of added serial work. We had budgeted for a 95% hit rate based on nothing but optimism.

**Postgres was already the cache.** Our hot rows lived in `shared_buffers`. The uncached point query had a p50 of 1.4ms and a p99 of 4ms — it was already a memory read with some protocol overhead on top. Redis wasn't replacing a disk seek; it was replacing another memory read, one network hop away, and charging us a hop to do it.

**We added a dependency's tail to every request.** The shared Redis cluster had a p99 of 6ms and a p99.9 of 40ms under our real payload sizes (our serialized rows were ~4KB, larger than the cluster's typical traffic). Redis is single-threaded per node; a neighboring service's pipeline burst became our latency. Worse, RDB fork stalls during `BGSAVE` produced 200-400ms pauses. Hits inherited that. Misses inherited it *and* still paid for Postgres.

Then there was the failure mode we only saw in production: our 60s TTLs were set at write time, so a traffic spike would populate thousands of keys within the same second and expire them within the same second, sixty seconds later. Every minute, a stampede of misses hit Postgres simultaneously. Our p99.9 spikes were synchronized TTL expiry, and no amount of Postgres tuning was going to fix them from that side.

## What we should have measured first

**Where the p99 actually was.** We never traced it. When we finally did, the DB accounted for 9ms of a 190ms p99. The rest was an N+1 fan-out in the serializer and TLS handshake churn against the auth service. We spent four months optimizing 5% of the tail.

**The real key distribution.** Replay a day of production query keys against a simulated LRU at your intended memory size. It takes an afternoon. It would have told us 61% instead of 95%, and 61% was never going to work.

**The break-even, explicitly.** A cache wins on latency only when

```
hit_latency × h + (miss_overhead + db_latency) × (1 − h) < db_latency
```

which reduces to `h > miss_overhead / (db_latency − hit_latency)`. With a 1.6ms miss penalty, 1.4ms from Postgres and 0.7ms from Redis, that needs a hit rate above 100%. The arithmetic was fatal before we wrote a line of code. Run it with p99 numbers too, not just medians — that's the version that would have stopped us.

**The dependency's own tail, on the instance you'll actually use.** Not the vendor's benchmark. Yours, with your payload sizes, your connection count, your noisy neighbors, persistence configured the way it's configured.

**What invalidation costs.** We treated TTLs as free. Jittered TTLs would have fixed the stampede; we never thought about it because we never modeled expiry as a load pattern.

## What we did instead

We added a covering index, fixed the N+1, and put PgBouncer in transaction mode in front of the primary. p99 went from 190ms to 48ms with no new moving parts.

We kept exactly one cache: the FX rate table, which takes 400ms to compute, changes four times a day, and has a hit rate of 99.8%. That one is doing real work.

Caching is a bet that recomputation is expensive and repetition is high. Measure both before you take the bet — and measure them at the tail, because the tail is where the bet gets settled.
