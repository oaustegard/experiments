# Why we removed our cache

Four months ago we put a Redis read-through cache in front of our primary Postgres database. Last week we deleted it. Our p99 latency dropped by 38% the moment the deploy finished.

This is a post about how a textbook optimization made things worse, why it took us so long to notice, and what we should have measured before we started.

## The setup

Our service is a fairly boring CRUD API for account and entitlement data. Around 4,200 requests per second at peak, roughly 92% reads. The read path was: request → application → Postgres (a single primary, `db.r6g.2xlarge`, two read replicas).

Before the cache, our latency profile looked like this:

| Percentile | Latency |
|---|---|
| p50 | 4.1 ms |
| p95 | 11 ms |
| p99 | 31 ms |
| p99.9 | 140 ms |

p99 of 31 ms isn't bad, but we had a target of 25 ms and a roadmap full of new features that would add queries. Caching seemed like the obvious lever. We had a Redis cluster already in place for sessions, and we knew our access pattern was skewed: the top 5% of accounts generated about 60% of reads.

We built a read-through layer: check Redis by key, on miss fall through to Postgres, populate Redis with a 5-minute TTL, write-invalidate on updates. Standard stuff. We shipped it behind a flag and turned it on for 100% of traffic within a week.

## What we saw

Cache hit rate stabilized at 87%. Postgres CPU dropped from ~55% to ~22%. Both graphs went on the team Slack channel with celebration emoji.

Here's what the latency table looked like after two weeks:

| Percentile | Before | After |
|---|---|---|
| p50 | 4.1 ms | 1.9 ms |
| p95 | 11 ms | 9 ms |
| p99 | 31 ms | 44 ms |
| p99.9 | 140 ms | 410 ms |

p50 got dramatically better. p99 got worse. p99.9 got *much* worse.

We didn't catch this for about six weeks, for an embarrassing reason: our latency dashboard defaulted to p50 and p95, and both of those improved. The p99 alert threshold was set at 50 ms, so 44 ms never fired. We only found it when a downstream team complained that their timeouts to us had increased.

## Why the tail got worse

Once we started looking, there were four overlapping causes.

**1. Cache misses paid for two round trips.** A miss went to Redis (~0.8 ms including the network hop, since Redis was in a different AZ than 2/3 of our app pods), then to Postgres, then back to Redis to populate. A request that previously cost 4 ms now cost about 6 ms on a miss. 13% of requests missing means roughly 550 requests per second were strictly slower than before. That alone pushed the p95–p99 region up.

**2. Postgres was already caching the hot set.** This was the one that stung. Our working set was about 9 GB; `shared_buffers` was 16 GB. Almost every "hot" read that Redis was now serving had previously been served from Postgres memory in under 3 ms. We weren't replacing disk I/O with memory access. We were replacing one memory access with a different memory access plus a network hop. The Postgres CPU drop was real, but it wasn't translating into latency because CPU was never the bottleneck.

**3. TTL expiry synchronized our misses.** Our hottest ~200 keys were requested hundreds of times per second. When one expired, every request in the next 6–8 ms missed simultaneously, all hit Postgres for the same row, and all tried to repopulate Redis. We measured bursts of 40+ identical queries within a 10 ms window. Postgres handled them fine individually, but the connection pool (pgbouncer, 60 server connections) would briefly saturate, and unrelated requests queued behind the stampede. This is a well-known problem with well-known fixes (jittered TTLs, request coalescing, probabilistic early refresh). We hadn't implemented any of them.

**4. Redis had its own tail.** Every 15 minutes, our Redis instance forked for an RDB snapshot. On a 12 GB dataset, the fork itself took 30–60 ms of stalled command processing. We also had a handful of entitlement blobs that serialized to 300–600 KB, and reading those on a single-threaded Redis blocked everyone else for a millisecond or two. Neither of these was catastrophic. Both added spikes that Postgres, with its multi-process architecture and mature I/O scheduling, simply did not have.

## What we tried before giving up

We spent about three weeks on remediation: jittered TTLs (helped p99.9 by ~30%), disabling RDB in favor of AOF with `everysec` (removed the fork stalls), and moving Redis into the same AZ as the app tier (saved ~0.4 ms per call).

After all of that, p99 was 34 ms. Still worse than the 31 ms we started with, with an extra system to operate, an invalidation code path that had already caused one stale-data incident, and a Redis bill of about $1,100/month.

So we turned the flag off.

## What we learned

**Measure the thing you're trying to fix.** We wanted to fix p99. We instrumented p50. If we'd looked at p99 on day two, we'd have saved four months.

**Know what your database is actually doing.** A cache in front of a database that's already serving from memory is a cache in front of a cache. Check your buffer hit ratio before you build one. Ours was 99.6%.

**Average hit rate is a vanity metric.** 87% sounds great. But latency is determined by the distribution of misses, not the count of hits, and our misses were clustered in exactly the worst way.

**Every hop has a tail.** Adding a component never removes latency variance; at best it trades variance in one place for variance in another. If your original bottleneck isn't actually slow, you've just added tail.

We may cache again someday—probably in-process, probably for a specific query we can prove is expensive. But we'll look at p99 first.
