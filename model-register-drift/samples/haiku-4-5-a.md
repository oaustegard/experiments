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
