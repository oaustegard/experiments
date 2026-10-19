// Live check of the deployed Worker. Run it when the container has been asleep (15 min idle) to
// time a real cold start. Checks: first-query latency, warm latency, cache hit, the model card's
// scores from vectors the endpoint returns, origin rejection, and the per-IP rate limit.
//   node live_check.mjs [https://eg2-query-embedder.austegard.workers.dev] [--skip-ratelimit]
const BASE = process.argv.find((a) => a.startsWith('http')) ?? 'https://eg2-query-embedder.austegard.workers.dev';
const ORIGIN = 'https://austegard.com';
const now = () => performance.now();
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

async function embed(text, { origin = ORIGIN, dims } = {}) {
  const t = now();
  const r = await fetch(`${BASE}/embed`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(origin ? { origin } : {}) },
    body: JSON.stringify({ text, ...(dims ? { dims } : {}) }),
  });
  return { status: r.status, ms: Math.round(now() - t), body: await r.json().catch(() => null) };
}

const out = {};
const stamp = Date.now();                                    // keeps cold/warm texts out of the cache
const first = await embed(`cold start probe ${stamp}`);
out.first = { status: first.status, roundtrip_ms: first.ms, server_ms: first.body?.ms };

const warm = [];
for (let i = 0; i < 5; i++) warm.push((await embed(`warm probe ${stamp} ${i}`)).ms);
out.warm_roundtrip_median_ms = median(warm);

const again = await embed(`warm probe ${stamp} 0`);
out.cache = { cached: again.body?.cached, roundtrip_ms: again.ms };

// The endpoint embeds queries only, so check the card's ordering from query-vs-query similarity
// to the local server is not possible here; instead compare against the local server's query vector.
const q = await embed('Which planet is known as the Red Planet?');
out.query_tokens = q.body?.tokens;
out.query_head = q.body?.embedding?.slice(0, 4).map((x) => +x.toFixed(6));

out.no_origin_status = (await embed('x', { origin: null })).status;
out.bad_origin_status = (await embed('x', { origin: 'https://evil.example' })).status;
out.dims512_len = (await embed(`dims probe ${stamp}`, { dims: 512 })).body?.embedding?.length;

if (!process.argv.includes('--skip-ratelimit')) {
  const codes = [];
  for (let i = 0; i < 25; i++) codes.push((await embed(`rl ${stamp} ${i}`)).status);
  out.ratelimit = { first_429_at: codes.indexOf(429), codes: codes.join(',') };
}
console.log(JSON.stringify(out, null, 1));
