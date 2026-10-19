// Public front door for the EG2 query embedder. The browser page calls this; the Worker answers
// from cache when it can and otherwise forwards to one Cloudflare Container running server.cjs.
//
//   POST /embed {text, dims?}  -> {embedding, dims, tokens, ms, cached}
//   POST /warm                 -> 202; starts the container so the first real query is warm
//
// Abuse controls, in the order a request meets them: CORS origin allow-list (a browser-only
// courtesy, not a defense), per-IP rate limit, text length cap, then the cache. The container
// only ever sees the "query" prompt, so this is not a general-purpose embedding API.
import { Container, getContainer } from '@cloudflare/containers';

export class Embedder extends Container {
  defaultPort = 8080;
  sleepAfter = '10m';
  // onnxruntime defaults to one thread per visible core; under a 1-vCPU quota that thrashes and
  // doubles warm latency (203 vs 110 ms in RESULTS.md).
  envVars = { THREADS: '1' };
}

const MAX_CHARS = 512;
const DIMS = new Set([768, 512, 256, 128]);
const CACHE_TTL = 86400;

function cors(env, request) {
  const origin = request.headers.get('origin') ?? '';
  const allowed = (env.ALLOWED_ORIGINS ?? '*').split(',').map((s) => s.trim());
  const allow = allowed.includes('*') ? '*' : allowed.includes(origin) ? origin : null;
  return allow ? {
    'access-control-allow-origin': allow,
    'access-control-allow-methods': 'POST, OPTIONS',
    'access-control-allow-headers': 'content-type',
    'access-control-max-age': '86400',
    vary: 'origin',
  } : null;
}

const reply = (status, body, headers) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

async function sha256(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    // Operator route: stop the container now. The token is a Worker secret (ADMIN_TOKEN); without
    // it the route does not exist.
    if (url.pathname === '/admin/stop' && request.method === 'POST') {
      if (!env.ADMIN_TOKEN || request.headers.get('x-admin-token') !== env.ADMIN_TOKEN) return reply(404, { error: 'not found' }, {});
      // RPC to the Container class's own methods; a fetch would be proxied to the container.
      const stub = getContainer(env.EMBEDDER);
      const before = await stub.getState();
      // ?destroy=1 sends SIGKILL, for a process that ignores SIGTERM.
      if (url.searchParams.get('destroy') === '1') await stub.destroy(); else await stub.stop();
      return reply(200, { stopped: true, destroyed: url.searchParams.get('destroy') === '1', before }, {});
    }
    const ch = cors(env, request);
    if (!ch) return reply(403, { error: 'origin not allowed' }, {});
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: ch });
    if (request.method !== 'POST' || !['/embed', '/warm'].includes(url.pathname)) return reply(404, { error: 'not found' }, ch);

    const ip = request.headers.get('cf-connecting-ip') ?? 'unknown';
    const { success } = await env.EMBED_RL.limit({ key: ip });
    if (!success) return reply(429, { error: 'rate limited' }, { ...ch, 'retry-after': '60' });

    const container = getContainer(env.EMBEDDER);
    if (url.pathname === '/warm') {
      // Starting the container takes seconds; answer now and let the start finish in the background.
      ctx.waitUntil(container.fetch(new Request('http://container/health')).catch(() => {}));
      return reply(202, { warming: true }, ch);
    }

    let body;
    try { body = await request.json(); } catch { return reply(400, { error: 'body must be JSON' }, ch); }
    const { text, dims = 768 } = body ?? {};
    if (typeof text !== 'string' || !text.trim()) return reply(400, { error: 'text must be a non-empty string' }, ch);
    if (text.length > MAX_CHARS) return reply(413, { error: `text over ${MAX_CHARS} characters` }, ch);
    if (!DIMS.has(dims)) return reply(400, { error: 'dims must be 768, 512, 256 or 128' }, ch);

    // Cache on the exact text: normalizing whitespace or case would change the tokens and so the vector.
    const key = new Request(`https://embed-cache.internal/${env.MODEL_REV}/${dims}/${await sha256(text)}`);
    const hit = await caches.default.match(key);
    if (hit) return reply(200, { ...(await hit.json()), cached: true }, ch);

    const res = await container.fetch(new Request('http://container/embed', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, dims, prompt: 'query' }),
    }));
    if (!res.ok) return reply(502, { error: `embedder answered ${res.status}` }, ch);
    const out = await res.json();
    const payload = { embedding: out.embedding, dims: out.dims, tokens: out.tokens, ms: out.ms, model: env.MODEL_REV };
    ctx.waitUntil(caches.default.put(key, new Response(JSON.stringify(payload), {
      headers: { 'content-type': 'application/json', 'cache-control': `max-age=${CACHE_TTL}` },
    })));
    return reply(200, { ...payload, cached: false }, ch);
  },
};
