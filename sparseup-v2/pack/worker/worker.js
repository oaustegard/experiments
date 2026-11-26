// model-packs: serves browser model packs from the austegard-media R2 bucket at
// https://model-packs.austegard.workers.dev/<key>, for pages and extensions that need a
// production URL (r2.dev public URLs are rate-limited and meant for development).
//
// Only keys under PREFIXES are served; everything else is 404. GET and HEAD, Range requests
// (206), conditional requests (304), CORS for any origin, and a one-year immutable cache, since
// every prefix is a versioned, never-rewritten pack.

const PREFIXES = ["sparseup-v2/int8-v1/"];

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Allow-Headers": "Range, If-None-Match",
  "Access-Control-Expose-Headers": "Content-Length, Content-Range, ETag, Accept-Ranges",
  "Access-Control-Max-Age": "86400",
};

export default {
  async fetch(req, env) {
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
    if (req.method !== "GET" && req.method !== "HEAD") return new Response("method not allowed", { status: 405, headers: { ...CORS, Allow: "GET, HEAD, OPTIONS" } });
    const key = decodeURIComponent(new URL(req.url).pathname.slice(1));
    if (key.includes("..") || !PREFIXES.some((p) => key.startsWith(p) && key.length > p.length)) {
      return new Response("not found", { status: 404, headers: CORS });
    }
    const obj = req.method === "HEAD"
      ? await env.BUCKET.head(key)
      : await env.BUCKET.get(key, { range: req.headers, onlyIf: req.headers });
    if (!obj) return new Response("not found", { status: 404, headers: CORS });

    const h = new Headers(CORS);
    obj.writeHttpMetadata(h);
    h.set("ETag", obj.httpEtag);
    h.set("Accept-Ranges", "bytes");
    h.set("Cache-Control", "public, max-age=31536000, immutable");
    if (req.method === "HEAD") { h.set("Content-Length", String(obj.size)); return new Response(null, { headers: h }); }
    if (!("body" in obj)) return new Response(null, { status: 304, headers: h });   // If-None-Match matched
    if (obj.range) {
      const off = obj.range.offset ?? (obj.range.suffix != null ? obj.size - obj.range.suffix : 0);
      const len = obj.range.length ?? (obj.range.suffix ?? obj.size - off);
      h.set("Content-Range", `bytes ${off}-${off + len - 1}/${obj.size}`);
      h.set("Content-Length", String(len));
      return new Response(obj.body, { status: 206, headers: h });
    }
    h.set("Content-Length", String(obj.size));
    return new Response(obj.body, { headers: h });
  },
};
