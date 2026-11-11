// Firehose side of the demo page: live posts from Bluesky's Jetstream (JSON over WebSocket), the coarse
// filters and the exact-rate sampler. No DOM here.
//
//   const js = new Jetstream({ onPost: (p) => ..., onState: (state, endpoint, note) => ... });
//   js.start();  ...  js.stop();
//   const keep = makeFilter({ lang: "en", replies: false, minlen: 20, include: "", exclude: "" });
//   const s = makeSampler(10);  s() -> true on the 1st, 11th, 21st... call
//
// Jetstream events: { did, time_us, kind: "commit" | "identity" | "account", commit?: { operation,
// collection, rkey, record } }. Only commits that create an app.bsky.feed.post reach onPost, as
// { did, rkey, time_us, text, langs, reply, createdAt }.

export const ENDPOINTS = [
  "wss://jetstream2.us-east.bsky.network",
  "wss://jetstream1.us-east.bsky.network",
  "wss://jetstream2.us-west.bsky.network",
  "wss://jetstream1.us-west.bsky.network",
];
const PATH = "/subscribe?wantedCollections=app.bsky.feed.post";

// Event -> post, or null for anything that is not a post create.
export function toPost(ev) {
  if (!ev || ev.kind !== "commit") return null;
  const c = ev.commit;
  if (!c || c.operation !== "create" || c.collection !== "app.bsky.feed.post" || !c.record) return null;
  const r = c.record;
  return { did: String(ev.did || ""), rkey: String(c.rkey || ""), time_us: +ev.time_us || 0, text: typeof r.text === "string" ? r.text : "",
    langs: Array.isArray(r.langs) ? r.langs.filter((x) => typeof x === "string") : [], reply: !!r.reply,
    createdAt: typeof r.createdAt === "string" ? r.createdAt : "" };
}

// One connection at a time; on a failure or a drop, the next endpoint in ENDPOINTS after a backoff
// (0.5 s doubling to 30 s, reset once a connection delivers a message), resuming at the last time_us seen.
export class Jetstream {
  constructor({ onPost, onEvent = null, onState = null, endpoints = ENDPOINTS } = {}) {
    Object.assign(this, { onPost, onEvent, onState, endpoints });
    this.ws = null; this.i = 0; this.failures = 0; this.cursor = 0; this.timer = null;
    this.state = "idle"; this.endpoint = null; this.running = false;
  }
  start() {
    if (this.running) return;
    this.running = true; this.failures = 0;
    this.connect();
  }
  stop(note = "stopped") {
    this.running = false;
    clearTimeout(this.timer); this.timer = null;
    const ws = this.ws; this.ws = null;
    if (ws) { ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null; try { ws.close(1000); } catch { /* already closed */ } }
    this.set("stopped", note);
  }
  set(state, note = "") { this.state = state; this.onState?.(state, this.endpoint, note); }
  connect() {
    if (!this.running) return;
    const base = this.endpoints[this.i % this.endpoints.length];
    this.endpoint = base.replace(/^wss:\/\//, "");
    const url = base + PATH + (this.cursor ? `&cursor=${this.cursor}` : "");
    this.set(this.failures ? "reconnecting" : "connecting");
    let ws;
    try { ws = new WebSocket(url); } catch (e) { this.retry(String(e?.message || e)); return; }
    this.ws = ws;
    let got = false;
    ws.onopen = () => this.set("open");
    ws.onmessage = (m) => {
      let ev;
      try { ev = JSON.parse(m.data); } catch { return; }
      const t = +ev?.time_us || 0;
      if (t && t <= this.cursor) return;     // replayed after a resume
      if (t) this.cursor = t;
      if (!got) { got = true; this.failures = 0; }
      this.onEvent?.(ev);
      const p = toPost(ev);
      if (p) this.onPost(p);
    };
    ws.onerror = () => {};                   // a close always follows
    ws.onclose = (e) => { if (this.ws === ws) { this.ws = null; this.retry(`closed (${e?.code ?? "?"})`); } };
  }
  retry(why) {
    if (!this.running) return;
    this.failures++;
    this.i++;
    const ms = Math.min(30_000, 500 * 2 ** (this.failures - 1)) * (0.8 + 0.4 * Math.random());
    this.set("reconnecting", `${why}; trying ${this.endpoints[this.i % this.endpoints.length].replace(/^wss:\/\//, "")} in ${(ms / 1000).toFixed(1)} s`);
    this.timer = setTimeout(() => this.connect(), ms);
  }
}

const words = (s) => String(s || "").split(",").map((x) => x.trim().toLowerCase()).filter(Boolean);

// Coarse filters, applied before sampling. lang: "any" or a code matched against record.langs ("en" also
// matches "en-GB"); replies: keep replies; minlen: characters (code points); include: at least one of these
// comma-separated words must occur in the text; exclude: none may. Case-insensitive substring matches.
export function makeFilter({ lang = "en", replies = false, minlen = 20, include = "", exclude = "" } = {}) {
  const inc = words(include), exc = words(exclude), code = String(lang || "any").toLowerCase();
  return (p) => {
    if (!replies && p.reply) return false;
    if (code !== "any" && !p.langs.some((l) => { l = l.toLowerCase(); return l === code || l.startsWith(code + "-"); })) return false;
    let n = 0;
    for (const _ of p.text) if (++n >= minlen) break;
    if (n < minlen) return false;
    if (inc.length || exc.length) {
      const t = p.text.toLowerCase();
      if (inc.length && !inc.some((w) => t.includes(w))) return false;
      if (exc.some((w) => t.includes(w))) return false;
    }
    return true;
  };
}

// Exactly one in `every`, by counter: true on calls 1, every+1, 2*every+1, ...
export function makeSampler(every) {
  let n = 0;
  every = Math.max(1, Math.floor(every) || 1);
  return () => (n++ % every) === 0;
}
