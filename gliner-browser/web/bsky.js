// Bluesky side of the demo page: read posts from the public AppView (no login) and group extracted
// entities. No DOM here.
//
//   const src = await resolveSource("alice.bsky.social");      // or a profile / list / feed URL, an at:// URI or a DID
//   for await (const page of fetchPosts(src, { max: 500, reposts: false })) ...   // [{ uri, url, text, author, ... }]
//   const g = new EntityGroups();  g.add(post, ents);  g.view()  // per label, entities by post count

export const APPVIEW = "https://public.api.bsky.app/xrpc";

async function xrpc(method, params, signal) {
  const r = await fetch(`${APPVIEW}/${method}?` + new URLSearchParams(params), { signal });
  if (!r.ok) {
    let msg = "";
    try { const j = await r.json(); msg = j.message || j.error || ""; } catch { /* not JSON */ }
    throw new Error(`${method}: HTTP ${r.status}${msg ? ` (${msg})` : ""}`);
  }
  return r.json();
}

async function did(actor, signal) {
  actor = actor.replace(/^@/, "");
  if (actor.startsWith("did:")) return actor;
  return (await xrpc("com.atproto.identity.resolveHandle", { handle: actor }, signal)).did;
}

// Input -> { kind: "author" | "list" | "feed", param, label }. Accepts a handle (with or without @), a DID,
// a bsky.app profile, list or feed URL, or an at:// URI of a list or a feed generator.
export async function resolveSource(input, signal) {
  const s = input.trim();
  if (!s) throw new Error("enter a handle, a list URL or a feed URL");
  let m = s.match(/^https?:\/\/(?:www\.)?bsky\.app\/profile\/([^/?#]+)(?:\/(lists|feed)\/([^/?#]+))?/i);
  if (m) {
    const [, actor, kind, rkey] = m;
    if (!kind) return { kind: "author", param: await did(actor, signal), label: `posts by @${actor.replace(/^@/, "")}` };
    const d = await did(actor, signal);
    return kind === "lists"
      ? { kind: "list", param: `at://${d}/app.bsky.graph.list/${rkey}`, label: `list ${rkey}` }
      : { kind: "feed", param: `at://${d}/app.bsky.feed.generator/${rkey}`, label: `feed ${rkey}` };
  }
  m = s.match(/^at:\/\/([^/]+)\/(app\.bsky\.graph\.list|app\.bsky\.feed\.generator)\/([^/?#]+)$/);
  if (m) {
    const uri = `at://${await did(m[1], signal)}/${m[2]}/${m[3]}`;
    return m[2] === "app.bsky.graph.list" ? { kind: "list", param: uri, label: `list ${m[3]}` } : { kind: "feed", param: uri, label: `feed ${m[3]}` };
  }
  m = s.match(/^at:\/\/([^/]+)\/?$/);
  const actor = m ? m[1] : s.replace(/^@/, "");
  if (!/^(did:[a-z]+:[A-Za-z0-9._:%-]+|[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+)$/.test(actor)) throw new Error(`not a handle, DID, or bsky.app / at:// link: ${s}`);
  return { kind: "author", param: await did(actor, signal), label: `posts by @${actor}` };
}

// https://bsky.app link for a post at-URI.
export function postUrl(uri, handle) {
  const m = uri.match(/^at:\/\/([^/]+)\/app\.bsky\.feed\.post\/([^/]+)$/);
  return m ? `https://bsky.app/profile/${handle || m[1]}/post/${m[2]}` : null;
}

// Pages of posts, newest first, until `max` posts (after skipping) or the end of the feed.
export async function* fetchPosts(src, { max = 200, reposts = false, signal } = {}) {
  const method = { author: "app.bsky.feed.getAuthorFeed", list: "app.bsky.feed.getListFeed", feed: "app.bsky.feed.getFeed" }[src.kind];
  const key = { author: "actor", list: "list", feed: "feed" }[src.kind];
  const seen = new Set();
  let cursor, n = 0;
  while (n < max) {
    const q = { [key]: src.param, limit: 100 };
    if (src.kind === "author") q.filter = "posts_with_replies";
    if (cursor) q.cursor = cursor;
    const d = await xrpc(method, q, signal);
    const page = [];
    for (const it of d.feed || []) {
      const reason = it.reason?.$type || "";
      if (!reposts && reason.includes("reasonRepost")) continue;
      const p = it.post, rec = p?.record || {};
      if (!p || seen.has(p.uri) || typeof rec.text !== "string" || !rec.text.trim()) continue;
      seen.add(p.uri);
      page.push({ uri: p.uri, url: postUrl(p.uri, p.author?.handle), text: rec.text, author: p.author?.handle || "",
        name: p.author?.displayName || "", createdAt: rec.createdAt || "", repost: reason.includes("reasonRepost") });
      if (++n >= max) break;
    }
    if (page.length) yield page;
    cursor = d.cursor;
    if (!cursor || !(d.feed || []).length) break;
  }
}

// Grouping key for an entity's surface form: Unicode NFKC, case-folded, leading @ and # stripped,
// whitespace collapsed, a trailing possessive ('s, ’s, or a bare ' / ’ after s) dropped.
export function normEntity(s) {
  let t = s.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();
  t = t.replace(/^[@#]+/, "");
  t = t.replace(/['’]s$/, "").replace(/(s)['’]$/, "$1");
  return t.trim();
}

// Entities grouped per label, then per normalised name, counting posts (an entity twice in one post counts once).
export class EntityGroups {
  constructor() { this.posts = []; this.g = new Map(); }
  // post: { text, url, ... }; ents: ner.extract's list for post.text
  add(post, ents) {
    const pi = this.posts.push({ ...post, ents }) - 1;
    for (const e of ents) {
      const key = normEntity(e.text);
      if (!key) continue;
      let byLabel = this.g.get(e.label);
      if (!byLabel) this.g.set(e.label, (byLabel = new Map()));
      let x = byLabel.get(key);
      if (!x) byLabel.set(key, (x = { label: e.label, key, forms: new Map(), posts: new Set(), mentions: 0 }));
      x.forms.set(e.text, (x.forms.get(e.text) || 0) + 1);
      x.posts.add(pi);
      x.mentions++;
    }
  }
  // [{ label, entities: [{ key, name, count, mentions, posts: [index] }] }], labels in `order` first;
  // entities by post count, then mentions, then name. name: the most frequent surface form (first seen on a tie).
  view(order = []) {
    const labels = [...new Set([...order.filter((l) => this.g.has(l)), ...this.g.keys()])];
    return labels.map((label) => ({
      label,
      entities: [...this.g.get(label).values()].map((x) => {
        let name = "", best = 0;
        for (const [f, c] of x.forms) if (c > best) { name = f; best = c; }
        return { key: x.key, name, count: x.posts.size, mentions: x.mentions, posts: [...x.posts] };
      }).sort((a, b) => b.count - a.count || b.mentions - a.mentions || a.name.localeCompare(b.name)),
    }));
  }
}
