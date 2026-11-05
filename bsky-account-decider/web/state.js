// The text the model reads for a Bluesky account and for one post. Mirrors run/state.py line for line;
// run/check_state.mjs compares the two on saved accounts.
export const POST_CHARS = 140;
const EMBED = { images: "image", video: "video", external: "link", record: "quote", recordWithMedia: "quote" };

// Python's f"{x:.1f}" rounds the float's exact binary value, and an exact tie to even (2.25 -> "2.2");
// toFixed agrees except on exact ties, where it rounds up. A tenths tie can only be exact when x*4 is an
// odd integer (x ends in .25 or .75), so handle that case and leave the rest to toFixed.
const fix1 = (x) => {
  if (Number.isInteger(x * 4) && Math.abs(x * 4) % 2 === 1) {
    const s = x * 10, f = Math.floor(s);
    return ((f % 2 === 0 ? f : f + 1) / 10).toFixed(1);
  }
  return x.toFixed(1);
};
const tenths = (n, unit) => fix1(n / unit);
const num = (n) => { n = n || 0; return n >= 1e6 ? tenths(n, 1e6) + "M" : n >= 1e3 ? tenths(n, 1e3) + "k" : String(n); };
const clean = (t, n = POST_CHARS) => {
  // Python's str.split(): JS \s minus U+FEFF, plus the \x1c-\x1f separators
  t = (t || "").split(/(?:[^\S\uFEFF]|[\u001c-\u001f])+/).filter(Boolean).join(" ");
  const cps = [...t];   // Python slices by code point
  return cps.length <= n ? t : cps.slice(0, n - 1).join("") + "…";
};
const ts = (s) => { const d = new Date(s); return isNaN(d) ? null : d; };
const embedTag = (e) => (e ? ` [${EMBED[e] ?? e}]` : "");

export function accountState(acct, nPosts = 12) {
  const p = acct.profile, items = acct.items.slice(0, nPosts);
  const lines = [`@${p.handle}` + (p.displayName ? ` · ${clean(p.displayName, 80)}` : "")];
  if (p.description) lines.push("Bio: " + clean(p.description, 300));
  const created = (p.createdAt || "").slice(0, 7);
  lines.push(`Account: created ${created || "unknown"}, ${num(p.followersCount)} followers, follows ${num(p.followsCount)}, ${num(p.postsCount)} posts`);
  const t = items.map((i) => ts(i.indexedAt || i.createdAt)).filter(Boolean);
  const kinds = { post: 0, reply: 0, repost: 0 };
  for (const i of items) kinds[i.kind]++;
  const span = t.length > 1 ? (Math.max(...t) - Math.min(...t)) / 86400000 : 0;
  lines.push(`Recent activity: ${items.length} items over ${fix1(span)} days (${kinds.post} posts, ${kinds.reply} replies, ${kinds.repost} reposts)`);
  items.forEach((it, k) => {
    const tag = it.kind !== "repost" ? it.kind : `repost of @${it.author || "?"}`;
    lines.push(`${k + 1}. [${tag}]${embedTag(it.embed)} ${clean(it.text) || "(no text)"}`);
  });
  return lines.join("\n");
}

export function fitAccountState(acct, countTokens, budget = 830, nPosts = 12) {
  for (let n = nPosts; n > 0; n--) {
    const s = accountState(acct, n);
    if (countTokens(s) <= budget) return { state: s, n };
  }
  return { state: accountState(acct, 1), n: 1 };
}

// A reply carrying its parent ({ author, text }) gets the parent appended (run/state.py post_state).
export const postState = (it) => `[${it.kind}]${embedTag(it.embed)} ${clean(it.text, 300) || "(no text)"}` +
  (it.parent ? ` (in reply to @${it.parent.author || "?"}: ${clean(it.parent.text) || "(no text)"})` : "");

// Fetch an account the way run/collect.py does, from the public AppView.
const PUBLIC = "https://public.api.bsky.app/xrpc";
export async function fetchAccount(actor) {
  const get = async (ep, q) => {
    const r = await fetch(`${PUBLIC}/${ep}?` + new URLSearchParams(q));
    if (!r.ok) throw new Error(`${ep}: HTTP ${r.status}`);
    return r.json();
  };
  const p = await get("app.bsky.actor.getProfile", { actor });
  const f = await get("app.bsky.feed.getAuthorFeed", { actor: p.did, limit: 40, filter: "posts_with_replies" });
  const items = [];
  for (const it of f.feed || []) {
    const post = it.post, rec = post.record || {}, reason = (it.reason || {}).$type || "";
    const kind = reason.includes("reasonRepost") ? "repost" : rec.reply ? "reply" : "post";
    if (kind === "repost" && post.author.did === p.did) continue;
    const emb = (post.embed || {}).$type || "";
    const par = kind === "reply" ? (it.reply || {}).parent : null;   // notFoundPost / blockedPost have no record
    items.push({ kind, text: rec.text || "", createdAt: rec.createdAt || "",
                 indexedAt: (it.reason || {}).indexedAt || post.indexedAt || "",
                 author: kind === "repost" ? post.author.handle : null,
                 embed: emb ? emb.split(".").pop().split("#")[0] : null, uri: post.uri,
                 parent: par && (par.$type || "").endsWith("#postView") ? { author: par.author.handle, text: (par.record || {}).text || "" } : null });
    if (items.length >= 20) break;
  }
  const profile = { did: p.did, handle: p.handle, displayName: p.displayName, description: p.description,
                    followersCount: p.followersCount, followsCount: p.followsCount, postsCount: p.postsCount,
                    createdAt: p.createdAt, avatar: p.avatar };
  return { did: p.did, profile, items };
}
