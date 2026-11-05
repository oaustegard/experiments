// GPU Bridge job body: time the account decider on a real GPU with the page's own code path.
// Inputs: weights (u8), manifest / tokenizer / fixtures (u8 JSON bytes), cfg = { sha, reps }.
// fixtures = { accounts: [{ did, profile, items }], questions, post_question, ref: [{ account_type: [...logits] }] }
const { cfg } = ctx.inputs;
const dec = new TextDecoder();
const json = (k) => JSON.parse(dec.decode(ctx.inputs[k]));
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
ctx.progress(0.02, "loading code");
const { deciderFrom } = await ctx.importModule(gh + "browser-decider/web/decider.js");
const { fitAccountState, postState } = await ctx.importModule(gh + "bsky-account-decider/web/state.js");
const manifest = json("manifest"), tj = json("tokenizer"), fx = json("fixtures");
const device = await ctx.gpu();
const t0 = performance.now();
const d = await deciderFrom({ manifest, tj, data: ctx.inputs.weights }, { device });
const setupMs = performance.now() - t0;
const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const p90 = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length * 0.9)]; };
const run = async (a) => {
  const { state, n } = fitAccountState(a, d.countTokens);
  const items = a.items.slice(0, n);
  return d.decideMany([{ state, questions: fx.questions }, ...items.map((it) => ({ state: postState(it), questions: fx.post_question }))]);
};
// parity: account_type probabilities vs the NumPy reference (run/pack_page.py Ref), per fixture account
let maxDiff = 0, agree = 0;
const tokens = [];
for (let i = 0; i < fx.accounts.length; i++) {
  const [acc] = await run(fx.accounts[i]);
  const p = Object.values(acc.account_type.probabilities), r = fx.ref[i].account_type;
  p.forEach((x, k) => { maxDiff = Math.max(maxDiff, Math.abs(x - r[k])); });
  agree += p.indexOf(Math.max(...p)) === r.indexOf(Math.max(...r));
  const { state, n } = fitAccountState(fx.accounts[i], d.countTokens);
  tokens.push(d.countTokens(state) * 5 + fx.accounts[i].items.slice(0, n).reduce((s, it) => s + d.countTokens(postState(it)), 0));
}
ctx.progress(0.2, "timing");
const ms = [];
for (let r = 0; r < (cfg.reps ?? 5); r++) {
  for (const a of fx.accounts) { const t = performance.now(); await run(a); ms.push(performance.now() - t); }
  ctx.progress(0.2 + 0.8 * (r + 1) / (cfg.reps ?? 5), "timing");
}
return { setupMs, parity: { maxAbsProbDiff: maxDiff, argmaxAgree: agree, n: fx.accounts.length },
         accountMsMedian: med(ms), accountMsP90: p90(ms), approxTokensPerAccountMedian: med(tokens), runs: ms.length };
