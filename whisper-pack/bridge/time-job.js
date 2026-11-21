// GPU Bridge job: load packs given as inputs, check tokens against the reference, time encode and decode.
// inputs: sha (experiments commit), packs = [{slug, prompt, ids}], and per pack <slug>_manifest (json),
// <slug>_weights (u8), <slug>_mel (f32), <slug>_enc (f32).
const { sha, packs, reps = 3 } = ctx.inputs;
const { Whisper } = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${sha}/whisper-pack/web/whisper.js`);
const device = await ctx.gpu();
const errors = [];
device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
const out = { adapter: ctx.describe(), rows: [], errors };
for (const [pi, pk] of packs.entries()) {
  const key = pk.slug.replace(/[^a-z0-9]/gi, "_");
  const wbytes = ctx.inputs[key + "_weights"];
  let t = performance.now();
  const w = Whisper.fromBytes(device, ctx.inputs[key + "_manifest"], wbytes.buffer.slice(wbytes.byteOffset, wbytes.byteOffset + wbytes.byteLength));
  await device.queue.onSubmittedWorkDone();
  const loadMs = performance.now() - t;
  const mel = ctx.inputs[key + "_mel"], encRef = ctx.inputs[key + "_enc"];
  const encMs = [];
  for (let r = 0; r < reps; r++) { t = performance.now(); await w.encode(mel); encMs.push(performance.now() - t); ctx.progress((pi + (r + 1) / (reps + 2)) / packs.length, pk.slug + " encode"); }
  const enc = await w.probe("enc", encRef.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < enc.length; i++) { dot += enc[i] * encRef[i]; na += enc[i] * enc[i]; nb += encRef[i] * encRef[i]; }
  const decMs = [];
  let ids = [];
  for (let r = 0; r < reps; r++) { t = performance.now(); ids = await w.decode(pk.prompt); decMs.push((performance.now() - t) / (ids.length + pk.prompt.length)); }
  const same = ids.length === pk.ids.length && ids.every((v, i) => v === pk.ids[i]);
  const row = { slug: pk.slug, scheme: pk.scheme, loadMs, encMs, encCos: dot / Math.sqrt(na * nb), msPerToken: decMs, tokens: ids.length, same };
  ctx.log(JSON.stringify(row));
  out.rows.push(row);
}
return out;
