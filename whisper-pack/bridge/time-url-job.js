// GPU Bridge job: load hosted packs by URL, check tokens against the reference, time load, encode and decode.
// inputs: sha (experiments commit), packs = [{slug, url, prompt, ids}], <slug>_mel (f32), <slug>_enc (f32), reps.
const { sha, packs, reps = 3 } = ctx.inputs;
const { Whisper } = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${sha}/whisper-pack/web/whisper.js`);
const device = await ctx.gpu();
const errors = [];
device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
const out = { rows: [], errors };
for (const [pi, pk] of packs.entries()) {
  const key = pk.slug.replace(/[^a-z0-9]/gi, "_");
  let t = performance.now();
  const w = await Whisper.load(device, pk.url, { onProgress: (a, b) => ctx.progress((pi + 0.5 * a / b) / packs.length, pk.slug + " download") });
  await device.queue.onSubmittedWorkDone();
  const loadMs = performance.now() - t;
  const mel = ctx.inputs[key + "_mel"], encRef = ctx.inputs[key + "_enc"];
  const encMs = [];
  for (let r = 0; r < reps; r++) { t = performance.now(); await w.encode(mel); encMs.push(performance.now() - t); ctx.progress((pi + 0.5 + 0.4 * (r + 1) / reps) / packs.length, pk.slug + " encode"); }
  const enc = await w.probe("enc", encRef.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < enc.length; i++) { dot += enc[i] * encRef[i]; na += enc[i] * enc[i]; nb += encRef[i] * encRef[i]; }
  const decMs = [];
  let ids = [];
  for (let r = 0; r < reps; r++) { t = performance.now(); ids = await w.decode(pk.prompt); decMs.push((performance.now() - t) / (ids.length + pk.prompt.length)); }
  const same = ids.length === pk.ids.length && ids.every((v, i) => v === pk.ids[i]);
  const row = { slug: pk.slug, scheme: pk.scheme, loadMs, encMs, encCos: dot / Math.sqrt(na * nb), msPerToken: decMs, tokens: ids.length, same, ids };
  ctx.log(JSON.stringify(row));
  out.rows.push(row);
}
return out;
