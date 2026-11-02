import fs from "fs";
import { emitInstanced, createRenderer } from "../web/instanced.js";
const j = JSON.parse(fs.readFileSync("scratch/speed_in.json")).mona_2048;
const r = { P: Int32Array.from(j.P), bg: Int32Array.from(j.bg), A: j.A, n: j.n };
// the same 2,048 splats as vec4u (as shipped) and, for comparison, only the first 1,024 (plain u32 array)
const spec = { shaders: [{ id: "mona_1024_u32", n: 1024, code: emitInstanced(r, 1024) }, { id: "mona_2048_vec4u", n: 2048, code: emitInstanced(r) }] };
fs.writeFileSync("bridge/diag_spec.json", JSON.stringify(spec));
fs.writeFileSync("bridge/diag_job.js", `// GPU Bridge job: where does the instanced renderer stall on WebKit? Logs each step with its time.
const spec = JSON.parse(new TextDecoder().decode(ctx.inputs.spec));
const ad = await navigator.gpu.requestAdapter();
const dev = await ad.requestDevice({ requiredFeatures: ad.features.has("float32-blendable") ? ["float32-blendable"] : [] });
ctx.log("float32-blendable " + dev.features.has("float32-blendable"));
${createRenderer.toString()}
const T = () => performance.now(); const out = [];
for (const s of spec.shaders) {
  let t = T(); const mod = dev.createShaderModule({ code: s.code }); const info = await mod.getCompilationInfo();
  ctx.log(s.id + " compile " + (T() - t).toFixed(0) + " ms, messages " + JSON.stringify(info.messages.map((m) => m.message).slice(0, 3)));
  t = T(); const rd = createRenderer(dev, "rgba8unorm", s.code); ctx.log(s.id + " pipelines " + (T() - t).toFixed(0) + " ms");
  const tex = dev.createTexture({ size: [768, 512], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  for (const n of [16, 256, s.n]) {
    t = T(); rd.draw(tex.createView(), 768, 512, n);
    const done = await Promise.race([dev.queue.onSubmittedWorkDone().then(() => "ok"), new Promise((r) => setTimeout(() => r("timeout"), 8000))]);
    ctx.log(s.id + " draw n=" + n + " " + done + " " + (T() - t).toFixed(0) + " ms"); out.push({ id: s.id, n, done, ms: T() - t });
    if (done !== "ok") break;
  }
}
return { out };
`);
console.log("ok");
