// End to end: grey fixture lines -> web/engine.js -> probs and greedy text, against the
// NumPy reference (fixture probs and texts). Model packs from run/pack.py in web/models/.
import { Gpu, compare } from "../gpu.js";
import { TessModel, recognize } from "../engine.js";

export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const man = await (await fetch(base + "manifest.json")).json();
    const gray = new Float32Array(await (await fetch(base + man.files.gray.file)).arrayBuffer());
    const want = new Float32Array(await (await fetch(base + man.files.probs.file)).arrayBuffer());
    const model = await TessModel.load(gpu, `/web/models/${man.case.replace("_", "-")}/`);
    const lines = man.lines.map((l) => ({ data: Uint8Array.from(gray.subarray(l.in_off, l.in_off + man.H * l.W)), width: l.W, height: man.H }));
    await recognize(gpu, model, lines.slice(0, 1), { decoder: "greedy" }); // warm-up: pipeline compiles
    const out = await recognize(gpu, model, lines, { wantProbs: true, decoder: "greedy" });
    const got = new Float32Array(want.length);
    man.lines.forEach((l, i) => got.set(out[i].probs, l.t_off * man.C));
    const c = compare(got, want, 1e-4);
    let argmaxOff = 0;
    for (let t = 0; t < man.total.t; t++) {
      let a = 0, b = 0;
      for (let k = 1; k < man.C; k++) {
        if (got[t * man.C + k] > got[t * man.C + a]) a = k;
        if (want[t * man.C + k] > want[t * man.C + b]) b = k;
      }
      if (a !== b) argmaxOff++;
    }
    const textsSame = out.filter((o, i) => o.text === man.texts[i]).length;
    res.push({ case: man.case, ...c, argmaxOff, textsSame: `${textsSame}/${out.length}`, ms: out.ms, sample: out[0].text });
  }
  return { ok: res.every((r) => r.maxDiff <= 1e-4 && r.argmaxOff === 0 && r.textsSame.split("/")[0] === r.textsSame.split("/")[1]), adapter: gpu.info, res };
}
