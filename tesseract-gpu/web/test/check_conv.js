// Parity of the Convolve+FC+tanh and Maxpool kernels (web/kernels/conv.js) with the packed
// fixtures from run/make_fixtures.py. Float cases: max|diff| <= 1e-4. Int case: no entry off
// by more than 1, and at most 0.1% of entries off by exactly 1.
import { Gpu, loadCase, compare } from "../gpu.js";
import { recordConv, recordPool } from "../kernels/conv.js";

function intStats(got, want) {
  let off1 = 0, offGt1 = 0;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (e > 1) offGt1++;
    else if (e > 0) off1++;
  }
  return { off1, offGt1 };
}

function stage(caseName, stageName, got, want, isInt) {
  const c = compare(got, want, 1e-4);
  if (!isInt) {
    return { case: caseName, stage: stageName, ...c, ok: c.maxDiff <= 1e-4 };
  }
  const s = intStats(got, want);
  return {
    case: caseName, stage: stageName, ...c, ...s,
    ok: s.offGt1 === 0 && s.off1 <= 0.001 * want.length,
  };
}

export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const { man, f32 } = await loadCase(base);
    const isInt = !!man.int_mode;
    const H = man.H;
    const lines = man.lines;

    const xB = gpu.upload(await f32("in"));
    const padB = gpu.upload(await f32("pads"));
    const wB = gpu.upload(await f32("w_conv"));
    const tB = gpu.upload(await f32("tanh"));
    const sB = isInt ? gpu.upload(await f32("s_conv")) : null;

    const wantConv = await f32("conv");
    const wantPool = await f32("pool");
    const convB = gpu.buffer(wantConv.byteLength);
    const poolB = gpu.buffer(wantPool.byteLength);

    recordConv(gpu, { x: xB, pads: padB, w: wB, s: sB, tanhT: tB, out: convB, lines, H, intMode: isInt });
    recordPool(gpu, { conv: convB, out: poolB, lines, H });

    const gotConv = await gpu.read(convB, Float32Array, wantConv.byteLength);
    const gotPool = await gpu.read(poolB, Float32Array, wantPool.byteLength);
    res.push(stage(man.case, "conv", gotConv, wantConv, isInt));
    res.push(stage(man.case, "pool", gotPool, wantPool, isInt));
  }
  return { ok: res.every((r) => r.ok), res };
}
