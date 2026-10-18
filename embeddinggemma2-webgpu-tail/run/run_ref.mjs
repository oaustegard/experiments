// Node driver for the CPU reference of the tail.
// Usage: node run/run_ref.mjs [FIXTURES_DIR] [OUT_DIR]
// Defaults: fixtures/ and out/ref/ under the working directory root.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { encodeRef, searchRef, topkRef, poolRef, D } from "../web/tail_ref.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const fx = resolve(process.argv[2] ?? join(root, "fixtures"));
const outDir = resolve(process.argv[3] ?? join(root, "out", "ref"));

function readBytes(name) {
  const buf = readFileSync(join(fx, name));
  // Copy into a fresh, 8-byte-aligned ArrayBuffer so typed-array views are valid.
  return new Uint8Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
const readF32 = (name) => new Float32Array(readBytes(name).buffer);
const readU32 = (name) => new Uint32Array(readBytes(name).buffer);

function write(name, typed) {
  writeFileSync(join(outDir, name), Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength));
}

const p = JSON.parse(readFileSync(join(fx, "params.json"), "utf8"));
const n = p.n_docs;
const nq = p.n_queries;
const k = p.k;
const d = p.d;
if (d !== D) throw new Error(`fixture d=${d} != ${D}`);

const params = {
  R: readF32("R.f32"),
  Rmu: readF32("Rmu.f32"),
  mu: readF32("mu.f32"),
  boundaries: p.boundaries,
  centroids: p.centroids,
  kappa: p.kappa,
};

const t0 = performance.now();
const X = readF32("docs_in.f32");
if (X.length !== n * d) throw new Error(`docs_in: ${X.length} values, expected ${n * d}`);
const enc = encodeRef(X, n, params, { normalize: true });
const t1 = performance.now();

const Q = readF32("queries_in.f32");
if (Q.length !== nq * d) throw new Error(`queries_in: ${Q.length} values, expected ${nq * d}`);
const scores = searchRef(Q, nq, enc, params, { normalize: true });
const top = topkRef(scores, nq, n, k);
const t2 = performance.now();

const pooled = poolRef(readF32("hidden.f32"), readU32("mask.u32"), p.pool_b, p.pool_s, d);
const t3 = performance.now();

mkdirSync(outDir, { recursive: true });
write("packed.u8", enc.packed);
write("scale.f32", enc.scale);
write("bias.f32", enc.bias);
write("len.f32", enc.len);
write("mdot.f32", enc.mdot);
write("scores.f32", scores);
write("topk.i32", top);
write("pooled.f32", pooled);

const ms = (a, b) => (b - a).toFixed(0);
console.log(
  `encode ${ms(t0, t1)} ms, search+topk ${ms(t1, t2)} ms, pool ${ms(t2, t3)} ms; wrote ${outDir}`,
);
