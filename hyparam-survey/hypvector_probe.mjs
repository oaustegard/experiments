/**
 * hypvector recall/latency sweep.
 *
 * Writes a synthetic clustered corpus to a Parquet file via hypvector's own
 * writeVectors, then measures recall@10 of each approximate configuration
 * against hypvector's own exact full scan. Scoring against its own exact scan
 * isolates the approximation and removes any question about whether the
 * synthetic corpus has a well-defined ground truth.
 *
 * Deterministic: one seeded LCG drives centers, noise and queries, and
 * hypvector's k-means takes seed 1 by default. Two runs give identical recall
 * and an identical v.parquet sha256; only the ms columns move.
 *
 * Writes results.json (committed) and prints a table.
 *   npm install && node hypvector_probe.mjs
 */
import { fileWriter } from 'hyparquet-writer'
import { writeVectors, searchVectors } from 'hypvector'
import { statSync, writeFileSync, readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'

const N = 50000
const DIM = 384
const K = 10
const CENTERS = 64
const NOISE = 0.9
const N_QUERIES = 20
const QUERY_NOISE = 0.15
const QUERY_STRIDE = 997
const SEED = 42
const PARQUET = './v.parquet'

// ── deterministic RNG ────────────────────────────────────────────────────────
function lcg(s) {
  let x = s >>> 0
  return () => { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; return x / 4294967296 }
}
const rand = lcg(SEED)
function gauss() {
  let u = 0, v = 0
  while (u === 0) u = rand()
  while (v === 0) v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
function norm(v) {
  let s = 0
  for (const x of v) s += x * x
  s = Math.sqrt(s)
  const o = new Float32Array(v.length)
  for (let i = 0; i < v.length; i++) o[i] = v[i] / s
  return o
}

// ── corpus ───────────────────────────────────────────────────────────────────
const centers = Array.from({ length: CENTERS }, () => Float32Array.from({ length: DIM }, gauss))
const vecs = []
for (let i = 0; i < N; i++) {
  const c = centers[i % CENTERS]
  const v = new Float32Array(DIM)
  for (let d = 0; d < DIM; d++) v[d] = c[d] + NOISE * gauss()
  vecs.push({ id: `row-${i}`, vector: norm(v) })
}

const t0Write = Date.now()
await writeVectors({ writer: fileWriter(PARQUET), dimension: DIM, vectors: vecs })
const writeMs = Date.now() - t0Write
const bytes = statSync(PARQUET).size
const rawBytes = N * DIM * 4
console.log(`write ${writeMs} ms   file ${(bytes / 1e6).toFixed(1)} MB   raw fp32 ${(rawBytes / 1e6).toFixed(1)} MB   ${(100 * bytes / rawBytes).toFixed(1)}% of raw`)

// ── queries: perturbed corpus vectors ────────────────────────────────────────
const queries = Array.from({ length: N_QUERIES }, (_, q) => {
  const base = vecs[q * QUERY_STRIDE % N].vector
  const v = new Float32Array(DIM)
  for (let d = 0; d < DIM; d++) v[d] = base[d] + QUERY_NOISE * gauss()
  return norm(v)
})

// ── sweep ────────────────────────────────────────────────────────────────────
async function run(label, options) {
  const t0 = Date.now()
  const all = []
  for (const q of queries) all.push(await searchVectors({ source: PARQUET, query: q, topK: K, ...options }))
  return { label, options, ms: (Date.now() - t0) / queries.length, all }
}

const CONFIGS = [
  ['exact full scan', { rerankFactor: 0 }],
  ['default (rerankFactor 10, probe 0.25)', {}],
  ['rerankFactor 17 (their N/3000 rule)', { rerankFactor: 17 }],
  ['rerankFactor 50', { rerankFactor: 50 }],
  ['rerankFactor 100', { rerankFactor: 100 }],
  ['probe 1.0, rerankFactor 10', { probe: 1 }],
  ['probe 1.0, rerankFactor 50', { probe: 1, rerankFactor: 50 }],
]

const runs = []
for (const [label, options] of CONFIGS) runs.push(await run(label, options))
const exact = runs[0]

/** recall@K of `b` against the exact scan `a`, pooled over all queries */
function recall(a, b) {
  let hit = 0, total = 0
  for (let i = 0; i < a.all.length; i++) {
    const truth = new Set(a.all[i].map(x => x.id))
    total += truth.size
    for (const x of b.all[i]) if (truth.has(x.id)) hit++
  }
  return hit / total
}

for (const x of runs) {
  console.log(`${x.label.padEnd(38)} ${x.ms.toFixed(1).padStart(7)} ms/q   recall@${K} ${(recall(exact, x) * 100).toFixed(1)}%`)
}

// ── provenance + results.json ────────────────────────────────────────────────
const lock = JSON.parse(readFileSync('./package-lock.json', 'utf8'))
const locked = name => lock.packages?.[`node_modules/${name}`]?.version ?? null
const parquetSha = createHash('sha256').update(readFileSync(PARQUET)).digest('hex')

writeFileSync('./results.json', JSON.stringify({
  generated_by: 'hypvector_probe.mjs',
  corpus: { n: N, dim: DIM, centers: CENTERS, noise_sigma: NOISE, seed: SEED, l2_normalized: true },
  queries: { count: N_QUERIES, source: 'perturbed corpus vectors', noise_sigma: QUERY_NOISE, stride: QUERY_STRIDE },
  top_k: K,
  file: {
    path: PARQUET,
    bytes,
    raw_fp32_bytes: rawBytes,
    pct_of_raw: Number((100 * bytes / rawBytes).toFixed(1)),
    sha256: parquetSha,
    write_ms: writeMs,
  },
  env: {
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    hypvector: locked('hypvector'),
    hyparquet: locked('hyparquet'),
    'hyparquet-writer': locked('hyparquet-writer'),
  },
  runs: runs.map(x => ({
    label: x.label,
    options: x.options,
    ms_per_query: Number(x.ms.toFixed(1)),
    recall_at_k: Number(recall(exact, x).toFixed(3)),
  })),
}, null, 2) + '\n')

console.log(`\nwrote results.json   v.parquet sha256 ${parquetSha}`)
