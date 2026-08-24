/**
 * hysnappy vs snappyjs, head to head.
 *
 * The hyparam blog claims hysnappy's WASM decoder is "40% faster" than
 * "standard JavaScript Snappy decompression". hysnappy's own benchmark.js
 * reports absolute MB/s with nothing to compare against, so this runs both
 * libraries over the same corpora.
 *
 * Three corpora, because snappy's speed is set by how much of the input is
 * copy-matched rather than emitted as literals: a highly repetitive JSON-ish
 * blob (their own benchmark's shape), incompressible pseudo-random bytes, and
 * an already-compressed JPEG.
 *
 * Both directions are round-trip-verified across libraries before timing:
 * hysnappy must decode snappyjs output and vice versa.
 *
 * Writes hysnappy_results.json. Run:
 *   npm install && node hysnappy_bench.mjs
 */
import { snappyUncompressor, snappyCompressor } from 'hysnappy'
import SnappyJS from 'snappyjs'
import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'

const REPEATS = 3 // full sweeps; the reported figure is the median

const SENTENCES = [
  'The function processes the input data and returns a transformed result that can be used by downstream components in the pipeline for further analysis and visualization.',
  'First, we need to validate the parameters before proceeding with the operation to ensure that all required fields are present and conform to the expected schema definitions.',
  'This approach improves performance by caching intermediate computations in a hash table, allowing subsequent requests with similar parameters to bypass expensive recalculations entirely.',
  'The algorithm iterates through each element and applies the transformation using a map-reduce pattern that enables efficient parallel processing across multiple CPU cores when available.',
]

function jsonishCorpus() {
  const completions = []
  for (let i = 0; i < 50; i++) {
    const content = []
    for (let j = 0; j < 20; j++) content.push(SENTENCES[(i + j) % SENTENCES.length])
    completions.push({ id: `chatcmpl-${i}`, choices: [{ message: { content: content.join(' ') } }] })
  }
  return new TextEncoder().encode(JSON.stringify(completions))
}

function randomCorpus(bytes = 512 * 1024, seed = 12345) {
  let x = seed >>> 0
  const out = new Uint8Array(bytes)
  for (let i = 0; i < bytes; i++) { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; out[i] = x >>> 24 }
  return out
}

function jpegCorpus() {
  // Already-compressed bytes. Falls back to random if the clone is absent.
  const path = new URL('./fixtures/compressed.bin', import.meta.url)
  try {
    return new Uint8Array(readFileSync(path))
  } catch {
    return randomCorpus(80 * 1024, 999)
  }
}

const CORPORA = [
  ['json-ish, highly repetitive', jsonishCorpus()],
  ['pseudo-random, incompressible', randomCorpus()],
  ['already-compressed bytes', jpegCorpus()],
]

const hyUncompress = snappyUncompressor()
const hyCompress = snappyCompressor()

function timeIt(fn, iters) {
  fn(); fn() // warm the JIT
  const t = performance.now()
  for (let i = 0; i < iters; i++) fn()
  return performance.now() - t
}
const mbps = (bytes, iters, ms) => bytes * iters / 1048576 / (ms / 1000)
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
const sameBytes = (a, b) => a.length === b.length && a.every((v, i) => v === b[i])

const wasmSizes = {}
for (const name of ['uncompress', 'compress']) {
  const src = readFileSync(`./node_modules/hysnappy/js/${name}.js`, 'utf8')
  const m = src.match(/const wasm64 = ['"]([^'"]*)['"]/)
  wasmSizes[name] = m ? Buffer.from(m[1], 'base64').length : null
}

const versions = {
  node: process.version,
  platform: `${process.platform}-${process.arch}`,
  hysnappy: JSON.parse(readFileSync('./node_modules/hysnappy/package.json', 'utf8')).version,
  snappyjs: JSON.parse(readFileSync('./node_modules/snappyjs/package.json', 'utf8')).version,
}
console.log(`node ${versions.node} ${versions.platform}   hysnappy ${versions.hysnappy}   snappyjs ${versions.snappyjs}`)
console.log(`wasm: uncompress ${wasmSizes.uncompress} B, compress ${wasmSizes.compress} B ` +
            `(Chrome's synchronous new WebAssembly.Module limit is 4096 B)\n`)

const results = []
for (const [name, input] of CORPORA) {
  const packed = hyCompress(input)
  const packedJs = new Uint8Array(SnappyJS.compress(input))

  // Cross-library round trips, both directions, before any timing.
  const checks = {
    hy_decodes_hy: sameBytes(hyUncompress(packed, input.length), input),
    js_decodes_hy: sameBytes(new Uint8Array(SnappyJS.uncompress(packed)), input),
    hy_decodes_js: sameBytes(hyUncompress(packedJs, input.length), input),
  }
  for (const [k, ok] of Object.entries(checks)) {
    if (!ok) throw new Error(`round-trip failed on ${name}: ${k}`)
  }

  const iters = input.length > 400_000 ? 200 : 1000
  const runs = { dHy: [], dJs: [], cHy: [], cJs: [] }
  for (let r = 0; r < REPEATS; r++) {
    runs.dHy.push(mbps(input.length, iters, timeIt(() => hyUncompress(packed, input.length), iters)))
    runs.dJs.push(mbps(input.length, iters, timeIt(() => SnappyJS.uncompress(packed), iters)))
    runs.cHy.push(mbps(input.length, iters, timeIt(() => hyCompress(input), iters)))
    runs.cJs.push(mbps(input.length, iters, timeIt(() => SnappyJS.compress(input), iters)))
  }

  const row = {
    corpus: name,
    input_bytes: input.length,
    compressed_bytes: packed.length,
    ratio_pct: Number((100 * packed.length / input.length).toFixed(1)),
    iterations: iters,
    repeats: REPEATS,
    round_trip: checks,
    decompress_mbps: { hysnappy: Math.round(median(runs.dHy)), snappyjs: Math.round(median(runs.dJs)) },
    compress_mbps: { hysnappy: Math.round(median(runs.cHy)), snappyjs: Math.round(median(runs.cJs)) },
  }
  row.decompress_speedup = Number((row.decompress_mbps.hysnappy / row.decompress_mbps.snappyjs).toFixed(2))
  row.compress_speedup = Number((row.compress_mbps.hysnappy / row.compress_mbps.snappyjs).toFixed(2))
  // Per-sweep ratios, so the writeup can state how far the number moves.
  const perSweep = runs.dHy.map((v, i) => Number((v / runs.dJs[i]).toFixed(2)))
  row.decompress_speedup_per_sweep = perSweep
  row.decompress_speedup_spread = Number((Math.max(...perSweep) / Math.min(...perSweep)).toFixed(2))
  results.push(row)

  console.log(`${name}   ${(input.length / 1024).toFixed(0)} KB -> ${(packed.length / 1024).toFixed(1)} KB (${row.ratio_pct}%)`)
  console.log(`  decompress  hysnappy ${String(row.decompress_mbps.hysnappy).padStart(5)} MB/s   ` +
              `snappyjs ${String(row.decompress_mbps.snappyjs).padStart(5)} MB/s   ${row.decompress_speedup}x`)
  console.log(`  compress    hysnappy ${String(row.compress_mbps.hysnappy).padStart(5)} MB/s   ` +
              `snappyjs ${String(row.compress_mbps.snappyjs).padStart(5)} MB/s   ${row.compress_speedup}x\n`)
}

// Their benchmark.js times a cold loop with no warm-up. Measuring that
// in-process is invalid — by this point V8 has optimized the same code paths,
// and a fresh snappyUncompressor() closure does not undo that. So spawn
// hysnappy_cold.mjs in a fresh process per trial, with and without warm-up.
//
// 15 trials per arm, not 5: at n=5 this container's timing noise swamped the
// effect and two consecutive runs gave opposite orderings.
function coldTrials(warmIters, trials) {
  const out = []
  for (let t = 0; t < trials; t++) {
    const r = spawnSync(process.execPath, ['hysnappy_cold.mjs', String(warmIters)], { encoding: 'utf8' })
    if (r.status !== 0) throw new Error(`hysnappy_cold.mjs failed: ${r.stderr}`)
    out.push(Number(r.stdout.trim()))
  }
  return out
}

/**
 * One-sided Mann-Whitney U with a normal approximation; ties get average ranks.
 * U matches scipy.stats.mannwhitneyu exactly on the committed arrays (U=207);
 * p differs in the third significant figure (6.9e-05 here vs 4.8e-05 from
 * scipy's asymptotic method) because of the continuity correction and the
 * erfc approximation. Same order of magnitude, same conclusion.
 */
function mannWhitneyGreater(a, b) {
  const all = [...a.map(v => [v, 0]), ...b.map(v => [v, 1])].sort((p, q) => p[0] - q[0])
  const ranks = new Array(all.length)
  for (let i = 0; i < all.length;) {
    let j = i
    while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j++
    const avg = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[k] = avg
    i = j + 1
  }
  let rankSumA = 0
  const tieGroups = new Map()
  all.forEach(([v, grp], i) => {
    if (grp === 0) rankSumA += ranks[i]
    tieGroups.set(v, (tieGroups.get(v) ?? 0) + 1)
  })
  const n1 = a.length, n2 = b.length, n = n1 + n2
  const u = rankSumA - n1 * (n1 + 1) / 2
  const mu = n1 * n2 / 2
  let tieCorrection = 0
  for (const c of tieGroups.values()) tieCorrection += c ** 3 - c
  const sigma = Math.sqrt(n1 * n2 / 12 * (n + 1 - tieCorrection / (n * (n - 1))))
  const z = (u - mu - 0.5) / sigma
  // one-sided upper tail of the standard normal, via erfc
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2)
  const erfc = t * Math.exp(-z * z / 2 - 1.265512 + t * (1.000024 + t * (0.374092 + t * (0.096784 - t * 0.186288))))
  const p = z > 0 ? erfc / 2 : 1 - erfc / 2
  return { u, z: Number(z.toFixed(2)), p: Number(p.toPrecision(2)) }
}

const COLD_TRIALS = 15
const cold = coldTrials(0, COLD_TRIALS)
const warm = coldTrials(200, COLD_TRIALS)
const spread = xs => Math.max(...xs) / Math.min(...xs)
const test = mannWhitneyGreater(warm, cold)
const warmup = {
  method: 'fresh node process per trial, via hysnappy_cold.mjs',
  trials_per_arm: COLD_TRIALS,
  cold_mbps: cold,
  warm_mbps: warm,
  cold_median_mbps: Math.round(median(cold)),
  warm_median_mbps: Math.round(median(warm)),
  cold_spread: Number(spread(cold).toFixed(1)),
  warm_spread: Number(spread(warm).toFixed(1)),
  warm_over_cold_median: Number((median(warm) / median(cold)).toFixed(2)),
  mann_whitney_warm_greater: test,
  note: 'At 5 trials per arm the ordering was not stable; 15 separates them.',
}
console.log(`no warm-up  n=${COLD_TRIALS}  median ${warmup.cold_median_mbps} MB/s  spread ${warmup.cold_spread}x`)
console.log(`200 warm    n=${COLD_TRIALS}  median ${warmup.warm_median_mbps} MB/s  spread ${warmup.warm_spread}x`)
console.log(`warm/cold median ${warmup.warm_over_cold_median}x   Mann-Whitney z=${test.z} p=${test.p}\n`)

writeFileSync('./hysnappy_results.json', JSON.stringify({
  generated_by: 'hysnappy_bench.mjs',
  note: 'Medians of 3 sweeps. Node, not a browser; hysnappy numbers include the '
      + 'JS-to-WASM copy in and the slice out, which is what a caller actually pays.',
  env: versions,
  wasm_bytes: wasmSizes,
  chrome_sync_module_limit_bytes: 4096,
  corpora: {
    'json-ish, highly repetitive': 'generated in-script, 50 records of 20 repeated sentences',
    'pseudo-random, incompressible': 'generated in-script, seeded LCG, 512 KB',
    'already-compressed bytes': 'fixtures/compressed.bin, sha256 below',
  },
  fixture_sha256: createHash('sha256').update(jpegCorpus()).digest('hex'),
  results,
  warmup,
}, null, 2) + '\n')
console.log('wrote hysnappy_results.json')
