/**
 * One cold decompress timing, in a fresh process.
 *
 * Mirrors the shape of hysnappy's own benchmark.js: instantiate, compress
 * once, then time 1000 decompress iterations with no warm-up. Prints a single
 * MB/s figure. hysnappy_bench.mjs spawns this repeatedly to measure how far a
 * cold, unwarmed timing loop moves between runs; measuring that in-process
 * does not work, because by then V8 has already optimized the same code paths
 * and a fresh `snappyUncompressor()` closure does not undo that.
 */
import { snappyUncompressor, snappyCompressor } from 'hysnappy'

const SENTENCES = [
  'The function processes the input data and returns a transformed result that can be used by downstream components in the pipeline for further analysis and visualization.',
  'First, we need to validate the parameters before proceeding with the operation to ensure that all required fields are present and conform to the expected schema definitions.',
  'This approach improves performance by caching intermediate computations in a hash table, allowing subsequent requests with similar parameters to bypass expensive recalculations entirely.',
  'The algorithm iterates through each element and applies the transformation using a map-reduce pattern that enables efficient parallel processing across multiple CPU cores when available.',
]
const completions = []
for (let i = 0; i < 50; i++) {
  const content = []
  for (let j = 0; j < 20; j++) content.push(SENTENCES[(i + j) % SENTENCES.length])
  completions.push({ id: `chatcmpl-${i}`, choices: [{ message: { content: content.join(' ') } }] })
}
const input = new TextEncoder().encode(JSON.stringify(completions))

const warmIters = Number(process.argv[2] ?? 0)
const uncompress = snappyUncompressor()
const packed = snappyCompressor()(input)

for (let i = 0; i < warmIters; i++) uncompress(packed, input.length)

const iters = 1000
const t0 = performance.now()
for (let i = 0; i < iters; i++) uncompress(packed, input.length)
const ms = performance.now() - t0
console.log((input.length * iters / 1048576 / (ms / 1000)).toFixed(0))
