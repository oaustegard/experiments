// src/cluster.js:15-62
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Binary k-means over the 1-bit codes: Hamming assignment, bit-majority-vote centroid update. README: 'runs k-means on the 1-bit codes'.

/**
 * @param {Uint8Array[]} codes        per-row binary codes (length n, each binaryBytes long)
 * @param {number} binaryBytes        bytes per code (= ceil(dim/8))
 * @param {number} k                  number of clusters
 * @param {number} [iterations=6]     number of k-means iterations
 * @param {number} [seed=1]           RNG seed (deterministic init)
 * @returns {{ assignments: Int32Array, centroids: Uint8Array[] }}
 */
export function binaryKMeans(codes, binaryBytes, k, iterations = 6, seed = 1) {
  const n = codes.length
  if (n === 0) return { assignments: new Int32Array(0), centroids: [] }
  if (binaryBytes % 4 !== 0) {
    // The SWAR popcount loop iterates u32 words; tail bytes would be silently
    // skipped. Dimensions like 384 → 48 bytes are fine; arbitrary dims aren't.
    throw new Error(`binaryKMeans requires binaryBytes to be a multiple of 4, got ${binaryBytes}`)
  }
  const effectiveK = Math.min(k, n)
  const wordsPerRow = binaryBytes >> 2

  // Aligned U32 views over a flat backing buffer (one contiguous copy).
  const flat = new Uint8Array(n * binaryBytes)
  for (let i = 0; i < n; i += 1) flat.set(codes[i], i * binaryBytes)
  const flatU32 = wordsPerRow > 0
    ? new Uint32Array(flat.buffer, 0, n * wordsPerRow)
    : new Uint32Array(0)

  // Random init: pick k distinct row indices as initial centroids.
  let rngState = seed >>> 0 || 1
  /** @returns {number} pseudo-random uint32 */
  function rng() {
    const stepped = Math.imul(rngState, 1664525) + 1013904223
    rngState = stepped >>> 0
    return rngState
  }
  const initIdx = pickDistinct(n, effectiveK, rng)
  /** @type {Uint8Array[]} */
  const centroids = initIdx.map(i => flat.slice(i * binaryBytes, (i + 1) * binaryBytes))
  /** @type {Uint32Array[]} */
  let centroidU32 = centroids.map(c => bytesToU32(c, wordsPerRow))

  const assignments = new Int32Array(n)

  for (let iter = 0; iter < iterations; iter += 1) {
    // Assign each row to nearest centroid.
    for (let i = 0; i < n; i += 1) {
      let best = 0
      let bestDist = Infinity
      const rowOff = i * wordsPerRow
