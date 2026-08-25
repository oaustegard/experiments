// src/search/ranges.js:8-59
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Phase-1 cluster selection: rank centroids by Hamming, take the top probe fraction, merge their row ranges. README: query path step 1.

/**
 * Pick exact contiguous row ranges based on cluster nearness to the query.
 * Uses `clusterCounts` KV metadata: since rows are sorted by cluster id,
 * cluster k occupies [cumsum[k], cumsum[k+1]). We pick the top-N nearest
 * clusters (by Hamming centroid distance), then merge their contiguous
 * row ranges so useOffsetIndex fetches only the pages that cover them.
 *
 * @param {HypVectorMetadata} meta
 * @param {Uint8Array} queryBin
 * @param {number | undefined} probe
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
export function selectClusterRowRanges(meta, queryBin, probe) {
  const centroids = meta.centroids ?? []
  const counts = meta.clusterCounts
  if (centroids.length === 0 || !counts) return [{ rowStart: 0, rowEnd: meta.count }]

  // Cumulative offsets so cluster k spans [offset[k], offset[k+1]).
  const offsets = new Uint32Array(centroids.length + 1)
  for (let c = 0; c < centroids.length; c += 1) offsets[c + 1] = offsets[c] + counts[c]

  // Rank clusters by Hamming to query.
  const clusterDist = new Array(centroids.length)
  for (let c = 0; c < centroids.length; c += 1) {
    clusterDist[c] = { cluster: c, hamming: hammingDistanceBytes(queryBin, centroids[c]) }
  }
  clusterDist.sort((a, b) => a.hamming - b.hamming)

  const probeFraction = probe === undefined ? defaultClusterProbeFraction : probe
  // probe in (0, 1] is a fraction of clusters (1.0 = all clusters);
  // probe > 1 is an absolute count.
  let targetClusters = probeFraction > 1
    ? Math.min(Math.ceil(probeFraction), centroids.length)
    : Math.max(1, Math.ceil(centroids.length * probeFraction))
  // The default fraction over-probes at large nlist (recall knees well before
  // 0.25 x nlist), so cap the *default* to bound roundtrips/bytes at scale.
  // An explicit `probe` — fraction or count — is taken literally.
  if (probe === undefined) targetClusters = Math.min(targetClusters, defaultClusterProbeCap)

  const wanted = clusterDist.slice(0, targetClusters).map(c => c.cluster).sort((a, b) => a - b)
  /** @type {{ rowStart: number, rowEnd: number }[]} */
  const ranges = []
  for (const c of wanted) {
    ranges.push({ rowStart: offsets[c], rowEnd: offsets[c + 1] })
  }
  return mergeRanges(ranges)
}

/**
 * Merge adjacent/overlapping ranges.
 *
 * @param {{ rowStart: number, rowEnd: number }[]} ranges (already in order)
