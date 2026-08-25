// src/cluster.js:161-203
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Greedy nearest-neighbour renumbering of cluster ids so adjacent ids are close in Hamming space. README: 'Greedy Hamming renumbering'.

/**
 * Renumber clusters so that adjacent ids correspond to centroids that are
 * close in Hamming space. Uses a greedy nearest-neighbor walk: pick the
 * centroid closest to the all-zero pivot as id 0, then repeatedly jump to
 * the unvisited centroid nearest the current one.
 *
 * This makes the top-N nearest clusters to ANY query more likely to land
 * in a contiguous id range, which lets mergeRanges coalesce phase-1 scan
 * ranges into fewer parquetRead calls.
 *
 * @param {Uint8Array[]} centroids
 * @returns {Int32Array}  permutation: newOrder[oldId] = newId
 */
export function reorderClustersByHamming(centroids) {
  const k = centroids.length
  const newOrder = new Int32Array(k)
  if (k === 0) return newOrder
  const visited = new Uint8Array(k)
  const zero = new Uint8Array(centroids[0].byteLength)

  // Start with centroid closest to the all-zero pivot (stable, deterministic).
  let cur = 0
  let bestD = Infinity
  for (let c = 0; c < k; c += 1) {
    const d = hammingDistanceBytes(centroids[c], zero)
    if (d < bestD) { bestD = d; cur = c }
  }
  newOrder[cur] = 0
  visited[cur] = 1
  for (let step = 1; step < k; step += 1) {
    let next = -1
    let nd = Infinity
    for (let c = 0; c < k; c += 1) {
      if (visited[c]) continue
      const d = hammingDistanceBytes(centroids[cur], centroids[c])
      if (d < nd) { nd = d; next = c }
    }
    newOrder[next] = step
    visited[next] = 1
    cur = next
  }
  return newOrder
}
