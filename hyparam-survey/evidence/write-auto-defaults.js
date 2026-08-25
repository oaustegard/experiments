// src/writeVectors.js:101-120
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Auto-binary threshold, the expectedHamming spread check, and clusterCount = round(sqrt(N)/2). README: write path.

  // Resolve auto defaults now that we know N. Auto-clusters only fires
  // when the caller also let `binary` auto; explicit `binary: true` means
  // "add the column, don't reshuffle rows". Degenerate sign-bit codes
  // (e.g. embeddings with all-non-negative components) are all ties, which
  // would make phase-1 ranking near-random, so auto-binary also requires
  // the codes to spread; without the column, search stays an exact scan.
  if (autoBinary) {
    binary = ids.length >= defaultAutoBinaryThreshold &&
      expectedHamming(packedBin, dimension) >= dimension * defaultBinaryMinSpread
  }
  binary = binary === true
  const clusterCount = clusters ?? (autoBinary && binary ? Math.max(1, Math.round(Math.sqrt(ids.length) / 2)) : 0)
  // Clustering operates on the binary codes, so it implies the binary column
  // even when auto-binary would have left it off at small N (explicit
  // `clusters > 0` with a sub-threshold corpus).
  if (clusterCount > 0) binary = true

  const effectivePageSize = pageSize ?? (binary ? defaultBinaryPageSize : undefined)

  /** @type {Uint8Array[] | null} */
