// src/search/rerank.js:51-110
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Phases 1 and 2 in full: range scan, candidate set, coalesced runs, useOffsetIndex float32 fetch. README: query path steps 1-2.

  // Phase 1: Hamming scan over selected ranges of the binary column.
  // With a prefetched in-memory buffer, score the row ranges directly from
  // RAM. Otherwise, parquetRead each range — the binary column is small
  // (dim/8 bytes/row), so per-page seeking via useOffsetIndex costs an
  // extra RT without saving meaningful bytes; read whole column chunks
  // instead. (Phase 2's float32 column is ~32x larger per row, so it
  // keeps useOffsetIndex below.)
  if (binary) {
    for (const { rowStart, rowEnd } of scanRanges) {
      hammingScoreFlatRange(binary, rowStart, rowEnd, binaryBytes, queryBinU32, candidateHeap, candidatesK)
    }
  } else {
    await Promise.all(scanRanges.map(({ rowStart, rowEnd }) => parquetRead({
      file,
      metadata,
      compressors,
      columns: [defaultBinaryColumn],
      rowStart,
      rowEnd,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        if (columnName !== defaultBinaryColumn) return
        hammingScoreChunk(columnData, chunkStart, binaryBytes, queryBinU32, candidateHeap, candidatesK)
      },
    })))
  }

  if (candidateHeap.length === 0) return []

  const candidateRows = [...new Set(candidateHeap.map(c => c.rowIndex))].sort((a, b) => a - b)
  const wantedRows = new Set(candidateRows)
  const runs = coalesceRuns(candidateRows, 64)

  /** @type {{ rowIndex: number, score: number }[]} */
  const scored = []

  await Promise.all(runs.map(async ({ rowStart, rowEnd }) => {
    /** @type {Map<number, Float32Array>} */
    const local = new Map()
    await parquetRead({
      file,
      metadata,
      compressors,
      columns: [defaultVectorColumn],
      rowStart,
      rowEnd,
      useOffsetIndex: true,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        if (columnName !== defaultVectorColumn) return
        for (let i = 0; i < columnData.length; i += 1) {
          const rowIndex = chunkStart + i
          if (!wantedRows.has(rowIndex)) continue
          const bytes = columnData[i]
          /** @type {Float32Array} */
          let vector
          if (bytes.byteOffset % 4 === 0) {
            vector = new Float32Array(bytes.buffer, bytes.byteOffset, dim)
          } else {
            vector = new Float32Array(dim)
            new Uint8Array(vector.buffer).set(bytes)
          }
