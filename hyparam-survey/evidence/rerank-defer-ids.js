// src/search/rerank.js:126-129
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Phase 3 id deferral. README: 'Fetch the id column for the top-K winners only'.

  // Phase 3: fetch ids for just the top-K winners.
  const ids = await fetchIds(file, metadata, winners.map(w => w.rowIndex), compressors)
  return winners.map((w, i) => ({ id: ids[i], score: w.score, rowIndex: w.rowIndex }))
}
