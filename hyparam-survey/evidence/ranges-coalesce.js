// src/search/ranges.js:76-101
// hypvector@0.2.2, sha256 5ac1eb31e4e81b19...
// why cited: Phase-2 run coalescing with maxGap. Called with 64 in rerank.js. README: 'merging gaps <= 64 rows'.

/**
 * Group a sorted list of row indices into contiguous runs, merging runs
 * whose gap is <= maxGap. Each run becomes one parquetRead call.
 *
 * @param {number[]} rows (sorted ascending)
 * @param {number} maxGap
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
export function coalesceRuns(rows, maxGap) {
  if (rows.length === 0) return []
  /** @type {{ rowStart: number, rowEnd: number }[]} */
  const runs = []
  let start = rows[0]
  let end = rows[0] + 1
  for (let i = 1; i < rows.length; i += 1) {
    if (rows[i] - end <= maxGap) {
      end = rows[i] + 1
    } else {
      runs.push({ rowStart: start, rowEnd: end })
      start = rows[i]
      end = rows[i] + 1
    }
  }
  runs.push({ rowStart: start, rowEnd: end })
  return runs
}
