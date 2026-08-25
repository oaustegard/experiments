import { parquetMetadataAsync, asyncBufferFromFile, parquetRead } from 'hyparquet'
import { readFileSync, writeFileSync } from 'node:fs'
import { snappyMix } from './mix.mjs'
import { snappyUncompressor } from 'hysnappy'
const snappyRef = snappyUncompressor()

// Pull the raw compressed page bytes straight out of each snappy parquet.
// hyparquet does not hand them over, so read the column chunk range and walk
// page headers is overkill; instead re-derive from the file by compressing the
// same plain pages is circular. Simplest honest route: hyparquet exposes the
// compressed bytes if we intercept the decompressor.
const cols = JSON.parse(readFileSync('cols.json', 'utf8'))
const out = {}
for (const col of Object.keys(cols)) {
  const path = `snappy_${col}.parquet`
  const file = await asyncBufferFromFile(path)
  const md = await parquetMetadataAsync(file)
  const seen = []
  const spy = {
    SNAPPY: (input, outputLength) => {
      seen.push(new Uint8Array(input))
      // real decode so the read completes
      return snappyRef(input, outputLength)
    },
  }
  await parquetRead({ file, metadata: md, compressors: spy, columns: [col],
    onChunk: () => {} })
  const mixes = seen.map(snappyMix)
  const tot = mixes.reduce((a, m) => ({
    literal: a.literal + m.literal, copy: a.copy + m.copy,
    nLit: a.nLit + m.nLit, nCopy: a.nCopy + m.nCopy,
    uncompressed: a.uncompressed + m.uncompressed }), {literal:0,copy:0,nLit:0,nCopy:0,uncompressed:0})
  out[col] = {
    type: cols[col].type, pages: seen.length,
    compressed_bytes: seen.reduce((a,b)=>a+b.length,0),
    uncompressed_bytes: tot.uncompressed,
    literal_bytes: tot.literal, copy_bytes: tot.copy,
    literal_pct: +(100*tot.literal/(tot.literal+tot.copy)).toFixed(1),
    mean_literal_run: +(tot.literal/Math.max(tot.nLit,1)).toFixed(1),
    mean_copy_len: +(tot.copy/Math.max(tot.nCopy,1)).toFixed(1),
  }
  console.log(col.padEnd(24), cols[col].type.padEnd(7), String(seen.length).padStart(3), 'pages',
    'literal', String(out[col].literal_pct).padStart(5) + '%',
    'mean lit run', String(out[col].mean_literal_run).padStart(6),
    'mean copy len', out[col].mean_copy_len)
}
writeFileSync('mix.json', JSON.stringify(out, null, 2))
