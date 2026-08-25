import { parquetMetadataAsync, asyncBufferFromFile, parquetRead } from 'hyparquet'
import { snappyUncompressor } from 'hysnappy'
import { readFileSync, writeFileSync } from 'node:fs'
const ref = snappyUncompressor()
const cols = JSON.parse(readFileSync('cols.json', 'utf8'))
const dump = {}
for (const col of Object.keys(cols)) {
  const file = await asyncBufferFromFile(`snappy_${col}.parquet`)
  const md = await parquetMetadataAsync(file)
  const pages = []
  await parquetRead({ file, metadata: md, columns: [col], onChunk: () => {},
    compressors: { SNAPPY: (input, outputLength) => {
      pages.push({ c: Buffer.from(input).toString('base64'), u: outputLength })
      return ref(input, outputLength) } } })
  dump[col] = pages
  console.log(col, pages.length, 'pages,',
    pages.reduce((a,p)=>a+p.u,0), 'uncompressed bytes')
}
writeFileSync('pages.json', JSON.stringify(dump))
