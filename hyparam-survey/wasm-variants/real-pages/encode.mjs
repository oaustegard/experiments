import { parquetReadObjects, parquetMetadataAsync, asyncBufferFromFile, parquetRead } from 'hyparquet'
import { compressors } from 'hyparquet-compressors'
import { parquetWriteBuffer } from 'hyparquet-writer'
import { snappyCompressor, snappyUncompressor } from 'hysnappy'
import { writeFileSync, existsSync } from 'node:fs'
import { execSync } from 'node:child_process'

const NYC = 'nyc.parquet'
const NYC_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
if (!existsSync(NYC)) {
  console.log('fetching', NYC_URL)
  execSync(`curl -sL --max-time 600 "${NYC_URL}" -o ${NYC}`, { stdio: 'inherit' })
}

const file = await asyncBufferFromFile(NYC)
const md = await parquetMetadataAsync(file)
const COLS = ['trip_distance', 'total_amount', 'tpep_pickup_datetime', 'PULocationID', 'store_and_fwd_flag']
const ROWS = 400000

const data = {}
for (const col of COLS) {
  const vals = []
  await parquetRead({ file, metadata: md, compressors, columns: [col], rowStart: 0, rowEnd: ROWS,
    onChunk: ({ columnName, columnData }) => { if (columnName === col) for (const v of columnData) vals.push(v) } })
  data[col] = vals.slice(0, ROWS)
  console.log(col.padEnd(24), vals.length, 'sample', JSON.stringify(vals.slice(0, 3)))
}

// Re-encode each column on its own as SNAPPY. Real values, snappy codec.
const pages = {}
for (const col of COLS) {
  const vals = data[col]
  const TYPES = { trip_distance: 'DOUBLE', total_amount: 'DOUBLE',
                  tpep_pickup_datetime: 'INT64', PULocationID: 'INT32',
                  store_and_fwd_flag: 'STRING' }
  const type = TYPES[col]
  // hyparquet parses INT64 timestamps into Date; write them back as epoch millis
  const written = type === 'INT64' ? vals.map(v => BigInt(v instanceof Date ? v.getTime() : v)) : vals
  const buf = parquetWriteBuffer({
    columnData: [{ name: col, data: written, type }],
    compressed: true, statistics: false,
    compressors: { SNAPPY: snappyCompressor() },
  })
  writeFileSync(`snappy_${col}.parquet`, Buffer.from(buf))
  pages[col] = { type, bytes: buf.byteLength }
  console.log(col.padEnd(24), type.padEnd(8), (buf.byteLength / 1048576).toFixed(2), 'MB')
}
writeFileSync('cols.json', JSON.stringify(pages, null, 2))
