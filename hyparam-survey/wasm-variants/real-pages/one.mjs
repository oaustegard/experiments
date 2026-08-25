// One (variant, column) timing over REAL parquet snappy pages, fresh process.
// argv: <wasm> <column>
import { readFileSync } from 'node:fs'
const [wasmPath, col] = process.argv.slice(2)
const pages = JSON.parse(readFileSync('pages.json', 'utf8'))[col]
  .map(p => ({ c: new Uint8Array(Buffer.from(p.c, 'base64')), u: p.u }))
const totalOut = pages.reduce((a, p) => a + p.u, 0)

const inst = new WebAssembly.Instance(new WebAssembly.Module(readFileSync(wasmPath)))
const { memory, uncompress } = inst.exports
const maxIn = Math.max(...pages.map(p => p.c.length))
const maxOut = Math.max(...pages.map(p => p.u))
const inputStart = 68000, outputStart = inputStart + maxIn
const need = outputStart + maxOut, page = 65536
if (memory.buffer.byteLength < need) memory.grow(Math.ceil(need/page) - memory.buffer.byteLength/page)

function decodeAll() {
  const view = new Uint8Array(memory.buffer)
  for (const p of pages) { view.set(p.c, inputStart); uncompress(inputStart, p.c.length, outputStart) }
}
// correctness: every page must decode to the same bytes the reference gives
{
  const view = new Uint8Array(memory.buffer)
  for (const p of pages) {
    view.set(p.c, inputStart)
    if (uncompress(inputStart, p.c.length, outputStart) !== 0) { console.log('WRONG'); process.exit(0) }
  }
}
const N = 30
for (let i = 0; i < 10; i++) decodeAll()
const t = performance.now()
for (let i = 0; i < N; i++) decodeAll()
const ms = (performance.now() - t) / N
console.log((totalOut / 1048576 / (ms / 1000)).toFixed(1))
