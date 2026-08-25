// One (variant, corpus) timing in a fresh process. argv: <wasm> <corpus>
import { readFileSync } from 'node:fs'
import { snappyCompressor } from '../node_modules/hysnappy/js/index.js'

const [wasmPath, corpus] = process.argv.slice(2)
const enc = new TextEncoder()
let x = 7
const rnd = n => { const o = new Uint8Array(n); for (let i = 0; i < n; i++) { x = (Math.imul(x,1664525)+1013904223)>>>0; o[i]=x>>>24 } return o }
function build(name) {
  if (name === 'literal') return rnd(4 * 1024 * 1024)
  if (name === 'match') return enc.encode('abcdefghijklmnopqrstuvwxyz'.repeat(160000))
  const one = enc.encode(JSON.stringify(Array.from({length:400},(_,i)=>({
    id:`usr_${i.toString(36)}`,name:`User ${i}`,email:`user${i}@example.com`,
    role:['admin','member','viewer'][i%3],active:i%7!==0}))))
  const big = new Uint8Array(4*1024*1024)
  for (let i=0;i<big.length;i++) big[i]=one[i%one.length]
  return big
}
const data = build(corpus)
const packed = snappyCompressor()(data)

const inst = new WebAssembly.Instance(new WebAssembly.Module(readFileSync(wasmPath)))
const { memory, uncompress } = inst.exports
const inputStart = 68000
const outputStart = inputStart + packed.byteLength
const total = inputStart + packed.byteLength + data.length
const page = 65536
if (memory.buffer.byteLength < total) memory.grow(Math.ceil(total/page) - memory.buffer.byteLength/page)
new Uint8Array(memory.buffer).set(packed, inputStart)

// correctness once, outside the timing loop
{
  const r = uncompress(inputStart, packed.byteLength, outputStart)
  const view = new Uint8Array(memory.buffer)
  let ok = r === 0
  if (ok) for (let i = 0; i < data.length; i++) if (view[outputStart+i] !== data[i]) { ok = false; break }
  if (!ok) { console.log('WRONG'); process.exit(0) }
}

// time the wasm call only: no slice(), no allocation, nothing to GC
const N = 40
for (let i = 0; i < 15; i++) uncompress(inputStart, packed.byteLength, outputStart)
const t = performance.now()
for (let i = 0; i < N; i++) uncompress(inputStart, packed.byteLength, outputStart)
const ms = (performance.now() - t) / N
console.log((data.length / 1048576 / (ms / 1000)).toFixed(0))
