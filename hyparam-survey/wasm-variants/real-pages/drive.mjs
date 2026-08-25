import { spawnSync } from 'node:child_process'
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
const variants = readdirSync('../build').filter(f => f.endsWith('.wasm')).sort()
const cols = Object.keys(JSON.parse(readFileSync('pages.json','utf8')))
const TRIALS = 9
const med = a => [...a].sort((p,q)=>p-q)[Math.floor(a.length/2)]
// bootstrap CI on the ratio of medians
function bootCI(a, b, iters = 4000) {
  const rs = []
  for (let k = 0; k < iters; k++) {
    const ra = Array.from({length:a.length}, () => a[(Math.random()*a.length)|0])
    const rb = Array.from({length:b.length}, () => b[(Math.random()*b.length)|0])
    rs.push(med(ra) / med(rb))
  }
  rs.sort((x,y)=>x-y)
  return [ +rs[Math.floor(iters*0.025)].toFixed(3), +rs[Math.floor(iters*0.975)].toFixed(3) ]
}
const mix = JSON.parse(readFileSync('mix.json', 'utf8'))
const out = {
  generated_by: 'drive.mjs',
  source: 'NYC TLC yellow_tripdata_2024-01.parquet, five columns re-encoded as SNAPPY',
  trials: TRIALS,
  ci: '95% bootstrap on the ratio of medians, 4000 resamples',
  mix,
  results: {},
}
for (const col of cols) {
  out.results[col] = {}
  for (const v of variants) {
    const runs = []
    for (let t = 0; t < TRIALS; t++) {
      const r = spawnSync(process.execPath, ['one.mjs', `../build/${v}`, col], {encoding:'utf8'})
      const s = r.stdout.trim()
      if (s === 'WRONG') { out.results[col][v] = 'WRONG'; break }
      if (!s) { console.error(r.stderr.slice(0,300)); process.exit(1) }
      runs.push(Number(s))
    }
    if (out.results[col][v] !== 'WRONG') out.results[col][v] = { median_mbps: +med(runs).toFixed(1), runs }
  }
  const base = out.results[col]['base.wasm']
  console.log(`\n${col}`)
  for (const [v, r] of Object.entries(out.results[col])) {
    if (r === 'WRONG') { console.log(`  ${v.padEnd(22)} WRONG`); continue }
    if (v !== 'base.wasm') { r.vs_base = +(r.median_mbps / base.median_mbps).toFixed(3); r.ci95 = bootCI(r.runs, base.runs) }
    const tag = v === 'base.wasm' ? '' :
      `${r.vs_base.toFixed(2)}x  CI [${r.ci95[0].toFixed(2)}, ${r.ci95[1].toFixed(2)}]` +
      (r.ci95[0] > 1 ? '  FASTER' : r.ci95[1] < 1 ? '  SLOWER' : '  no effect')
    console.log(`  ${v.padEnd(22)} ${String(r.median_mbps).padStart(8)} MB/s  ${tag}`)
  }
}
// Byte-weighted rollup over the file's real column mix: the number that says
// whether any of this is worth doing on an actual Parquet file.
const bytesOf = Object.fromEntries(Object.entries(mix).map(([c, m]) => [c, m.uncompressed_bytes]))
const totalBytes = Object.values(bytesOf).reduce((a, b) => a + b, 0)
const medOf = xs => med(xs)
const weightedMbps = pick => {
  const secs = Object.keys(bytesOf).reduce((a, c) => a + bytesOf[c] / (pick(c) * 1048576), 0)
  return totalBytes / 1048576 / secs
}
const pickRun = (c, v) => { const r = out.results[c][v].runs; return medOf(r.map(() => r[(Math.random()*r.length)|0])) }
out.weighted = { total_bytes: totalBytes, byte_share: Object.fromEntries(
  Object.entries(bytesOf).map(([c, b]) => [c, +(100*b/totalBytes).toFixed(1)])), variants: {} }
const baseW = weightedMbps(c => out.results[c]['base.wasm'].median_mbps)
out.weighted.base_mbps = +baseW.toFixed(1)
for (const v of variants) {
  const w = weightedMbps(c => out.results[c][v].median_mbps)
  const rs = []
  for (let k = 0; k < 4000; k++) {
    const tb = Object.keys(bytesOf).reduce((a,c)=>a+bytesOf[c]/(pickRun(c,'base.wasm')*1048576),0)
    const tv = Object.keys(bytesOf).reduce((a,c)=>a+bytesOf[c]/(pickRun(c,v)*1048576),0)
    rs.push(tb/tv)
  }
  rs.sort((a,b)=>a-b)
  const lo = +rs[100].toFixed(3), hi = +rs[3899].toFixed(3)
  out.weighted.variants[v] = { mbps: +w.toFixed(1), vs_base: +(w/baseW).toFixed(3), ci95: [lo, hi],
    verdict: lo > 1 ? 'faster' : hi < 1 ? 'slower' : 'no effect' }
}
console.log('\nbyte-weighted over the real column mix')
console.log(`  base.wasm             ${out.weighted.base_mbps} MB/s`)
for (const [v, r] of Object.entries(out.weighted.variants).sort((a,b)=>b[1].vs_base-a[1].vs_base))
  console.log(`  ${v.padEnd(22)}${String(r.mbps).padStart(8)} MB/s  ${r.vs_base.toFixed(3)}x  CI [${r.ci95[0]}, ${r.ci95[1]}]  ${r.verdict}`)
writeFileSync('results.json', JSON.stringify(out, null, 2))
