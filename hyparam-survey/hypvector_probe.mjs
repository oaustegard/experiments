import { fileWriter } from 'hyparquet-writer'
import { writeVectors, searchVectors } from 'hypvector'
import { statSync } from 'node:fs'

const N = 50000, DIM = 384, K = 10
// Deterministic clustered synthetic data: 64 true centers + gaussian noise
function lcg(s){let x=s>>>0;return()=>{x=(Math.imul(x,1664525)+1013904223)>>>0;return x/4294967296}}
const r = lcg(42)
function gauss(){let u=0,v=0;while(u===0)u=r();while(v===0)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const centers = Array.from({length:64},()=>Float32Array.from({length:DIM},gauss))
function norm(v){let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s);const o=new Float32Array(v.length);for(let i=0;i<v.length;i++)o[i]=v[i]/s;return o}
const vecs=[]
for(let i=0;i<N;i++){const c=centers[i%64];const v=new Float32Array(DIM)
  for(let d=0;d<DIM;d++)v[d]=c[d]+0.9*gauss()
  vecs.push({id:`row-${i}`,vector:norm(v)})}

let t=Date.now()
await writeVectors({writer:fileWriter('./v.parquet'),dimension:DIM,vectors:vecs})
const writeMs=Date.now()-t
const bytes=statSync('./v.parquet').size
console.log(`write ${writeMs} ms   file ${(bytes/1e6).toFixed(1)} MB   raw fp32 ${(N*DIM*4/1e6).toFixed(1)} MB`)

// queries = perturbed corpus vectors
const queries = Array.from({length:20},(_,q)=>{const base=vecs[(q*997)%N].vector
  const v=new Float32Array(DIM);for(let d=0;d<DIM;d++)v[d]=base[d]+0.15*gauss();return norm(v)})

async function run(label, opts){
  const t0=Date.now(); const all=[]
  for(const q of queries) all.push(await searchVectors({source:'./v.parquet',query:q,topK:K,...opts}))
  return {label, ms:(Date.now()-t0)/queries.length, all}
}
const exact = await run('exact', {rerankFactor:0})
const auto  = await run('auto (binary+cluster+rerank)', {})
const rf17  = await run('rerankFactor=17 (their rule N/3000)', {rerankFactor:17})
const wide  = await run('rerankFactor=50', {rerankFactor:50})
const rf100 = await run('rerankFactor=100', {rerankFactor:100})
const p1    = await run('probe=1.0, rerankFactor=10', {probe:1.0})
const both  = await run('probe=1.0, rerankFactor=50', {probe:1.0, rerankFactor:50})

function recall(a,b){let hit=0,tot=0
  for(let i=0;i<a.all.length;i++){const truth=new Set(a.all[i].map(r=>r.id));tot+=truth.size
    for(const r of b.all[i]) if(truth.has(r.id)) hit++}
  return hit/tot}
for(const r of [exact,auto,rf17,wide,rf100,p1,both])
  console.log(`${r.label.padEnd(30)} ${r.ms.toFixed(1).padStart(7)} ms/q   recall@10 ${(recall(exact,r)*100).toFixed(1)}%`)
