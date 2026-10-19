// Cold-start and warm-latency bench. Each rep spawns a fresh server process and times, from spawn:
// the port accepting a request (listening), the first /embed answered, then a warm median over
// QUERIES. With --drop-caches it empties the page cache before each spawn (needs root), which
// stands in for a container whose image layers are not yet in memory.
// With --docker IMAGE it times `docker run` of the image instead, under --cpus/--memory limits
// matching a Cloudflare Containers instance type (basic = 0.25 vCPU / 1 GiB, standard-1 = 0.5 / 4 GiB,
// standard-2 = 1 / 6 GiB).
//   node bench_cold.mjs [--reps 5] [--drop-caches] [--threads N] [--docker IMAGE --cpus 0.5 --memory 4g]
import { spawn, execSync } from 'node:child_process';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i < 0 ? d : process.argv[i + 1]; };
const has = (k) => process.argv.includes(k);
const REPS = Number(arg('--reps', 5));
const PORT = 18080;
const QUERIES = [
  'how does the remex rotation work', 'cheap vector search in the browser', 'Which planet is known as the Red Planet?',
  'cloudflare worker rate limiting', 'matryoshka embeddings truncation quality', 'bluesky firehose sampling',
  'what is a hadamard transform', 'quantization aware training for embeddings', 'webgpu top-k selection', 'cold start latency',
];
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function post(text) {
  const r = await fetch(`http://127.0.0.1:${PORT}/embed`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`embed HTTP ${r.status}`);
  return r.json();
}

const runs = [];
for (let rep = 0; rep < REPS; rep++) {
  if (has('--drop-caches')) execSync('sync; echo 3 > /proc/sys/vm/drop_caches');
  const env = { ...process.env, PORT: String(PORT), ...(arg('--threads') ? { THREADS: arg('--threads') } : {}) };
  const t0 = performance.now();
  const DOCKER = arg('--docker');
  const name = `bench-cold-${process.pid}-${rep}`;          // killing `docker run` leaves the container up; stop it by name
  const child = DOCKER
    ? spawn('docker', ['run', '--rm', '--name', name, '-p', `${PORT}:8080`, '--cpus', arg('--cpus', '1'), '--memory', arg('--memory', '6g'),
        ...(arg('--threads') ? ['-e', `THREADS=${arg('--threads')}`] : []), DOCKER], { stdio: ['ignore', 'pipe', 'inherit'] })
    : spawn(process.execPath, ['server.cjs'], { env, stdio: ['ignore', 'pipe', 'inherit'] });
  const exited = new Promise((r) => child.on('exit', r));
  let serverTimings = null;
  child.stdout.on('data', (b) => { try { serverTimings = JSON.parse(b.toString()).timings_ms; } catch {} });
  let listening = null, died = null;
  child.on('exit', (code) => { died = code; });
  while (listening === null) {
    if (died !== null) throw new Error(`server exited with ${died} before listening`);
    // docker's port proxy accepts the connection before the server listens, so time out and retry.
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/health`, { signal: AbortSignal.timeout(500) });
      if (r.ok) listening = performance.now() - t0; else await sleep(10);
    } catch { await sleep(10); }
  }
  const first = await post(QUERIES[0]);
  const firstMs = performance.now() - t0;
  const warm = [];
  for (const q of QUERIES) { const t = performance.now(); await post(q); warm.push(performance.now() - t); }
  if (DOCKER) execSync(`docker rm -f ${name}`, { stdio: 'ignore' });
  else child.kill();
  await exited;
  runs.push({ listening_ms: Math.round(listening), first_embed_ms: Math.round(firstMs),
    first_inference_ms: first.ms, warm_roundtrip_median_ms: +median(warm).toFixed(1), server: serverTimings });
  console.error(JSON.stringify(runs.at(-1)));
}
console.log(JSON.stringify({
  reps: REPS, docker: arg('--docker') ?? null, cpus: arg('--cpus') ?? null, memory: arg('--memory') ?? null, drop_caches: has('--drop-caches'), threads: arg('--threads') ?? 'default',
  listening_ms: median(runs.map((r) => r.listening_ms)),
  first_embed_ms: median(runs.map((r) => r.first_embed_ms)),
  warm_roundtrip_median_ms: median(runs.map((r) => r.warm_roundtrip_median_ms)),
  runs,
}, null, 1));
