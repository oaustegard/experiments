const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path'), { execFileSync, spawn } = require('child_process');
// End-to-end check: gpu_bridge.py (session side) -> mockgh.js (GitHub REST over a local bare repo) -> the page on SwiftShader.
// Run from a scratch directory: node /path/to/gpu-bridge/run/e2e.js. CLI defaults to the claude-workspace checkout.
const S = process.cwd(), WEB = path.join(__dirname, '..', 'web'), CLI = process.env.GPU_BRIDGE_CLI || '/home/user/claude-workspace/scripts/gpu_bridge.py';
const MOCK = path.join(__dirname, 'mockgh.js');
const sh = (cmd, args, env) => execFileSync(cmd, args, { encoding: 'utf8', env: { ...process.env, ...env } });
(async () => {
  // bare repo seeded with a README
  fs.rmSync('bare.git', { recursive: true, force: true }); fs.rmSync('seed', { recursive: true, force: true }); fs.rmSync('clone', { recursive: true, force: true });
  sh('git', ['init', '-q', '--bare', '-b', 'main', 'bare.git']); sh('git', ['clone', '-q', 'bare.git', 'seed']);
  fs.writeFileSync('seed/README.md', 'comms\n'); sh('git', ['-C', 'seed', 'add', '.']); sh('git', ['-C', 'seed', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'init']); sh('git', ['-C', 'seed', 'push', '-q', 'origin', 'HEAD:main']);
  const api = spawn('node', [MOCK, path.join(S, 'bare.git'), process.env.MOCK_LATENCY_MS || '250'], { stdio: 'inherit' });
  const web = http.createServer((q, r) => { const f = path.join(WEB, new URL(q.url, 'http://x').pathname.replace(/\/$/, '/index.html')); let buf; try { buf = fs.readFileSync(f); } catch { r.writeHead(404); return r.end(); } r.writeHead(200, { 'content-type': f.endsWith('.js') ? 'text/javascript' : 'text/html' }); r.end(buf); }).listen(8765);
  const env = { GPU_BRIDGE_REMOTE: path.join(S, 'bare.git'), GPU_BRIDGE_REPO: path.join(S, 'clone') };
  // session side: submit a real WGSL job with an .npy input
  sh('python3', ['-c', 'import numpy as np; np.save("a.npy", np.arange(1,1025,dtype=np.float32))']);
  fs.writeFileSync('job.js', `const { a } = ctx.inputs;
const wgsl = \`@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x < arrayLength(&a)) { a[g.x] = a[g.x] * a[g.x]; } }\`;
const r = await ctx.compute({ wgsl, bindings: [a], dispatch: [Math.ceil(a.length / 64)], read: [0] });
ctx.log("n", a.length);
return { sq: r.out[0], ms: r.medianMs, meta: { k: ctx.inputs.k } };`);
  // page side
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'] });
  const p = await b.newPage({ viewport: { width: 390, height: 1000 } });
  const errs = []; p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto('http://localhost:8765/?api=http://localhost:8766'); await p.waitForTimeout(2000);
  await p.fill('#token', 'test-token'); await p.click('#connect');
  // a second tab on the same repo: each job must still run exactly once
  const p2 = await b.newPage(); p2.on('pageerror', (e) => errs.push('tab2: ' + e.message));
  await p2.goto('http://localhost:8765/?api=http://localhost:8766'); await p2.waitForTimeout(1500);
  await p2.fill('#token', 'test-token'); await p2.click('#connect');
  await p.evaluate(() => { const f = document.querySelector('#connForm'); f.requestSubmit(); f.requestSubmit(); });   // two connects 4 ms apart in one tab, as on the phone 2026-10-25
  await p.waitForTimeout(1500);   // both tabs listening before any job exists
  console.log('submit:', sh('python3', [CLI, 'submit', 'job.js', '--id', 'sq-1', '--note', 'square 1024', '--input', 'a=a.npy', '--input', 'k=@{"n":3}'], env).trim());
  fs.writeFileSync('bad.js', 'throw new Error("boom")');
  sh('python3', [CLI, 'submit', 'bad.js', '--id', 'bad-1'], env);
  for (const k of [2, 3, 4]) sh('python3', [CLI, 'submit', 'job.js', '--id', `sq-${k}`, '--input', 'a=a.npy', '--input', 'k=@{"n":3}'], env);   // enough jobs that each tab gets several
  // a job that reports progress: some tab must show it on the job's bar while it runs
  fs.writeFileSync('prog.js', 'for (let i = 1; i <= 8; i++) { ctx.progress(i / 8, `step ${i} of 8`); await new Promise((r) => setTimeout(r, 500)); } return { ok: 1 };');
  sh('python3', [CLI, 'submit', 'prog.js', '--id', 'prog-1'], env);
  let progSeen = null;
  for (const t1 = Date.now(); !progSeen && Date.now() - t1 < 60000; await new Promise((r) => setTimeout(r, 200))) {
    for (const pg of [p, p2]) {
      const txt = await pg.$$eval('[data-prog] span', (e) => e.map((x) => x.textContent)).catch(() => []);
      progSeen = progSeen || txt.find((x) => /step \d of 8 · \d+% · /.test(x)) || null;
    }
  }
  console.log('progress bar (want a step label):', progSeen);
  const t0 = Date.now(); let st;
  while (Date.now() - t0 < 120000) { try { st = sh('python3', [CLI, 'status', 'bad-1'], env); } catch (e) { st = e.stdout; if (e.status === 1 && ['sq-2', 'sq-3', 'sq-4'].every((j) => { try { sh('python3', [CLI, 'status', j], env); return true; } catch { return false; } })) break; } await new Promise((r) => setTimeout(r, 1500)); }
  console.log('connNote:', await p.textContent('#connNote'));
  let fetchOut; try { fetchOut = sh('python3', [CLI, 'fetch', 'sq-1', '--out', path.join(S, 'out')], env); } catch (e) { fetchOut = e.stdout + e.stderr; }
  const f = JSON.parse(fetchOut);
  const sq = sh('python3', ['-c', `import numpy as np; a=np.load("${f.outputs.sq.npy}"); print(bool((a==np.arange(1,1025,dtype=np.float32)**2).all()), a[:3].tolist())`]).trim();
  console.log('sq-1:', f.status, 'logs', f.logs, 'meta', JSON.stringify(f.outputs.meta), 'decoded ok:', sq);
  console.log('bad-1 status:', st.trim());
  console.log('wake comments:', JSON.stringify(await (await fetch('http://localhost:8766/__comments')).json()));
  console.log('bridge:', sh('python3', [CLI, 'bridge'], env).trim());
  const subjects = sh('git', ['--git-dir', 'bare.git', 'log', '--format=%s', 'main']).trim().split('\n');
  console.log('runs per job (want 1 each):', JSON.stringify({ sq: subjects.filter((x) => x === 'gpu-bridge: sq-1 done').length, bad: subjects.filter((x) => x === 'gpu-bridge: bad-1 error').length, claims: subjects.filter((x) => x.startsWith('gpu-bridge: claim')).length, sqAll: subjects.filter((x) => /^gpu-bridge: sq-\d done$/.test(x)).length }));
  console.log('tab statuses:', JSON.stringify(await p.$$eval('.job .st', (e) => e.map((x) => x.textContent))), JSON.stringify(await p2.$$eval('.job .st', (e) => e.map((x) => x.textContent))));
  // per tab, a job's claim must come after that tab's previous result commit (no overlapping jobs in one tab)
  const commits = sh('git', ['--git-dir', 'bare.git', 'log', '--reverse', '--format=%H %s', 'main']).trim().split('\n').map((l) => { const i = l.indexOf(' '); return [l.slice(0, i), l.slice(i + 1)]; });
  const open = {}; let overlaps = 0; const tabOf = {};
  for (const [sha, subj] of commits) {
    let m = /^gpu-bridge: claim (\S+)$/.exec(subj);
    if (m) { const tab = JSON.parse(sh('git', ['--git-dir', 'bare.git', 'show', `${sha}:results/${m[1]}/claim.json`])).tab; tabOf[m[1]] = tab; if (open[tab]) overlaps++; open[tab] = m[1]; continue; }
    m = /^gpu-bridge: (\S+) (done|error)$/.exec(subj);
    if (m && tabOf[m[1]] && open[tabOf[m[1]]] === m[1]) open[tabOf[m[1]]] = null;
  }
  console.log('overlapping jobs within a tab (want 0):', overlaps);
  console.log('repo log:', sh('git', ['--git-dir', 'bare.git', 'log', '--format=%s', 'main']).trim().split('\n').join(' | '));
  console.log('h-overflow:', await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), 'errors:', errs);
  await p.screenshot({ path: 'page.png', fullPage: true });
  await b.close(); web.close(); api.kill();
})().catch((e) => { console.error('E2E FAIL', e); process.exit(1); });
