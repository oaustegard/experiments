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
  const hits = {};   // /flaky/<name>/<n>: answers 403 to the first n requests per name, like jsDelivr right after a push
  const web = http.createServer((q, r) => { const fl = /^\/flaky\/([\w-]+)\/(\d+)/.exec(q.url); if (fl) { hits[fl[1]] = (hits[fl[1]] || 0) + 1; r.writeHead(hits[fl[1]] <= +fl[2] ? 403 : 200, { 'access-control-allow-origin': '*' }); return r.end('ok'); } const f = path.join(WEB, new URL(q.url, 'http://x').pathname.replace(/\/$/, '/index.html')); let buf; try { buf = fs.readFileSync(f); } catch { r.writeHead(404); return r.end(); } r.writeHead(200, { 'content-type': f.endsWith('.js') ? 'text/javascript' : 'text/html' }); r.end(buf); }).listen(8765);
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
  await p.goto('http://localhost:8765/?api=http://localhost:8766&retryWaits=1500,1500'); await p.waitForTimeout(2000);
  await p.fill('#token', 'test-token'); await p.click('#connect');
  // a second tab on the same repo: each job must still run exactly once
  const p2 = await b.newPage(); p2.on('pageerror', (e) => errs.push('tab2: ' + e.message));
  await p2.goto('http://localhost:8765/?api=http://localhost:8766&retryWaits=1500,1500'); await p2.waitForTimeout(1500);
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
  // transient failures: ctx.fetchBytes retries a 403 inside the job; a raw fetch that 403s once reruns the job;
  // a job that throws its own error, or keeps getting 403 past the retries, is not masked
  fs.writeFileSync('flaky.js', 'const b = await ctx.fetchBytes("http://localhost:8765/flaky/fb/2"); return { n: b.length };');
  sh('python3', [CLI, 'submit', 'flaky.js', '--id', 'flaky-fb'], env);
  fs.writeFileSync('flaky2.js', 'const r = await fetch("http://localhost:8765/flaky/raw/1"); if (!r.ok) throw new Error("x: HTTP " + r.status); return { ok: 1 };');
  sh('python3', [CLI, 'submit', 'flaky2.js', '--id', 'flaky-job'], env);
  fs.writeFileSync('flaky3.js', 'const r = await fetch("http://localhost:8765/flaky/dead/99"); if (!r.ok) throw new Error("x: HTTP " + r.status); return { ok: 1 };');
  sh('python3', [CLI, 'submit', 'flaky3.js', '--id', 'flaky-dead'], env);
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
  for (const t1 = Date.now(); Date.now() - t1 < 60000; await new Promise((r) => setTimeout(r, 1000))) { try { sh('python3', [CLI, 'status', 'flaky-dead'], env); } catch (e) { if (e.status === 1) break; } }
  const res = (id) => { try { return JSON.parse(sh('python3', [CLI, 'fetch', id, '--out', path.join(S, 'out-' + id)], env)); } catch (e) { return JSON.parse(e.stdout); } };
  for (const id of ['flaky-fb', 'flaky-job', 'flaky-dead', 'bad-1']) { const r = res(id); console.log(`${id}: ${r.status} attempts=${(r.attempts || []).length} logs=${JSON.stringify(r.logs)} error=${JSON.stringify((r.error || '').split('\n')[0])}`); }
  console.log('want: flaky-fb done 0 attempts with 2 retry logs; flaky-job done 1 attempt; flaky-dead error 2 attempts; bad-1 error 0 attempts, message "Error: boom"');
  // an open panel survives polls, and its Copy button copies the error
  const pg = (await p.$('details[data-key="bad-1"]')) ? p : p2;
  await pg.evaluate(() => { document.querySelector('details[data-key="bad-1"]').open = true; });
  await pg.waitForTimeout(12000);
  console.log('bad-1 panel still open after 12 s of polls (want true):', await pg.evaluate(() => document.querySelector('details[data-key="bad-1"]')?.open));
  await pg.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await pg.click('details[data-key="bad-1"] [data-copy]');
  console.log('copied:', JSON.stringify(await pg.evaluate(() => navigator.clipboard.readText())));
  // targeting: name the two tabs, send a job to each by name, and one to a tab that isn't open
  const fails = [];
  const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails.push(what); };
  for (const [pg_, name] of [[p, 'alpha'], [p2, 'beta']]) { await pg_.fill('#tabName', name); await pg_.click('#saveName'); }
  const tabIds = { alpha: (await p.textContent('#dev')).match(/· id ([a-z0-9]{8})/)[1], beta: (await p2.textContent('#dev')).match(/· id ([a-z0-9]{8})/)[1] };
  console.log('tab ids:', JSON.stringify(tabIds));
  await p.waitForTimeout(6000);   // both renames committed and rescanned
  fs.writeFileSync('who.js', 'await new Promise((r) => setTimeout(r, 1500)); return { ok: 1 };');
  // submit order alternates so that, untargeted, each tab would take some of the other's jobs
  for (const [id, t] of [['tgt-b1', 'beta'], ['tgt-a1', 'alpha'], ['tgt-b2', 'BETA'], ['tgt-a2', tabIds.alpha], ['tgt-none', 'nosuchtab']]) sh('python3', [CLI, 'submit', 'who.js', '--id', id, '--target', t], env);
  // a result commit that fails: the page must commit an error result instead, and claim once
  fs.writeFileSync('rej.js', 'return { x: "mock-reject-blob" };');
  sh('python3', [CLI, 'submit', 'rej.js', '--id', 'fail-commit', '--target', 'alpha'], env);
  // a job whose error result can't be committed either: the page must release its claim, once, and not take it again
  sh('python3', [CLI, 'submit', 'who.js', '--id', 'mock-reject-blob-1', '--target', 'beta'], env);
  const subj = () => sh('git', ['--git-dir', 'bare.git', 'log', '--format=%s', 'main']).trim().split('\n');
  for (const t1 = Date.now(); Date.now() - t1 < 120000; await new Promise((r) => setTimeout(r, 1500))) {
    const ss = subj();
    if (['tgt-a1', 'tgt-a2', 'tgt-b1', 'tgt-b2'].every((j) => ss.includes(`gpu-bridge: ${j} done`)) && ss.includes('gpu-bridge: fail-commit error') && ss.includes('gpu-bridge: release mock-reject-blob-1')) break;
  }
  await p.waitForTimeout(20000);   // several more polls: no tab may take the released job back, or the untargeted-elsewhere one
  const ss = subj(), count = (x) => ss.filter((y) => y === x).length;
  for (const [j, want] of [['tgt-a1', 'alpha'], ['tgt-a2', 'alpha'], ['tgt-b1', 'beta'], ['tgt-b2', 'beta']]) {
    const r = res(j);
    check(r.status === 'done' && r.tabName === want && r.tab === tabIds[want] && count(`gpu-bridge: claim ${j}`) === 1, `${j} ran once, in ${want} (got ${r.status} in ${r.tabName}/${r.tab}, ${count(`gpu-bridge: claim ${j}`)} claims)`);
  }
  check(count('gpu-bridge: claim tgt-none') === 0, 'tgt-none (target nosuchtab) was never claimed');
  let noneSt; try { sh('python3', [CLI, 'status', 'tgt-none'], env); noneSt = 'exit 0'; } catch (e) { noneSt = 'exit ' + e.status; }
  check(noneSt === 'exit 3', `tgt-none still pending (${noneSt})`);
  const p2txt = await p2.$$eval('.job', (e) => e.map((x) => x.textContent.replace(/\s+/g, ' ').trim()).filter((t) => t.startsWith('tgt-none')));
  check(p2txt.some((t) => /queued.*for nosuchtab, not this tab \(beta\)/.test(t)), `beta's row for tgt-none says it is for another tab: ${JSON.stringify(p2txt)}`);
  const ptxt = await p.$$eval('.job', (e) => e.map((x) => x.textContent.replace(/\s+/g, ' ').trim()).filter((t) => /^tgt-b/.test(t)));
  check(ptxt.length > 0 && ptxt.every((t) => !/taken/.test(t)), `alpha never showed beta's jobs as taken: ${JSON.stringify(ptxt)}`);
  const fc = res('fail-commit');
  check(fc.status === 'error' && /^Could not commit the result: .*blob rejected/.test(fc.error || '') && count('gpu-bridge: claim fail-commit') === 1 && fc.tabName === 'alpha', `fail-commit: error result committed in place of the rejected one, 1 claim (${fc.status}, ${count('gpu-bridge: claim fail-commit')} claims, ${JSON.stringify((fc.error || '').slice(0, 90))})`);
  const head = sh('git', ['--git-dir', 'bare.git', 'ls-tree', '-r', '--name-only', 'main']).split('\n');
  check(count('gpu-bridge: claim mock-reject-blob-1') === 1 && count('gpu-bridge: release mock-reject-blob-1') === 1 && !head.includes('results/mock-reject-blob-1/claim.json') && !head.includes('results/mock-reject-blob-1/result.json'),
    `mock-reject-blob-1: claimed once, claim released, no result (${count('gpu-bridge: claim mock-reject-blob-1')} claims, ${count('gpu-bridge: release mock-reject-blob-1')} releases)`);
  const relTxt = await p2.$$eval('.job', (e) => e.map((x) => x.textContent.replace(/\s+/g, ' ').trim()).filter((t) => t.startsWith('mock-reject-blob-1')));
  console.log('beta row for the released job:', JSON.stringify(relTxt));
  const br = JSON.parse(sh('python3', [CLI, 'bridge'], env));
  check(['alpha', 'beta'].every((n) => br.tabs.some((t) => t.name === n && t.tab === tabIds[n] && t.device && t.lastSeenAgoMin < 5)), `bridge lists both tabs by name: ${JSON.stringify(br.tabs.map((t) => [t.name, t.device, t.tab, t.lastSeenAgoMin]))}`);
  const w = require('child_process').spawnSync('python3', [CLI, 'submit', 'who.js', '--id', 'tgt-warn2', '--target', 'gamma'], { encoding: 'utf8', env: { ...process.env, ...env } }); const warn = w.stderr;
  check(/no tab matching 'gamma'/.test(warn), `submit --target warns about a tab that isn't open: ${JSON.stringify(warn.trim())}`);
  console.log(`targeting and failure handling: ${fails.length ? 'FAIL (' + fails.length + ')' : 'PASS'}`);
  // a fresh tab lists earlier results from the commit log and loads one on demand
  const p3 = await b.newPage(); await p3.goto('http://localhost:8765/?api=http://localhost:8766'); await p3.waitForTimeout(1500);
  await p3.fill('#token', 'test-token'); await p3.click('#connect'); await p3.waitForTimeout(10000);
  console.log('fresh tab history:', JSON.stringify(await p3.$$eval('.job', (e) => e.map((x) => x.querySelector('.id')?.textContent + ':' + x.querySelector('.st')?.textContent).slice(0, 12))));
  await p3.evaluate(() => { const d = document.querySelector('details[data-key="flaky-dead"]'); if (d) d.open = true; }); await p3.waitForTimeout(2000);
  console.log('fresh tab history, after:', JSON.stringify(await p3.$$eval('.job', (e) => e.map((x) => x.querySelector('.id')?.textContent + ':' + x.querySelector('.st')?.textContent).slice(0, 12))));
  console.log('fresh tab flaky-dead panel:', JSON.stringify(await p3.evaluate(() => document.querySelector('details[data-key="flaky-dead"]')?.textContent.slice(0, 300))));
  await p3.screenshot({ path: 'page3.png', fullPage: true });
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
    m = /^gpu-bridge: (?:release (\S+)|(\S+) (?:done|error))$/.exec(subj); if (m) m[1] = m[1] || m[2];
    if (m && tabOf[m[1]] && open[tabOf[m[1]]] === m[1]) open[tabOf[m[1]]] = null;
  }
  console.log('overlapping jobs within a tab (want 0):', overlaps);
  console.log('repo log:', sh('git', ['--git-dir', 'bare.git', 'log', '--format=%s', 'main']).trim().split('\n').join(' | '));
  console.log('h-overflow:', await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), 'errors:', errs);
  await p.screenshot({ path: 'page.png', fullPage: true });
  await b.close(); web.close(); api.kill();
  if (fails.length) { console.error('E2E FAIL:', fails.join('; ')); process.exit(1); }
})().catch((e) => { console.error('E2E FAIL', e); process.exit(1); });
