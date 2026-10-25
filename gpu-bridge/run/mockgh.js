// Minimal GitHub REST mock over a real bare repo: just the calls the bridge page makes.
const http = require('http'), { execFileSync } = require('child_process'), fs = require('fs'), path = require('path'), os = require('os');
const BARE = process.argv[2]; const comments = [];
const git = (args, opts = {}) => execFileSync('git', ['--git-dir', BARE, ...args], { encoding: opts.buf ? null : 'utf8', input: opts.input, env: { ...process.env, ...(opts.env || {}) } });
const send = (res, code, body, headers = {}) => { res.writeHead(code, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Expose-Headers': 'ETag', 'Content-Type': Buffer.isBuffer(body) ? 'application/octet-stream' : 'application/json', ...headers }); res.end(Buffer.isBuffer(body) ? body : JSON.stringify(body)); };
http.createServer((req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'GET,POST,PATCH' }); return res.end(); }
  let body = ''; req.on('data', (c) => body += c); req.on('end', () => {
    try {
      const u = new URL(req.url, 'http://x'), p = u.pathname.replace(/^\/repos\/[^/]+\/[^/]+/, ''), j = body ? JSON.parse(body) : null;
      if (req.url === '/__comments') return send(res, 200, comments);
      if (p === '' ) return send(res, 200, { full_name: 'o/r', permissions: { push: true } });
      if (p === '/commits/main') { const sha = git(['rev-parse', 'main']).trim(); if (req.headers['if-none-match'] === `"${sha}"`) { res.writeHead(304, { 'Access-Control-Allow-Origin': '*' }); return res.end(); } return send(res, 200, { sha, commit: { tree: { sha: git(['rev-parse', 'main^{tree}']).trim() } } }, { ETag: `"${sha}"` }); }
      let m;
      if ((m = /^\/git\/trees\/(\w+)$/.exec(p)) && req.method === 'GET') return send(res, 200, { tree: git(['ls-tree', '-r', m[1]]).trim().split('\n').filter(Boolean).map((l) => { const [meta, path] = l.split('\t'); const [, type, sha] = meta.split(' '); return { path, type, sha }; }) });
      if ((m = /^\/git\/blobs\/(\w+)$/.exec(p))) return send(res, 200, git(['cat-file', 'blob', m[1]], { buf: true }));
      if (p === '/git/blobs') return send(res, 201, { sha: git(['hash-object', '-w', '--stdin'], { input: j.encoding === 'base64' ? Buffer.from(j.content, 'base64') : Buffer.from(j.content) }).trim() });
      if (p === '/git/ref/heads/main') return send(res, 200, { object: { sha: git(['rev-parse', 'main']).trim() } });
      if ((m = /^\/git\/commits\/(\w+)$/.exec(p))) return send(res, 200, { tree: { sha: git(['rev-parse', m[1] + '^{tree}']).trim() } });
      if (p === '/git/trees') { const idx = path.join(os.tmpdir(), 'idx-' + Date.now() + Math.random()); const env = { GIT_INDEX_FILE: idx }; git(['read-tree', j.base_tree], { env }); for (const e of j.tree) git(['update-index', '--add', '--cacheinfo', `${e.mode},${e.sha},${e.path}`], { env }); const t = git(['write-tree'], { env }).trim(); fs.rmSync(idx, { force: true }); return send(res, 201, { sha: t }); }
      if (p === '/git/commits') return send(res, 201, { sha: git(['-c', 'user.name=page', '-c', 'user.email=p@p', 'commit-tree', j.tree, ...j.parents.flatMap((x) => ['-p', x]), '-m', j.message]).trim() });
      if (p === '/git/refs/heads/main' && req.method === 'PATCH') { const old = git(['rev-parse', 'main']).trim(); try { git(['merge-base', '--is-ancestor', old, j.sha]); } catch { return send(res, 422, { message: 'Update is not a fast forward' }); } git(['update-ref', 'refs/heads/main', j.sha, old]); return send(res, 200, { object: { sha: j.sha } }); }
      if (p === '/pulls') return send(res, 200, [{ number: 1 }]);
      if (p === '/issues/1/comments') { comments.push(j.body); return send(res, 201, { id: comments.length }); }
      send(res, 404, { message: 'mock: ' + req.method + ' ' + p });
    } catch (e) { send(res, 500, { message: String(e) }); }
  });
}).listen(8766);
