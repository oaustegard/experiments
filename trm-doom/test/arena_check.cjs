// Headless smoke test of arena.html (SwiftShader, BFS stand-in, no token): plays until two attempts end.
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, "../../..");
const types = { ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };
const srv = http.createServer((req, res) => { const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); fs.createReadStream(f).pipe(res); }).listen(0, "127.0.0.1", async () => {
  const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
  const pg = await b.newPage(); const errs = [];
  pg.on("pageerror", (e) => errs.push(e.message)); pg.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
  await pg.goto(`http://127.0.0.1:${srv.address().port}/experiments/trm-doom/web/arena.html?doom=/oaustegard.github.io/fun-and-games/doom/&planner=oracle&auto&fast&map=E1M1&stuck=8&cap=1&tries=2`);
  try { await pg.waitForFunction("window.ARENA && window.ARENA.attempts >= 2 && window.ARENA.outbox.length >= 3", null, { timeout: 900000 }); } catch (e) { console.log("timeout"); }
  console.log(JSON.stringify({ errs: errs.slice(0, 5), log: await pg.evaluate("window.ARENA && window.ARENA.log"), outbox: await pg.evaluate("window.ARENA && window.ARENA.outbox.map(j => [j.message, Object.keys(j.files).length])"), status: await pg.evaluate("document.getElementById('status').textContent") }, null, 1));
  await b.close(); srv.close();
});
