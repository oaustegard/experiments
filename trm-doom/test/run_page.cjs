// Serve the experiments checkout and run a page in headless Chromium on SwiftShader; print window.RESULT.
// node test/run_page.cjs PAGE [timeoutSec] [query]
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, "../../..");  // .spokes: serves experiments/ and oaustegard.github.io/
const [page, tsec = "300", query = ""] = process.argv.slice(2);
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".json": "application/json", ".wasm": "application/wasm" };
const srv = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
  fs.createReadStream(f).pipe(res);
}).listen(0, "127.0.0.1", async () => {
  const url = `http://127.0.0.1:${srv.address().port}/experiments/trm-doom/${page}${query ? "?" + query : ""}`;
  const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--no-sandbox", "--enable-unsafe-webgpu",
    "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
  const pg = await b.newPage();
  pg.on("console", (m) => console.log("console:", m.text()));
  pg.on("pageerror", (e) => console.log("pageerror:", e.message));
  await pg.goto(url);
  await pg.waitForFunction("window.RESULT !== undefined", null, { timeout: +tsec * 1000 });
  console.log(JSON.stringify(await pg.evaluate("window.RESULT"), null, 1));
  await b.close(); srv.close();
});
