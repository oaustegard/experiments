// Render hero.html in headless Chromium (SwiftShader WebGPU) and write hero.png: node render.cjs [w] [h]
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright");
const root = __dirname, [w = "1600", h = "840"] = process.argv.slice(2);
const srv = http.createServer((req, res) => { const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": f.endsWith(".html") ? "text/html" : "application/json" }); fs.createReadStream(f).pipe(res); }).listen(0, "127.0.0.1", async () => {
  const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
  const pg = await b.newPage(); pg.on("pageerror", (e) => console.log("pageerror:", e.message));
  await pg.goto(`http://127.0.0.1:${srv.address().port}/hero.html?w=${w}&h=${h}`);
  await pg.waitForFunction("window.RESULT !== undefined", null, { timeout: 600000 });
  const r = await pg.evaluate("window.RESULT");
  if (r.error) console.log(r.error); else { fs.writeFileSync(path.join(root, "hero.png"), Buffer.from(r.png.split(",")[1], "base64")); console.log("wrote hero.png"); }
  await b.close(); srv.close();
});
