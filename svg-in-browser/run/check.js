// Runs web/trace.js headless on both backends and writes the SVGs + stats.
// Usage: node run/check.js <out_dir> [K] [side]
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [outDir, K = "56", side = "1000"] = process.argv.slice(2);
const root = path.join(__dirname, "..", "web");
const types = { ".html": "text/html", ".js": "text/javascript", ".jpg": "image/jpeg" };
const server = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  fs.readFile(f, (err, buf) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(buf);
  });
}).listen(0, "127.0.0.1", async () => {
  const port = server.address().port;
  const browser = await chromium.launch({
    executablePath: "/opt/pw-browsers/chromium",
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader",
           "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"],
  });
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && console.error("page:", m.text()));
  await page.goto(`http://127.0.0.1:${port}/index.html?manual`);
  await page.waitForFunction(() => window.imageToSvg && document.getElementById("src").complete);
  fs.mkdirSync(outDir, { recursive: true });
  for (const backend of ["webgpu", "js"]) {
    const r = await page.evaluate(async ({ backend, K, side }) => {
      const device = backend === "webgpu" ? await window.getDevice() : null;
      if (backend === "webgpu" && !device) return null;
      const img = window.pixels(document.getElementById("src"), side);
      return window.imageToSvg(img, { K, device });
    }, { backend, K: +K, side: +side });
    if (!r) { console.log(backend, "unavailable"); continue; }
    fs.writeFileSync(path.join(outDir, `${backend}.svg`), r.svg);
    console.log(backend, JSON.stringify(r.stats));
  }
  await browser.close();
  server.close();
});
