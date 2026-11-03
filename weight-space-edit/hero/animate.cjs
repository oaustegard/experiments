// Render the hero animation frame by frame and encode it: node animate.cjs OUTDIR [w] [h] [fps] [seconds] [hold]
// The automaton step t runs from 0 to 2D + 8 (wave out to the start, marks back to the goal), then holds.
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright");
const root = __dirname, [out, w = "1280", h = "672", fps = "30", secs = "7", hold = "2", only = ""] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const srv = http.createServer((req, res) => { const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": f.endsWith(".html") ? "text/html" : "application/json" }); fs.createReadStream(f).pipe(res); }).listen(0, "127.0.0.1", async () => {
  const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
  const pg = await b.newPage(); pg.on("pageerror", (e) => console.log("pageerror:", e.message));
  await pg.goto(`http://127.0.0.1:${srv.address().port}/hero.html?w=${w}&h=${h}`);
  await pg.waitForFunction("window.RESULT !== undefined", null, { timeout: 600000 });
  const { D } = await pg.evaluate("window.META"), tEnd = 2 * D + 8;
  const nAnim = Math.round(fps * secs), n = nAnim + Math.round(fps * hold);
  const frames = only ? only.split(",").map(Number) : [...Array(n).keys()];
  for (const f of frames) {
    const x = Math.min(1, f / (nAnim - 1));
    const t = (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2) * tEnd; // ease in and out
    const url = await pg.evaluate((t) => window.frame(t), t);
    fs.writeFileSync(path.join(out, `f${String(f).padStart(4, "0")}.png`), Buffer.from(url.split(",")[1], "base64"));
    if (f % 30 === 0) console.log("frame", f, "of", n, "t", t.toFixed(1));
  }
  await b.close(); srv.close();
});
