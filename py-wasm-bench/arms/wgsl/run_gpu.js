// Runs one translated kernel on WebGPU in headless Chromium, driven by playwright-core.
//
//   node run_gpu.js out/<kernel>.json [--runs 5]
//
// Reads the manifest that build.py wrote, serves page/ on localhost, and calls
// window.runKernel (page/index.html) with the shader text and the input buffers. The
// page checks getCompilationInfo(), dispatches, reads back and times each run.
// Writes out/<kernel>.<buffer>.gpu.f32 for every output buffer.
//
// Last stdout line is JSON: {ok, kernel, n, adapter, gpu_ms_median, times_ms, outputs}.
// On failure ok is false, with the stage, the error, navigator.gpu and Chromium stderr.
const fs = require("fs");
const path = require("path");
const http = require("http");
const { chromium } = require("playwright-core");

const HERE = __dirname;
const CHROME = process.env.CHROME_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const CHROME_ARGS = [
  "--enable-unsafe-webgpu",
  "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader",
  "--no-sandbox",
];

function arg(name, dflt) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

function serve(dir) {
  const srv = http.createServer((req, res) => {
    const file = path.join(dir, req.url === "/" ? "index.html" : req.url.replace(/^\/+/, ""));
    if (!file.startsWith(dir) || !fs.existsSync(file)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": file.endsWith(".html") ? "text/html" : "application/octet-stream" });
    res.end(fs.readFileSync(file));
  });
  return new Promise((resolve) => srv.listen(0, "127.0.0.1", () => resolve(srv)));
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

async function main() {
  const manifestPath = path.resolve(process.argv[2] || "");
  const runs = parseInt(arg("--runs", "5"), 10);
  const outDir = path.dirname(manifestPath);
  const man = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const wgsl = fs.readFileSync(path.join(outDir, man.wgsl), "utf8");
  const inputs = {};
  for (const b of man.buffers) {
    if (b.file) inputs[b.name] = fs.readFileSync(path.join(outDir, b.file)).toString("base64");
  }

  const srv = await serve(path.join(HERE, "page"));
  const url = `http://127.0.0.1:${srv.address().port}/`;
  const stderr = [];
  let browser;
  const fail = (error, extra) => {
    console.log(JSON.stringify({ ok: false, kernel: man.kernel, n: man.n, error: String(error), chromium_stderr: stderr.join("").slice(-2000), ...extra }));
    process.exitCode = 1;
  };
  try {
    browser = await chromium.launch({ executablePath: CHROME, headless: true, args: CHROME_ARGS, timeout: 120000 });
    const proc = typeof browser.process === "function" ? browser.process() : null;
    if (proc && proc.stderr) proc.stderr.on("data", (d) => stderr.push(String(d)));
    const page = await browser.newPage();
    await page.goto(url);
    const r = await page.evaluate((job) => window.runKernel(job), { spec: man, wgsl, inputs, runs });
    if (!r.ok) {
      fail(`${r.stage}: ${r.error}`, { stage: r.stage, navigator_gpu: r.navigator_gpu, adapter: r.adapter, messages: r.messages });
      return;
    }
    const outputs = {};
    for (const [name, b64] of Object.entries(r.outputs)) {
      const file = `${man.kernel}.${name}.gpu.f32`;
      fs.writeFileSync(path.join(outDir, file), Buffer.from(b64, "base64"));
      outputs[name] = path.join(outDir, file);
    }
    console.log(JSON.stringify({
      ok: true,
      kernel: man.kernel,
      n: man.n,
      grid: man.grid,
      workgroups: man.workgroups,
      navigator_gpu: r.navigator_gpu,
      adapter: r.adapter,
      messages: r.messages,
      times_ms: r.times_ms,
      gpu_ms_median: median(r.times_ms),
      outputs,
    }));
  } catch (e) {
    fail(e && e.stack ? e.stack.split("\n").slice(0, 3).join(" | ") : e);
  } finally {
    if (browser) await browser.close().catch(() => {});
    srv.close();
  }
}

main();
