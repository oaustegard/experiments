// The connection pill against a stubbed GitHub: a good token, then a bad one.
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, "../../..");
const types = { ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };
const srv = http.createServer((req, res) => { const f = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); fs.createReadStream(f).pipe(res); }).listen(0, "127.0.0.1", async () => {
  const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium", args: ["--no-sandbox"] });
  const pg = await b.newPage(); const errs = []; pg.on("pageerror", (e) => errs.push(e.message));
  await pg.route("https://api.github.com/**", (route) => {
    const u = new URL(route.request().url()), auth = route.request().headers().authorization || "";
    const json = (status, body) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body), headers: { "access-control-allow-origin": "*" } });
    if (!auth.endsWith("good")) return json(401, { message: "Bad credentials" });
    if (u.pathname === "/repos/oaustegard/gpu-bridge-comms") return json(200, { permissions: { push: true } });
    if (u.pathname === "/user") return json(200, { login: "oaustegard" });
    if (u.pathname.endsWith("/pulls")) return json(200, [{ number: 1, head: { ref: "wake/sessions" } }]);
    return json(404, {});
  });
  await pg.goto(`http://127.0.0.1:${srv.address().port}/experiments/trm-doom/web/arena.html?doom=/oaustegard.github.io/fun-and-games/doom/`);
  const pill = () => pg.evaluate(() => document.getElementById("connstate").textContent);
  console.log("initial:", await pill());
  for (const tok of ["nope", "good"]) {
    await pg.evaluate(() => { document.getElementById("conn").hidden = false; });
    await pg.fill("#token", tok); await pg.click("#connect"); await pg.waitForTimeout(1500);
    console.log(tok + ":", await pill(), "| form hidden:", await pg.evaluate(() => document.getElementById("conn").hidden));
  }
  await pg.reload(); await pg.waitForTimeout(1500); console.log("after reload:", await pill());
  console.log("errors:", errs);
  await b.close(); srv.close();
});
