// Headless test of the demo page's link parameters and firehose mode (SwiftShader WebGPU in Chromium).
//
//   NODE_PATH=$(npm root -g) node run/page_smoke2.mjs TOKENIZERS_MJS
//
// TOKENIZERS_MJS as for run/page_smoke.mjs. Serves the experiments repo on 127.0.0.1; the AppView is stubbed
// from run/fixtures/bsky_author_feed.json and WebSocket is replaced (addInitScript) by a fake that replays
// run/fixtures/jetstream_events.json on the test's command. No live network.
//   e. malformed parameters, no WebGPU: defaults kept, a label holding markup shown as text.
//   a. ?source=fixture.example.com&n=30&labels=person,company: says it waits for the model, starts by itself
//      once the model is ready, analyses all 30 posts, the grouped view has only those two labels.
//   b. a changed threshold reaches the address bar; Copy link copies that URL.
//   d. ?mode=firehose opens no socket; with &start=1 one socket opens once the model is ready.
//   c. firehose on the fixture (sample=5, lang=en, replies=0, minlen=20, exclude=giveaway, window=15): the
//      counters match an independent count of the fixture (events, post creates, passed, sampled = exactly
//      ceil(passed / 5), analysed), entities appear, markup in a post renders as text, Microsoft's window count
//      goes down to nothing once phase-2 posts push it out, the window holds exactly 15 posts, renders stay
//      at most two a second, a burst drops the oldest past 2,000 queued and counts them, Stop closes the
//      socket and nothing reconnects, switching tabs while running closes it too.
// Screenshots (deviceScaleFactor 2): out/page_link_source.png, out/page_link_source_phone.png,
// out/page_firehose.png, out/page_firehose_phone.png. Exits 0 on PASS; details in out/page_smoke2.json.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(ROOT, "..");
const tokLib = process.argv[2];
if (!tokLib || !existsSync(tokLib)) { console.error("usage: node run/page_smoke2.mjs TOKENIZERS_MJS"); process.exit(2); }
const fx = JSON.parse(readFileSync(resolve(ROOT, "run", "fixtures", "bsky_author_feed.json"), "utf8"));
const js = JSON.parse(readFileSync(resolve(ROOT, "run", "fixtures", "jetstream_events.json"), "utf8"));
const OUT = resolve(ROOT, "out");
mkdirSync(OUT, { recursive: true });
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const base = `http://127.0.0.1:${port}/${ROOT.split("/").pop()}/web/index.html`;
const res = { checks: {}, errors: [], offsite: [], appview: [] };
const check = (name, ok, detail) => { res.checks[name] = { ok: !!ok, detail }; console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail !== undefined ? " " + JSON.stringify(detail).slice(0, 400) : ""}`); };

// Independent count of what the firehose should do with the fixture under the test's settings.
const SAMPLE = 5, WINDOW = 15;
function expected(events) {
  const c = { events: 0, posts: 0, passed: 0 };
  for (const e of events) {
    c.events++;
    const m = e.commit;
    if (e.kind !== "commit" || m?.operation !== "create" || m.collection !== "app.bsky.feed.post" || !m.record) continue;
    c.posts++;
    const r = m.record, text = typeof r.text === "string" ? r.text : "";
    if (r.reply) continue;
    if (!(r.langs || []).some((l) => l.toLowerCase() === "en" || l.toLowerCase().startsWith("en-"))) continue;
    if ([...text].length < 20) continue;
    if (text.toLowerCase().includes("giveaway")) continue;
    c.passed++;
  }
  return c;
}
const ex1 = expected(js.events.slice(0, js.phase2_from)), exAll = expected(js.events);
ex1.sampled = Math.ceil(ex1.passed / SAMPLE); exAll.sampled = Math.ceil(exAll.passed / SAMPLE);
res.expected = { phase1: ex1, all: exAll };

// Fake WebSocket: records each socket; opens after 10 ms; __fws.replay(a, b) sends fixture events a..b-1
// synchronously, __fws.burst(n) sends n synthetic passing posts.
function fakeWs(events) {
  const W = window.__fws = { sockets: [], closed: 0, events, t: [] };
  class FakeWS {
    constructor(url) {
      this.url = String(url); this.readyState = 0; this.created = performance.now();
      W.sockets.push(this);
      setTimeout(() => { if (this.readyState === 0) { this.readyState = 1; this.onopen?.({ type: "open" }); } }, 10);
    }
    send() {}
    close(code = 1000) {
      if (this.readyState >= 2) return;
      this.readyState = 3; W.closed++;
      setTimeout(() => this.onclose?.({ type: "close", code, wasClean: true }), 0);
    }
    _emit(obj) { if (this.readyState === 1) this.onmessage?.({ type: "message", data: JSON.stringify(obj) }); }
  }
  FakeWS.CONNECTING = 0; FakeWS.OPEN = 1; FakeWS.CLOSING = 2; FakeWS.CLOSED = 3;
  window.WebSocket = FakeWS;
  W.replay = (a, b) => { const s = W.sockets.at(-1); for (const e of W.events.slice(a, b)) s._emit(e); };
  W.burst = (n, t0) => {
    const s = W.sockets.at(-1);
    for (let i = 0; i < n; i++) s._emit({ did: "did:plc:burst", time_us: t0 + i + 1, kind: "commit",
      commit: { operation: "create", collection: "app.bsky.feed.post", rkey: "b" + i, record: { text: `Burst post number ${i} about nothing much`, langs: ["en"] } } });
  };
}

async function newPage(browser, opts, init) {
  const ctx = await browser.newContext(opts);
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") res.errors.push("console: " + m.text()); });
  page.on("pageerror", (e) => res.errors.push("pageerror: " + String(e)));
  if (init) await page.addInitScript(...init);
  await page.route("**/*", (route) => {
    const u = new URL(route.request().url());
    if (u.hostname === "127.0.0.1") return route.continue();
    if (u.hostname === "cdn.jsdelivr.net") return route.fulfill({ status: 200, contentType: "text/javascript", body: readFileSync(tokLib) });
    if (u.hostname === "public.api.bsky.app") {
      res.appview.push(u.pathname.split("/").pop() + u.search);
      const m = u.pathname.split("/").pop(), q = u.searchParams;
      let body;
      if (m === "com.atproto.identity.resolveHandle" && q.get("handle") === fx.handle) body = { did: fx.did };
      else if (m === "app.bsky.feed.getAuthorFeed" && q.get("actor") === fx.did) body = fx.pages[q.get("cursor") ?? ""];
      if (!body) return route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ error: "InvalidRequest", message: "not in fixture" }) });
      return route.fulfill({ status: 200, contentType: "application/json", headers: { "access-control-allow-origin": "*" }, body: JSON.stringify(body) });
    }
    res.offsite.push(u.href);
    return route.abort();
  });
  return { ctx, page };
}
const waitReady = (page) => page.waitForFunction("window.app && (app.state === 'ready' || app.state === 'error')", null, { timeout: 900_000 });
const noOverflow = (page) => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

let browser;
try {
  browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });

  // e. malformed parameters (no WebGPU, so no model)
  {
    const { ctx, page } = await newPage(browser, { viewport: { width: 900, height: 700 } }, [() => { delete Navigator.prototype.gpu; }]);
    const q = "?labels=" + encodeURIComponent("<img src=x onerror=window.__xss=1>,person, person ,,") +
      "&threshold=7&n=abc&reposts=maybe&mode=nonsense&sample=3&lang=" + encodeURIComponent("<b>") + "&window=5&minlen=-1&source=%20";
    await page.goto(base + q);
    await page.waitForFunction("window.app && app.state === 'nogpu'", null, { timeout: 30_000 });
    const s = await page.evaluate(() => ({ chips: [...document.querySelectorAll(".chip > span")].map((x) => x.textContent), imgs: document.querySelectorAll("img").length,
      xss: window.__xss, threshold: app.threshold, mode: app.mode, textTab: document.getElementById("tab-text").getAttribute("aria-selected"),
      n: document.getElementById("bsky-count").value, sample: document.getElementById("fh-sample").value, lang: document.getElementById("fh-lang").value,
      window: document.getElementById("fh-window").value, minlen: document.getElementById("fh-minlen").value, reposts: document.getElementById("bsky-reposts").checked }));
    check("e. malformed parameters keep defaults; markup in a label is text", s.chips.length === 2 && s.chips[0] === "<img src=x onerror=window.__xss=1>" && s.chips[1] === "person"
      && s.imgs === 0 && s.xss === undefined && s.threshold === 0.5 && s.mode === "text" && s.textTab === "true" && s.n === "200" && s.sample === "1"
      && s.lang === "en" && s.window === "1000" && s.minlen === "20" && s.reposts === false, s);
    await ctx.close();
  }

  // a. a source link starts by itself once the model is ready
  {
    const { ctx, page } = await newPage(browser, { viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2, permissions: ["clipboard-read", "clipboard-write"] });
    await page.goto(base + "?source=fixture.example.com&n=30&labels=person,company");
    const early = await page.evaluate(() => ({ state: app.state, progress: document.getElementById("bsky-progress").textContent, bsky: !!app.bsky,
      tab: document.getElementById("tab-bsky").getAttribute("aria-selected"), input: document.getElementById("bsky-input").value }));
    check("a. before the model is ready: Bluesky tab, input filled, says it is waiting, nothing fetched", early.state !== "ready" && !early.bsky
      && /Waiting for the model/.test(early.progress) && early.tab === "true" && early.input === "fixture.example.com" && res.appview.length === 0, early);
    await waitReady(page);
    await page.waitForFunction("app.bsky && app.bsky.done && !app.bsky.running", null, { timeout: 600_000 });
    const a = await page.evaluate(() => ({ started: app.bsky.t0 >= app.readyAt, analysed: app.bsky.analysed, error: app.bsky.error && String(app.bsky.error),
      labels: app.labels, groups: [...document.querySelectorAll("#groups .group")].map((g) => g.dataset.label), search: location.search }));
    res.a = a;
    check("a. started by itself after the model loaded, all 30 posts analysed", a.started && a.analysed === 30 && !a.error, a);
    check("a. grouped view has only person and company", a.labels.join() === "person,company" && a.groups.length >= 1 && a.groups.every((l) => l === "person" || l === "company") && a.groups.includes("company"), a.groups);
    check("a. address bar carries the source settings", /source=fixture\.example\.com/.test(a.search) && /n=30/.test(a.search) && /labels=person,company/.test(a.search) && /mode=bluesky/.test(a.search), a.search);
    await page.screenshot({ path: resolve(OUT, "page_link_source.png"), fullPage: true });

    // b. threshold -> address bar -> Copy link
    await page.$eval("#threshold", (el) => { el.value = "0.35"; el.dispatchEvent(new Event("input", { bubbles: true })); });
    const moved = await page.waitForFunction(() => /threshold=0\.35/.test(location.search), null, { timeout: 5000 }).then(() => true, () => false);
    await page.click("#copy-link");
    await page.waitForFunction("app.copied", null, { timeout: 5000 }).catch(() => {});
    const b = await page.evaluate(async () => ({ href: location.href, copied: app.copied, clip: await navigator.clipboard.readText().catch((e) => "ERR " + e),
      button: document.getElementById("copy-link").textContent }));
    check("b. changed threshold is in the address bar", moved && /threshold=0\.35/.test(b.href) && /source=fixture\.example\.com/.test(b.href), b.href);
    check("b. Copy link copies that URL and confirms", b.clip === b.href && b.copied === b.href && /copied/i.test(b.button), b);
    const reopened = new URL(b.href);
    check("b. copied link has no start flag and keeps labels", !reopened.searchParams.has("start") && reopened.searchParams.get("labels") === "person,company", reopened.search);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(400);
    check("a. no horizontal scroll at 390 px (Bluesky tab)", (await noOverflow(page)) <= 0);
    await page.screenshot({ path: resolve(OUT, "page_link_source_phone.png"), fullPage: true });
    await ctx.close();
  }

  // d + c. firehose with a fake WebSocket
  {
    const { ctx, page } = await newPage(browser, { viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 }, [fakeWs, js.events]);
    await page.goto(base + "?mode=firehose");
    await waitReady(page);
    await page.waitForTimeout(1500);
    const d0 = await page.evaluate(() => ({ sockets: __fws.sockets.length, tab: document.getElementById("tab-firehose").getAttribute("aria-selected"),
      start: !document.getElementById("fh-start").disabled && !document.getElementById("fh-start").hidden }));
    check("d. ?mode=firehose alone opens no socket (Start enabled)", d0.sockets === 0 && d0.tab === "true" && d0.start, d0);

    const q = `?mode=firehose&start=1&sample=${SAMPLE}&lang=en&replies=0&minlen=20&exclude=giveaway&window=${WINDOW}&labels=person,company`;
    await page.goto(base + q);
    const early = await page.evaluate(() => ({ sockets: __fws.sockets.length, state: app.state, stats: document.getElementById("fh-stats").textContent }));
    await waitReady(page);
    await page.waitForFunction("__fws.sockets.length === 1 && __fws.sockets[0].readyState === 1 && app.fh && app.fh.conn.state === 'open'", null, { timeout: 10_000 });
    const d1 = await page.evaluate(() => ({ url: __fws.sockets[0].url, afterReady: __fws.sockets[0].created >= app.readyAt, search: location.search }));
    check("d. &start=1 opens one socket, only once the model is ready, after saying it waits", early.sockets === 0 && early.state !== "ready" && /Waiting for the model/.test(early.stats) && d1.afterReady
      && d1.url === "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post", { early, d1 });
    check("d. address bar keeps the firehose settings, drops start", /mode=firehose/.test(d1.search) && /sample=5/.test(d1.search) && /window=15/.test(d1.search)
      && /exclude=giveaway/.test(d1.search) && !/start=/.test(d1.search), d1.search);

    // pure filter / sampler checks
    const unit = await page.evaluate(async () => {
      const m = await import(new URL("firehose.js", location.href));
      const P = (text, langs = ["en"], reply = false) => ({ text, langs, reply });
      const f = m.makeFilter({ lang: "en", replies: false, minlen: 5, include: "apple, Google", exclude: "giveaway" });
      const s = m.makeSampler(10), picks = [];
      for (let i = 0; i < 1000; i++) if (s()) picks.push(i);
      return { inc: f(P("I like Apple pie")), incG: f(P("GOOGLE it")), none: f(P("nothing relevant here")), exc: f(P("apple giveaway now")),
        reply: f(P("apple reply here", ["en"], true)), de: f(P("apple auf deutsch", ["de"])), enGB: f(P("apple in britain", ["en-GB"])),
        short: f(P("appl")), emoji: m.makeFilter({ minlen: 3, lang: "any" })(P("👍👍", [])), any: m.makeFilter({ lang: "any", minlen: 0 })(P("x", [])),
        picks: picks.length, first: picks.slice(0, 3),
        toPost: [m.toPost({ kind: "identity" }), m.toPost({ kind: "commit", commit: { operation: "delete", collection: "app.bsky.feed.post" } })] };
    });
    check("c. filter (include/exclude/lang/replies/minlen) and sampler units", unit.inc && unit.incG && !unit.none && !unit.exc && !unit.reply && !unit.de && unit.enGB
      && !unit.short && !unit.emoji && unit.any && unit.picks === 100 && unit.first.join() === "0,10,20" && unit.toPost.every((x) => x === null), unit);

    // phase 1
    const tRun0 = Date.now();
    await page.evaluate((n) => __fws.replay(0, n), js.phase2_from);
    await page.waitForFunction((n) => app.fh.counts.analysed === n && !app.fh.queue.length, ex1.sampled, { timeout: 600_000 });
    await page.waitForTimeout(700);
    const c1 = await page.evaluate(() => ({ counts: { ...app.fh.counts }, size: app.fh.groups.size,
      view: app.fh.view.map((g) => ({ label: g.label, entities: g.entities.map((e) => [e.name, e.count, e.key]) })),
      groups: [...document.querySelectorAll("#fh-groups .group")].map((g) => g.dataset.label), stats: document.getElementById("fh-stats").textContent,
      totals: document.getElementById("fh-totals").textContent }));
    res.c1 = c1;
    const k1 = c1.counts;
    check("c. phase 1 counters match the fixture (events, post creates, passed, sampled = ceil(passed/5), analysed)", k1.events === ex1.events && k1.posts === ex1.posts
      && k1.passed === ex1.passed && k1.sampled === ex1.sampled && k1.analysed === ex1.sampled && k1.dropped === 0, { got: k1, want: ex1 });
    const ms1 = c1.view.find((g) => g.label === "company")?.entities.find((e) => e[2] === "microsoft");
    check("c. entities appear in the grouped view, only the two labels", c1.groups.includes("company") && c1.groups.every((l) => l === "person" || l === "company") && ms1 && ms1[1] >= 3, { groups: c1.groups, microsoft: ms1 });
    check("c. stats line: connection, endpoint, rates, queue, dropped", /connected to jetstream2\.us-east\.bsky\.network/.test(c1.stats) && /received [\d.,]+\/s/.test(c1.stats)
      && /passed filters [\d.,]+\/s/.test(c1.stats) && /analysed [\d.,]+\/s/.test(c1.stats) && /queue 0/.test(c1.stats) && /dropped 0/.test(c1.stats), c1.stats);

    // markup in a post renders as text: open an entity of the first (markup) post
    const pick = await page.evaluate(() => {
      const p = [...app.fh.groups.posts.values()].find((x) => x.text.includes("<img"));
      const e = p?.ents.find((x) => x.label === "company") || p?.ents[0];
      return p && e ? { label: e.label, text: e.text } : null;
    });
    let xss = { pick };
    if (pick) {
      const key = await page.evaluate(async (t) => (await import(new URL("bsky.js", location.href))).normEntity(t), pick.text);
      await page.click(`#fh-groups .group[data-label="${pick.label}"] button[data-key="${key}"]`);
      await page.waitForTimeout(700);
      xss = await page.evaluate(() => {
        const d = document.getElementById("fh-detail");
        const posts = [...d.querySelectorAll("li .post")].map((x) => x.textContent);
        return { hidden: d.hidden, posts: posts.length, markupAsText: posts.some((t) => t.includes('<img src=x onerror="window.__xss=1">') && t.includes("<script>window.__xss=2</script>")),
          elements: document.querySelectorAll("#fh-detail img, #fh-detail script, #fh-groups img, #fh-groups script").length, flag: window.__xss,
          links: [...d.querySelectorAll("a")].map((a) => a.href), meta: d.querySelector(".meta")?.textContent };
      });
    }
    res.xss = xss;
    check("c. <script> / <img onerror> in a post rendered as text", !!pick && !xss.hidden && xss.markupAsText && xss.elements === 0 && xss.flag === undefined, xss);
    check("c. detail links are bsky.app/profile/<did>/post/<rkey>, author a shortened DID", xss.links?.length === xss.posts && xss.links.every((h) => /^https:\/\/bsky\.app\/profile\/did:plc:[a-z2-7]{24}\/post\/[a-z2-7]{13}$/.test(h))
      && /did:plc:\w+…\w{4}/.test(xss.meta || ""), { links: xss.links?.slice(0, 2), meta: xss.meta });
    await page.screenshot({ path: resolve(OUT, "page_firehose.png"), fullPage: true });

    // phase 2: Microsoft has to leave the window
    await page.evaluate(() => { app.fh.selected = null; });
    // in four slices, each analysed before the next, so the window moves in steps
    const series = [{ a: k1.analysed, ms: ms1?.[1] ?? 0 }];
    const n2 = js.events.length - js.phase2_from;
    for (let i = 0; i < 4; i++) {
      const a = js.phase2_from + Math.floor((n2 * i) / 4), b = js.phase2_from + Math.floor((n2 * (i + 1)) / 4);
      const want = ex1.sampled + Math.ceil(expected(js.events.slice(0, b)).passed / SAMPLE) - ex1.sampled;
      await page.evaluate(([a, b]) => __fws.replay(a, b), [a, b]);
      await page.waitForFunction((n) => app.fh.counts.analysed === n && !app.fh.queue.length, want, { timeout: 600_000 });
      await page.waitForTimeout(700);
      series.push(await page.evaluate(() => ({ a: app.fh.counts.analysed, size: app.fh.groups.size,
        ms: app.fh.view?.find((g) => g.label === "company")?.entities.find((e) => e.key === "microsoft")?.count ?? 0 })));
    }
    await page.waitForTimeout(700);
    const c2 = await page.evaluate(() => ({ counts: { ...app.fh.counts }, size: app.fh.groups.size, fifo: app.fh.fifo.length, renders: app.fh.renders,
      sinceStart: performance.now() - app.fh.t0, msInGroups: app.fh.groups.g.get("company")?.has("microsoft") ?? false,
      view: app.fh.view.map((g) => ({ label: g.label, entities: g.entities.map((e) => [e.name, e.count]) })),
      domMicrosoft: !!document.querySelector('#fh-groups button[data-key="microsoft"]'), maxCount: Math.max(0, ...app.fh.view.flatMap((g) => g.entities.map((e) => e.count))) }));
    res.c2 = c2; res.series = series;
    const k2 = c2.counts;
    check("c. full fixture counters match (sampled exactly 1 in 5 of passed)", k2.events === exAll.events && k2.posts === exAll.posts && k2.passed === exAll.passed
      && k2.sampled === exAll.sampled && k2.analysed === exAll.sampled, { got: k2, want: exAll });
    check(`c. window of ${WINDOW}: holds exactly ${WINDOW} posts, no count above it`, c2.size === WINDOW && c2.fifo === WINDOW && c2.maxCount <= WINDOW, { size: c2.size, fifo: c2.fifo, maxCount: c2.maxCount });
    const peak = Math.max(...series.map((s) => s.ms));
    const downs = series.filter((s, i) => i > 0 && s.ms < series[i - 1].ms).length, wentDown = downs >= 2 && series.every((s, i) => i === 0 || s.ms <= series[i - 1].ms);
    check("c. Microsoft's count goes down as its posts leave, then it disappears", peak >= 3 && wentDown && !c2.msInGroups && !c2.domMicrosoft, { peak, downs, series: series.map((s) => [s.a, s.ms, s.size]) });
    check("c. phase-2 entities in the view", c2.view.some((g) => g.entities.length > 0), c2.view);
    check("c. re-rendered at most twice a second", c2.renders <= c2.sinceStart / 500 + 2, { renders: c2.renders, ms: Math.round(c2.sinceStart) });

    // a burst past the queue cap: the oldest are dropped and counted
    const burst = await page.evaluate(() => {
      const f = app.fh, before = { ...f.counts };
      __fws.burst(12_000, 5e15);
      return { queue: f.queue.length, dropped: f.counts.dropped - before.dropped, sampled: f.counts.sampled - before.sampled, passed: f.counts.passed - before.passed };
    });
    check("c. queue capped at 2,000, oldest dropped and counted", burst.passed === 12_000 && burst.queue === 2000 && burst.dropped === burst.sampled - 2000
      && burst.sampled === Math.ceil((exAll.passed + 12_000) / SAMPLE) - Math.ceil(exAll.passed / SAMPLE), burst);

    // Stop closes the socket; nothing reconnects
    await page.click("#fh-stop");
    await page.waitForTimeout(1500);
    const stop = await page.evaluate(() => ({ sockets: __fws.sockets.length, ready: __fws.sockets[0].readyState, closed: __fws.closed, running: app.fh.running,
      conn: app.fh.conn.state, stats: document.getElementById("fh-stats").textContent, queue: app.fh.queue.length, startShown: !document.getElementById("fh-start").hidden }));
    check("c. Stop closes the socket, no reconnect, says stopped", stop.sockets === 1 && stop.ready === 3 && stop.closed === 1 && !stop.running && stop.conn === "stopped"
      && /stopped/.test(stop.stats) && stop.startShown, stop);

    // phone width, then leaving the tab while running closes the socket
    await page.click("#fh-start");
    await page.waitForFunction("__fws.sockets.length === 2 && __fws.sockets[1].readyState === 1", null, { timeout: 5000 });
    await page.evaluate((n) => __fws.replay(0, n), js.phase2_from);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(1200);
    check("c. no horizontal scroll at 390 px (Firehose tab)", (await noOverflow(page)) <= 0);
    await page.screenshot({ path: resolve(OUT, "page_firehose_phone.png"), fullPage: true });
    await page.click("#tab-text");
    await page.waitForTimeout(500);
    const sw = await page.evaluate(() => ({ ready: __fws.sockets[1].readyState, closed: __fws.closed, running: app.fh.running, status: document.getElementById("status-text").textContent }));
    check("c. switching tabs stops the firehose and says so", sw.ready === 3 && sw.closed === 2 && !sw.running && /firehose was stopped/i.test(sw.status), sw);
    res.wallS = (Date.now() - tRun0) / 1000;
    await ctx.close();
  }
} catch (e) {
  res.fatal = String(e.stack || e);
  console.log("FATAL " + res.fatal);
} finally {
  await browser?.close();
  server.kill();
}
check("no page errors", res.errors.length === 0, res.errors.slice(0, 5));
check("no requests off 127.0.0.1 except the stubs", res.offsite.length === 0, res.offsite);
res.pass = !res.fatal && Object.values(res.checks).every((c) => c.ok);
writeFileSync(resolve(OUT, "page_smoke2.json"), JSON.stringify(res, null, 1));
console.log(res.pass ? "PASS" : "FAIL");
process.exit(res.pass ? 0 : 1);
