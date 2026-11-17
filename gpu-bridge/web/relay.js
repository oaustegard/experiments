// relay.js: send a page's logs to the GPU bridge repo, so a Claude session can read what happened on a device.
//
//   <script src="https://austegard.com/experiments/gpu-bridge/web/relay.js"></script>
//   Relay.start({ page: "tiny-voca-tuner", build: "2026-11-17-x" });   // as early as possible
//   Relay.log("model ready", { ms: 812 });                               // anything else worth knowing
//
// It does nothing unless both hold:
//   - this browser has the GPU Bridge page's token (localStorage gpubridge:token on this origin, saved when the
//     bridge page was connected here). Only Oskar's devices have it, so other visitors never send anything;
//   - logging was switched on for this page: ?log=1 turns it on and remembers that per page path, ?log=0 turns
//     it off. A badge in the corner shows that it is on, how many records went up, and a Stop link.
//
// What it records: console.log/info/warn/error, uncaught errors, unhandled promise rejections and Relay.log()
// calls, each with the time since the page started. The first line of each file describes the device: user
// agent, screen, memory hints, WebGPU, cross-origin isolation, the page URL and build.
//
// Where it goes: logs/<page>/<YYYY-MM-DD>/<HHMMSS>-<device>-<session>.jsonl on the bridge repo's main branch,
// one file per page load. Each upload is a commit with the message "relay: <page> <session> (<n> records)",
// rewriting that one file with everything so far. Uploads happen 2 s after a warning or error, every 15 s while
// there is something new, and when the page is hidden.
(function (root) {
  "use strict";
  const API = "https://api.github.com";
  const read = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const write = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} };
  const token = read("gpubridge:token");
  const repo = read("gpubridge:repo") || "oaustegard/gpu-bridge-comms";
  const optKey = "relay:on:" + location.pathname;
  const asked = new URLSearchParams(location.search).get("log");
  if (asked === "1") write(optKey, "1");
  if (asked === "0") write(optKey, null);
  const wanted = read(optKey) === "1";
  const t0 = performance.now();
  const session = Math.random().toString(36).slice(2, 8);
  const records = [];
  let page = "page", build = "", path = null, sha = null, sent = 0, timer = 0, busy = false, again = false, badge = null;
  const active = !!(token && wanted);
  const reason = !wanted ? "not switched on (add ?log=1)" : !token ? "no GPU Bridge token in this browser" : "on";

  function device() {
    const ua = navigator.userAgent;
    if (/iPhone/.test(ua)) return "iPhone";
    if (/iPad/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)) return "iPad";
    if (/Android/.test(ua)) return "Android";
    if (/Mac/.test(ua)) return "Mac";
    if (/Windows/.test(ua)) return "Windows";
    return "Linux";
  }
  const text = (v) => {
    if (v instanceof Error) return `${v.name}: ${v.message}${v.stack ? "\n" + v.stack.split("\n").slice(0, 6).join("\n") : ""}`;
    if (typeof v === "string") return v;
    try { return JSON.stringify(v); } catch (e) { return String(v); }
  };
  function add(level, message, attrs) {
    records.push({ t: new Date().toISOString(), ms: Math.round(performance.now() - t0), level, msg: String(message).slice(0, 4000),
                   ...(attrs && Object.keys(attrs).length ? { attrs } : {}) });
    if (!active) return;
    if (level === "error" || level === "warn") schedule(2000); else schedule(15000);
  }
  function schedule(ms) {
    if (timer && ms >= 15000) return;
    clearTimeout(timer);
    timer = setTimeout(() => { timer = 0; flush(); }, ms);
  }
  function b64(s) {
    const bytes = new TextEncoder().encode(s);
    let out = "";
    for (let i = 0; i < bytes.length; i += 0x8000) out += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(out);
  }
  async function flush({ keepalive = false } = {}) {
    if (!active || !path || records.length === sent) return;
    if (busy) { again = true; return; }
    busy = true;
    const count = records.length;
    const header = { relay: 1, page, build, session, device: device(), url: location.href, ua: navigator.userAgent,
      screen: [screen.width, screen.height, devicePixelRatio], deviceMemoryGB: navigator.deviceMemory || null,
      cores: navigator.hardwareConcurrency || null, webgpu: !!navigator.gpu, crossOriginIsolated: !!root.crossOriginIsolated,
      started: new Date(Date.now() - (performance.now() - t0)).toISOString() };
    const body = [header, ...records.slice(0, count)].map((r) => JSON.stringify(r)).join("\n") + "\n";
    try {
      const res = await fetch(`${API}/repos/${repo}/contents/${path}`, {
        method: "PUT", keepalive,
        headers: { Authorization: "Bearer " + token, Accept: "application/vnd.github+json", "Content-Type": "application/json" },
        body: JSON.stringify({ message: `relay: ${page} ${session} (${count} records)`, content: b64(body), branch: "main",
                               ...(sha ? { sha } : {}) }),
      });
      if (res.ok) {
        sha = (await res.json()).content.sha;
        sent = count;
        show(`logging to ${repo.split("/")[1]}: ${sent} sent`);
      } else {
        show(`logging failed: HTTP ${res.status}`, true);
        if (res.status === 409 || res.status === 422) sha = null;   // our sha went stale; next try re-reads it
        if (!sha && res.status !== 401 && res.status !== 403) {
          const head = await fetch(`${API}/repos/${repo}/contents/${path}?ref=main`, { headers: { Authorization: "Bearer " + token } });
          if (head.ok) sha = (await head.json()).sha;
        }
      }
    } catch (e) {
      show("logging failed: " + e.message, true);
    } finally {
      busy = false;
      if (again) { again = false; schedule(2000); }
    }
  }
  function show(message, bad = false) {
    if (!badge) return;
    badge.querySelector("span").textContent = message;
    badge.style.borderColor = bad ? "#c0392b" : "#6a3fb5";
  }
  function mountBadge() {
    if (!wanted || badge || !document.body) return;
    badge = document.createElement("div");
    badge.setAttribute("role", "status");
    badge.style.cssText = "position:fixed;z-index:2147483647;right:max(8px,env(safe-area-inset-right));bottom:max(8px,env(safe-area-inset-bottom));" +
      "font:12px/1.3 system-ui,sans-serif;background:#fff;color:#1a1d24;border:2px solid #6a3fb5;border-radius:999px;padding:4px 10px;" +
      "box-shadow:0 1px 4px rgba(0,0,0,.2);display:flex;gap:8px;align-items:center";
    badge.innerHTML = '<b style="color:#6a3fb5">●</b><span>logging on</span><a href="#" style="color:#6a3fb5">stop</a>';
    if (!token) {
      // Asked for, but this browser has no bridge token: say so instead of staying silent.
      badge.querySelector("span").textContent = "logging asked for, but this browser has no GPU Bridge token: open the GPU Bridge page here and connect once";
      badge.style.borderColor = "#c0392b";
      badge.style.borderRadius = "10px";
      badge.style.maxWidth = "min(92vw, 420px)";
    }
    badge.querySelector("a").addEventListener("click", (e) => {
      e.preventDefault(); write(optKey, null); flush(); badge.remove(); badge = null;
    });
    document.body.appendChild(badge);
  }

  // Capture console output and errors from the moment this script runs.
  for (const level of ["log", "info", "warn", "error"]) {
    const original = console[level].bind(console);
    console[level] = (...args) => { add(level === "log" ? "info" : level, args.map(text).join(" ")); original(...args); };
  }
  root.addEventListener("error", (e) => add("error", e.error ? text(e.error) : e.message,
    { source: (e.filename || "").split("/").pop(), line: e.lineno || undefined }));
  root.addEventListener("unhandledrejection", (e) => add("error", "unhandled rejection: " + text(e.reason)));
  root.addEventListener("pagehide", () => { add("info", "page hidden"); flush({ keepalive: true }); });

  root.Relay = {
    active, reason, session,
    start(options = {}) {
      page = String(options.page || location.pathname.split("/").filter(Boolean).slice(-2).join("-") || "page").replace(/[^\w.-]+/g, "-");
      build = String(options.build || "");
      const d = new Date();
      const stamp = d.toISOString().slice(11, 19).replace(/:/g, "");
      path = `logs/${page}/${d.toISOString().slice(0, 10)}/${stamp}-${device()}-${session}.jsonl`;
      add("info", "relay started", { page, build });
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountBadge); else mountBadge();
      if (active) schedule(1000);
      return this;
    },
    log(message, attrs) { add("info", message, attrs); },
    warn(message, attrs) { add("warn", message, attrs); },
    error(message, attrs) { add("error", message, attrs); },
    flush: () => flush(),
    records,
  };
})(typeof window !== "undefined" ? window : self);
