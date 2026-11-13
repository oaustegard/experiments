// Cross-origin isolation on a host that cannot send headers (GitHub Pages): a service worker re-serves every
// response with COOP/COEP, so SharedArrayBuffer and onnxruntime-web's wasm threads work. Without isolation the
// page runs on one thread, about 2.8x slower here (RESULTS.md). The pattern is gzuidhof's coi-serviceworker,
// reduced to what this page needs. Every cross-origin fetch the page makes (jsdelivr, the R2 model) uses CORS,
// which require-corp accepts. Load it as a classic script in <head>; ?nocoi turns it off.
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    const req = e.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
    e.respondWith(fetch(req).then((res) => {
      if (res.status === 0) return res; // opaque: nothing to add headers to
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Resource-Policy", "cross-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }));
  });
} else if (!window.crossOriginIsolated && window.isSecureContext && "serviceWorker" in navigator
           && !new URLSearchParams(location.search).has("nocoi")) {
  const once = "coi-reloaded";
  navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
    navigator.serviceWorker.ready.then(() => {
      // Reload once so the page itself comes through the worker; never loop if isolation still fails.
      try { if (sessionStorage.getItem(once)) return; sessionStorage.setItem(once, "1"); } catch { return; }
      location.reload();
    });
  });
}
