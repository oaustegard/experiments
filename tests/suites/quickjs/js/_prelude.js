// needs `qjs --std` (globals os, std)
globalThis.setTimeout = (f, ms, ...a) => os.setTimeout(() => f(...a), ms);
globalThis.clearTimeout = os.clearTimeout;
globalThis.setInterval = (f, ms, ...a) => { const h = { t: null }; const tick = () => { h.t = os.setTimeout(() => { tick(); f(...a); }, ms); }; tick(); return h; };
globalThis.clearInterval = h => os.clearTimeout(h.t);
globalThis.queueMicrotask = f => Promise.resolve().then(f);
