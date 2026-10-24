// tuned.js: read the kernel configs Refractometer measured for this GPU. ES module, no dependencies.
//
//   import { getTuned } from 'https://austegard.com/experiments/refractometer/web/tuned.js';
//   const { C, S, R } = await getTuned(adapter, 'gemv_q4:2048x512:m16', { C: 32, S: 8, R: 8 }, { defaultLimits: true });
//
// Lookup order: a profile this browser saved (localStorage, written by index.html on the same
// origin), then the bundled profiles.json next to this file. Within each, the adapter's exact
// vendor/architecture/device key first, then any profile with the same vendor/architecture (Chrome
// often reports an empty device). No match returns `fallback` unchanged. Never throws.

const LS = 'refractometer:profile:';
const METRICS = new Set(['us', 'gbps', 'gflops', 'default_limits', 'engine_config_us', 'speedup_vs_engine', 'tail_config_us', 'speedup_vs_tail', 'variant']);

export function profileKey(info) {
  const part = (s) => (String(s ?? '').trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || '-');
  return `${part(info?.vendor)}/${part(info?.architecture)}/${part(info?.device)}`;
}

let bundled = null;
export function loadBundled(url = new URL('./profiles.json', import.meta.url)) {
  bundled ??= fetch(url).then((r) => (r.ok ? r.json() : { profiles: {} })).catch(() => ({ profiles: {} }));
  return bundled;
}

function local() {
  const out = {};
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith(LS)) { const p = JSON.parse(localStorage.getItem(k)); if (p?.best) out[k.slice(LS.length)] = p; }
    }
  } catch { /* no storage: private window, Node, sandboxed frame */ }
  return out;
}

function pick(profiles, key) {
  if (profiles[key]) return { profile: profiles[key], match: 'exact' };
  const va = key.split('/').slice(0, 2).join('/') + '/';
  const near = Object.keys(profiles).filter((k) => k.startsWith(va)).sort();
  return near.length ? { profile: profiles[near[0]], match: 'vendor/architecture', key: near[0] } : null;
}

// Explain where a config would come from: { config, source, key, match } or source 'fallback'.
export async function explainTuned(adapterOrInfo, kernelName, fallback = {}, opts = {}) {
  try {
    const info = adapterOrInfo?.info ?? adapterOrInfo;
    const key = typeof adapterOrInfo === 'string' ? adapterOrInfo : profileKey(info);
    const sources = [['localStorage', local()], ['profiles.json', (await loadBundled(opts.bundledUrl)).profiles ?? {}]];
    for (const [source, profiles] of sources) {
      const hit = pick(profiles, key);
      let entry = hit?.profile.best?.[kernelName];
      if (!entry) continue;
      if (opts.defaultLimits && entry.default_limits) entry = entry.default_limits;
      const config = { ...fallback };
      for (const [k, v] of Object.entries(entry)) if (!METRICS.has(k)) config[k] = v;
      return { config, source, key: hit.key ?? key, match: hit.match, measured: entry };
    }
  } catch { /* fall through */ }
  return { config: { ...fallback }, source: 'fallback' };
}

// The saved config for kernelName on this adapter, merged over `fallback`.
// opts.defaultLimits: true when the device was created without raised limits (onnxruntime-web's,
// for instance); returns the best config that fits WebGPU's default limits.
export async function getTuned(adapterOrInfo, kernelName, fallback = {}, opts = {}) {
  return (await explainTuned(adapterOrInfo, kernelName, fallback, opts)).config;
}
