class LRU { constructor(n) { this.n = n; this.m = new Map(); } get(k) { if (!this.m.has(k)) return -1; const v = this.m.get(k); this.m.delete(k); this.m.set(k, v); return v; } put(k, v) { this.m.delete(k); this.m.set(k, v); if (this.m.size > this.n) this.m.delete(this.m.keys().next().value); } }
const c = new LRU(2); c.put(1, 1); c.put(2, 2); console.log(c.get(1)); c.put(3, 3); console.log(c.get(2), c.get(3)); c.put(4, 4); console.log(c.get(1), c.get(3), c.get(4));
const trie = {}; for (const w of ['car', 'card', 'care', 'cat', 'dog', 'do']) { let n = trie; for (const ch of w) n = n[ch] ||= {}; n.$ = true; }
const words = (n, p = '') => Object.entries(n).flatMap(([k, v]) => k === '$' ? [p] : words(v, p + k)); console.log(words(trie).sort().join(), words(trie.c.a).length);
