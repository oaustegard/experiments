let o = { big: new Array(1000).fill(1) }; const wr = new WeakRef(o); console.log(wr.deref() === o, typeof FinalizationRegistry, typeof gc);
const reg = new FinalizationRegistry(() => {}); reg.register(o, 'tok', o); console.log(reg.unregister(o), reg.unregister({}));
const cache = new Map(); function mk(i) { return { i, payload: 'x'.repeat(100) }; } for (let i = 0; i < 5000; i++) cache.set(i % 50, mk(i)); console.log(cache.size, cache.get(7).i);
let s = ''; for (let i = 0; i < 200; i++) { const a = []; for (let j = 0; j < 100; j++) a.push({ j, a }); s = String(a.length); } console.log(s);
