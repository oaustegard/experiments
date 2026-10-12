function* a() { try { yield 1; yield 2; } finally { yield 'cleanup'; console.log('after cleanup'); } }
const it = a(); console.log(it.next(), it.return('r'), it.next(), it.next());
function* b() { const x = yield* (function* () { const y = yield 'in'; return y * 2; })(); yield x; } const g = b(); console.log(g.next(), g.next(21), g.next());
const obj = { *[Symbol.iterator]() { yield* [1, 2, 3]; } }; console.log([...obj], Array.from(obj, x => x * x), new Set(obj).size);
function* inf() { let i = 0; while (true) yield i++; } const take = (n, it) => { const r = []; for (const v of it) { if (r.length >= n) break; r.push(v); } return r; }; console.log(take(5, inf()));
