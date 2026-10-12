function* g() { const x = yield 1; console.log('got', x); try { yield 2; } finally { console.log('fin'); } return 3; }
const it = g(); console.log(it.next('ignored'), it.next('A'), it.return(9), it.next());
function* fibs() { let [a, b] = [0, 1]; for (;;) { yield a; [a, b] = [b, a + b]; } }
const out = []; for (const v of fibs()) { if (v > 100) break; out.push(v); } console.log(out.join());
function* inner() { yield 'a'; return 'r'; } function* outer() { const r = yield* inner(); yield r; yield* [1, 2]; } console.log([...outer()]);
const t = g(); t.next(); try { t.throw(new Error('boom')); } catch (e) { console.log(e.message); }
console.log(Object.prototype.toString.call(g()), typeof g().next, g.constructor.name);
