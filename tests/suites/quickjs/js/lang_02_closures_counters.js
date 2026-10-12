function counter() { let n = 0; return { inc: () => ++n, dec: () => --n, get value() { return n; } }; }
const c1 = counter(), c2 = counter(); c1.inc(); c1.inc(); c2.dec();
console.log(c1.value, c2.value);
const adders = [1, 2, 3].map(x => y => x + y); console.log(adders.map(f => f(10)));
function memo(f) { const m = new Map(); return n => m.has(n) ? m.get(n) : (m.set(n, f(n)), m.get(n)); }
const fib = memo(n => n < 2 ? n : fib(n - 1) + fib(n - 2)); console.log(fib(50));
