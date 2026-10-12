const o = { _v: 1, get v() { return this._v * 2; }, set v(x) { this._v = x; } }; o.v = 5; console.log(o.v, Object.keys(o));
const d = {}; Object.defineProperty(d, 'hidden', { value: 42, enumerable: false }); Object.defineProperty(d, 'ro', { value: 1, writable: false, enumerable: true });
d.ro = 2; console.log(d.ro, d.hidden, Object.keys(d), JSON.stringify(d), Object.getOwnPropertyNames(d));
console.log(Object.getOwnPropertyDescriptor(o, 'v').get !== undefined, Object.getOwnPropertyDescriptors({ a: 1 }));
const f = Object.freeze({ a: 1, n: { b: 1 } }); f.a = 2; f.n.b = 2; console.log(f, Object.isFrozen(f), Object.isSealed(f));
const s = Object.seal({ a: 1 }); s.b = 2; delete s.a; s.a = 3; console.log(s);
