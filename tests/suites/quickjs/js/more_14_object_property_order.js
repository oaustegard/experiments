const o = { z: 1, 10: 'a', b: 2, 2: 'b', [Symbol('s')]: 3, '-1': 'neg', '01': 'lead', 4294967294: 'maxidx', 4294967295: 'notidx' };
console.log(Object.keys(o), JSON.stringify(o), Reflect.ownKeys(o).length); for (const k in o) process_k(k); function process_k(k) { console.log('key', k); }
o.a = 0; delete o.z; o.z = 9; console.log(Object.keys(o).join()); console.log(Object.entries({ b: 1, a: 2 }).sort().join(';'));
class P { constructor() { this.b = 1; this.a = 2; } } P.prototype.c = 3; console.log(Object.keys(new P()), (() => { const r = []; for (const k in new P()) r.push(k); return r; })());
