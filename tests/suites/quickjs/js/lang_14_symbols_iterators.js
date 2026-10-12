const it = { from: 1, to: 4, [Symbol.iterator]() { let c = this.from, t = this.to; return { next: () => c <= t ? { value: c++, done: false } : { value: undefined, done: true }, return() { console.log('closed'); return {}; } }; } };
console.log([...it], Array.from(it), Math.max(...it));
for (const x of it) { if (x > 2) break; }
const [first] = it; console.log(first);
const s1 = Symbol('d'), s2 = Symbol.for('g'); console.log(s1.toString(), s1.description, Symbol.keyFor(s2), Symbol.for('g') === s2, typeof Symbol.iterator);
const o = { [s1]: 1, a: 2 }; console.log(Object.keys(o), Object.getOwnPropertySymbols(o).length, JSON.stringify(o));
console.log(Object(1n) instanceof BigInt, `${{ [Symbol.toPrimitive](h) { return h; } }}`, +{ [Symbol.toPrimitive](h) { return h == 'number' ? 42 : 0; } });
