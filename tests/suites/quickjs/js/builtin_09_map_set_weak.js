const m = new Map([[1, 'a'], ['1', 'b'], [NaN, 'n'], [{}, 'o']]); console.log(m.size, m.get(1), m.get('1'), m.get(NaN), m.has({}), [...m.keys()].length, m.delete(1), m.size);
m.set(0, 'z').set(-0, 'zz'); console.log(m.get(0), [...m.entries()].slice(-1)); m.forEach((v, k, mm) => { if (k === '1') console.log(v, mm === m); });
const s = new Set([3, 1, 3, 2, 1]); console.log([...s], s.size, s.has(2), [...new Set('hello')].join(''), new Set([1, 2]).union?.(new Set([2, 3])).size, new Set([1, 2]).isSubsetOf?.(new Set([1, 2, 3])));
const it = s.values(); s.add(9); console.log([...it]); const d = new Set([1, 2, 3]); for (const x of d) { if (x < 3) d.add(x + 10); } console.log([...d]);
const wm = new WeakMap(), k = {}; wm.set(k, 1); console.log(wm.get(k), wm.has({}), wm.delete(k), wm.has(k)); const ws = new WeakSet([k]); console.log(ws.has(k));
try { wm.set(1, 1); } catch (e) { console.log(e.name); } console.log(new WeakRef(k).deref() === k, typeof FinalizationRegistry);
console.log(Object.fromEntries(new Map([['a', 1]])), new Map(Object.entries({ x: 1 })).get('x'), Map.groupBy ? Map.groupBy([1, 2, 3], x => x % 2).get(1) : 'n/a');
