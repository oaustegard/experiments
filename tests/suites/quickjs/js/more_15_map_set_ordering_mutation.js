const m = new Map([['a', 1], ['b', 2], ['c', 3]]); m.delete('a'); m.set('a', 4); console.log([...m.keys()]);
const seen = []; for (const [k] of m) { seen.push(k); if (k === 'b') { m.delete('c'); m.set('d', 5); } } console.log(seen, [...m.keys()]);
const s = new Set([1, 2, 3]); const out = []; for (const v of s) { out.push(v); if (v < 3) { s.delete(v); s.add(v + 10); } if (out.length > 10) break; } console.log(out, [...s]);
const big = new Map(); for (let i = 0; i < 1000; i++) big.set(i, i * i); for (let i = 0; i < 1000; i += 3) big.delete(i); console.log(big.size, [...big.keys()].slice(0, 5), big.get(999), big.get(3));
console.log(new Map([[NaN, 1]]).get(NaN), new Set([0, -0]).size, [...new Set([-0])][0] === 0 && 1 / [...new Set([-0])][0], new Map([[{}, 1], [{}, 2]]).size);
