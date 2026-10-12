const [a, , b = 5, ...rest] = [1, 2, undefined, 4, 5]; console.log(a, b, rest);
const { x, y: { z = 9, ...others } = {}, ...more } = { x: 1, y: { w: 2, v: 3 }, q: 4, r: 5 };
console.log(x, z, others, more);
function f({ p = 1, q } = {}, [r, s] = [7, 8]) { return [p, q, r, s]; } console.log(f(), f({ q: 2 }, [3]));
let m = 1, n = 2; [m, n] = [n, m]; console.log(m, n);
for (const [k, v] of Object.entries({ a: 1, b: 2 })) console.log(k, v);
const { length } = 'hello'; console.log(length);
