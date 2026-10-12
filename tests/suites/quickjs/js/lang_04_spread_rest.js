const a = [1, 2], b = [...a, 3, ...'hi', ...new Set([4, 4, 5])]; console.log(b);
const o = { ...{ x: 1, y: 2 }, y: 3, ...null, ...[9] }; console.log(o);
function sum(...xs) { return xs.reduce((p, c) => p + c, 0); } console.log(sum(), sum(1, 2, 3), sum(...b.filter(Number.isInteger)));
console.log(Math.max(...[3, 9, 2]), [...Array(3).keys()], Array.from({ length: 3 }, (_, i) => i * i));
