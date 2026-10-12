console.log(typeof globalThis, typeof console.log, typeof setTimeout, typeof clearTimeout, typeof setInterval, typeof queueMicrotask, typeof structuredClone, typeof TextEncoder, typeof URL, typeof atob, typeof performance);
let n = 0; const id = setInterval(() => { n++; if (n === 3) { clearInterval(id); console.log('interval done', n); } }, 1);
const t = setTimeout(() => console.log('never'), 5); clearTimeout(t); setTimeout((a, b) => console.log('args', a, b), 2, 'x', 'y');
console.log(Array.prototype.concat.call(1, 2).length, Array.prototype.slice.call('abc'), Array.prototype.map.call('ab', c => c + c), [].concat.apply([], [[1], [2]]), Math.max.apply(null, [1, 2]), Object.prototype.toString.call(arguments = 1));
console.log(Number.prototype.toString.call(5), String.prototype.trim.call(' x '), typeof Function.prototype, Function.prototype(), Object.getOwnPropertyNames(globalThis).includes('Array'));
