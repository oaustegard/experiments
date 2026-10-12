const a = [1, 2, 3]; a.length = 1; console.log(a, a.length); a[5] = 6; console.log(a, a.length, Object.keys(a)); delete a[0]; console.log(a, 0 in a);
console.log([].concat(1, [2], [[3]]), [1, 2, 3].reverse(), Array.isArray(Array.prototype), [10, 1, 5].sort().join(), ['a', 'B', 'c'].sort().join(), [true, false].sort().join());
console.log(new Array(3).join('-'), new Array('3').length, Array.apply(null, { length: 2 }), [...new Array(2)], Array.from({ length: 2, 0: 'x', 1: 'y' }));
const big = []; big[2 ** 32 - 2] = 1; console.log(big.length); try { big.length = -1; } catch (e) { console.log(e.name); }
console.log([1, 2, 3].map(parseInt), ['1', '2', '3'].map(Number), [0, 1, 2].filter(Boolean), [1, 2, 3].indexOf(2, -1), String([1, [2, 3]]), [] == false, [0] == false);
