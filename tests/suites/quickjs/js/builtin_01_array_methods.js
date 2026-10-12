const a = [5, 1, 4, 2, 3];
console.log(a.slice(1, -1), a.splice(1, 2, 'x', 'y', 'z'), a, a.indexOf('y'), a.includes(NaN), [NaN].includes(NaN), [NaN].indexOf(NaN));
console.log([1, [2, [3, [4]]]].flat(), [1, [2, [3, [4]]]].flat(Infinity), [1, 2].flatMap(x => [x, x * 2]), Array.of(7, 8), Array(3).fill(0), [1, 2, 3].at(-1));
console.log([3, 1, 2].toSorted(), [1, 2, 3].toReversed(), [1, 2, 3].with(1, 9), [1, 2, 3, 4].toSpliced(1, 2), [1, 2, 3].findLast(x => x < 3), [1, 2, 3].findLastIndex(x => x > 5));
console.log([1, 2, 3, 4].reduce((s, x) => s + x), [1, 2, 3].reduceRight((s, x) => s + x, ''), [1, 2, 3].every(x => x > 0), [1, 2, 3].some(x => x > 2), [[1, 2], [3]].join(';'), [null, undefined, 1].join('-'));
console.log(['a', 'b'].entries().next(), [...['a', 'b'].keys()], Array.from('abc', c => c.toUpperCase()), Array.from(new Map([[1, 2]])), [1, 2, 3].copyWithin(0, 1), [1, 2, 3].lastIndexOf(3));
const objs = [{ k: 'b', i: 0 }, { k: 'a', i: 1 }, { k: 'b', i: 2 }, { k: 'a', i: 3 }]; console.log(objs.sort((x, y) => x.k < y.k ? -1 : x.k > y.k ? 1 : 0).map(o => o.i).join());
console.log(Object.groupBy ? Object.keys(Object.groupBy([1, 2, 3, 4], x => x % 2 ? 'odd' : 'even')) : 'nogroupby');
