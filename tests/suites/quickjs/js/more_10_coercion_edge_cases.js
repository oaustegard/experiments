const vals = [0, -0, 1, NaN, '', ' ', '0', 'a', null, undefined, [], [0], [1, 2], {}, true, false, 1n, Symbol.iterator];
for (const v of vals) { let s; try { s = [Boolean(v), typeof v === 'symbol' ? 'sym' : String(v), typeof v === 'bigint' ? Number(v) : (typeof v === 'symbol' ? 'sym' : Number(v))].join('|'); } catch (e) { s = e.name; } console.log(s); }
console.log([1] + [2], {} + 1, [] * 2, '5' * '5', null + 1, undefined + 1, true + '1', [] == ![], NaN != NaN, '' == 0, '0' == false, null >= 0, null > 0, null == 0, undefined == 0);
console.log(new Number(5) + 1, new String('a') + 'b', typeof new Boolean(false), !!new Boolean(false), new Number(1) === 1, Object(1) == 1, JSON.stringify([new Number(1), new String('s'), new Boolean(true)]));
