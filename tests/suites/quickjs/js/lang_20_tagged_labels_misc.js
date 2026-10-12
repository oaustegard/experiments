var x = 10; var f = function () { return typeof x; }; console.log(f());
console.log(void 0, (1, 2, 3), !!'0', !!'', +[], +{}, +'0x1f', +'1e3', +' 12 ', +'1_0', parseInt('08'), parseInt('0x1f'), parseFloat('3.14abc'), Number(''), Number(null), Number(undefined), Number([5]), Number('Infinity'));
console.log([10, 9, 1].sort(), ['b', 'a'].sort().reverse(), [3, 1, 2].sort((a, b) => b - a), [, 1, , 2].length, 0 in [, 1], Object.keys([, 1, , 2]));
console.log(new Date(0).toISOString(), isNaN('abc'), isFinite('12'), Number.isNaN('abc'), Number.isInteger(5.0), Number.isSafeInteger(2 ** 53), Number.EPSILON > 0, Number.MAX_SAFE_INTEGER);
console.log(encodeURIComponent('a b&c/é'), decodeURIComponent('%E2%82%AC'), encodeURI('http://x/a b?q=é'), escape('a b'), unescape('%u20AC'));
