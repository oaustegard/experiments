let deep = {}; let cur = deep; for (let i = 0; i < 50; i++) { cur.n = { i }; cur = cur.n; } const s = JSON.stringify(deep); console.log(s.length, JSON.parse(s).n.n.n.i);
console.log(JSON.stringify({ 'ключ': 'значение', e: '😀', ctl: '\u0000\u001f\u007f', quote: '"\\/', big: 1e300, small: 1e-300, arr: [[]], o: {} }, null, '--'));
console.log(JSON.stringify('\ud83d'), JSON.stringify('\ud83d\ude00'), JSON.parse('"\\ud83d\\ude00"'), JSON.parse('{"__proto__": 1}').__proto__, Object.keys(JSON.parse('{"__proto__": {"x": 1}}')));
console.log(JSON.parse('[1, 2, 3]', function (k, v) { return Array.isArray(v) ? v.length : v * 2; }), JSON.stringify({ a: 1, b: 2, c: 3 }, ['c', 'a']), JSON.stringify([1, [2, [3]]], null, 1).split('\n').length);
