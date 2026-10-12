function f(x) { switch (typeof x) { case 'number': if (x > 0) { return 'pos'; } case 'string': return 'num-or-str'; default: return 'other'; case 'boolean': return 'bool'; } } console.log(f(1), f(-1), f('s'), f(true), f(null));
function g(x) { switch (x) { case 1: case 2: return 'low'; case NaN: return 'nan'; case '1': return 'str1'; default: return 'dflt'; } } console.log(g(1), g(2), g('1'), g(NaN), g(3));
a: for (var i = 0; i < 3; i++) { b: for (var j = 0; j < 3; j++) { if (j === 1) continue a; if (i === 2) break a; console.log(i, j); } } console.log(i, j);
var r = []; x: { r.push(1); y: { r.push(2); break x; } r.push(3); } console.log(r);
let n = 0; while (true) { if (++n > 5) break; else continue; } do n--; while (n > 3); console.log(n);
for (var k = 0, m = 10; k < m; k += 3, m -= 3); console.log(k, m);
