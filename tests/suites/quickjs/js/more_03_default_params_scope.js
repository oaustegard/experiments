let x = 'outer'; function f(a = x, b = () => a, c = a + '!') { var x = 'inner'; return [a, b(), c, x]; } console.log(f(), f(1, undefined, 3));
function g(a, b = a * 2, c = b + 1) { return [a, b, c]; } console.log(g(1), g(1, 5), g.length);
try { (function (a = b, b = 1) {})(); } catch (e) { console.log(e.name); }
function n() { return new.target === undefined ? 'call' : 'new'; } console.log(n(), new n() instanceof n, Reflect.construct(n, []) instanceof n);
