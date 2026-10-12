function Show() { return this === undefined ? 'undef' : typeof this; } console.log(Show(), Show.call(5), Show.call('s'));
'use strict';
const o = { n: 1, reg() { return this.n; }, arrow: () => typeof this, nested() { return [1].map(() => this.n); } };
console.log(o.reg(), o.arrow(), o.nested(), (0, o.reg)?.call({ n: 7 }));
function add(a, b, c) { return a + b + c; } const p = add.bind(null, 1, 2); console.log(p(3), p.name, p.length, add.length);
console.log((function () { return arguments.length; })(1, 2, 3), (() => 1).name, (function foo() {}).name, new Function('a', 'b', 'return a*b')(6, 7));
console.log(add.apply(null, [1, 2, 3]), Function.prototype.call.call(add, null, 4, 5, 6), add.toString().slice(0, 20));
