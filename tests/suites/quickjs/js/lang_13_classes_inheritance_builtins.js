class Stack extends Array { peek() { return this[this.length - 1]; } } const s = Stack.from([1, 2, 3]); s.push(4);
console.log(s.peek(), s.length, s instanceof Array, Array.isArray(s), s.map(x => x * 2) instanceof Stack);
class MyMap extends Map { getOr(k, d) { return this.has(k) ? this.get(k) : d; } } const m = new MyMap([[1, 'a']]); console.log(m.getOr(1), m.getOr(2, 'z'), m.size);
class E2 extends RangeError {} console.log(new E2('x') instanceof RangeError, new E2('x').name);
class P extends Promise {} console.log(P.resolve(1) instanceof P);
function Old(x) { this.x = x; } Old.prototype.get = function () { return this.x; }; class New extends Old { get2() { return this.get() * 2; } } console.log(new New(21).get2());
