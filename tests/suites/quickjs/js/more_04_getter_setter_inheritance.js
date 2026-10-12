class A { get v() { return 'A' + this._v; } set v(x) { this._v = x; } static get s() { return 'sA'; } }
class B extends A { get v() { return 'B' + super.v; } set v(x) { super.v = x * 2; } static get s() { return 'sB' + super.s; } }
const b = new B(); b.v = 5; console.log(b.v, b._v, B.s, Object.getOwnPropertyNames(B.prototype), Object.getOwnPropertyDescriptor(B.prototype, 'v').set !== undefined);
const o = { __proto__: { get g() { return this.k; } }, k: 7 }; console.log(o.g, o.__proto__ !== undefined, '__proto__' in o);
const p = {}; p.__defineGetter__('z', () => 99); console.log(p.z, p.__lookupGetter__('z') !== undefined);
