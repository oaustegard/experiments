class A { #p = 1; static count = 0; static { A.init = 'ok'; } constructor(n) { this.n = n; A.count++; } get p() { return this.#p; } inc() { return ++this.#p; } static make() { return new A(0); } toString() { return `A(${this.n})`; } #priv() { return 'priv'; } callPriv() { return this.#priv(); } static #sp = 's'; static getSp() { return A.#sp; } }
class B extends A { constructor() { super(2); } inc() { return super.inc() * 10; } get [Symbol.toStringTag]() { return 'BB'; } }
const b = new B(); console.log(b.inc(), b.p, A.count, String(b), Object.prototype.toString.call(b), A.init, b.callPriv(), A.getSp(), A.make().n);
console.log(b instanceof A, Object.getPrototypeOf(B) === A, typeof A, A.name, B.length);
try { A(); } catch (e) { console.log(e.name); }
const C = class Named { who() { return Named.name; } }; console.log(new C().who(), C.name);
