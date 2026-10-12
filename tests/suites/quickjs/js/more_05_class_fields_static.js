class C { a = 1; b = this.a + 1; static s = 'st'; static t = C.s + '!'; ['comp' + 1] = 'c'; static #cnt = 0; static inc() { return ++C.#cnt; } #priv = 'p'; static has(o) { return #priv in o; } arrow = () => this.a; }
const c = new C(); console.log(c, C.t, C.inc(), C.inc(), C.has(c), C.has({}), c.arrow.call(null), Object.keys(c));
class D extends C { a = 100; constructor() { super(); this.z = this.b; } } console.log(new D());
