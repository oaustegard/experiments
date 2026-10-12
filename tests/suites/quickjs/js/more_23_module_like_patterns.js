const mod = (function () { let priv = 0; const api = { inc: () => ++priv, get count() { return priv; } }; return Object.freeze(api); })(); mod.inc(); mod.inc(); console.log(mod.count, Object.isFrozen(mod));
const EventEmitter = class { #h = {}; on(e, f) { (this.#h[e] ||= []).push(f); return this; } emit(e, ...a) { (this.#h[e] || []).forEach(f => f(...a)); return !!this.#h[e]; } };
const em = new EventEmitter(); em.on('x', (a, b) => console.log('x1', a, b)).on('x', a => console.log('x2', a)); console.log(em.emit('x', 1, 2), em.emit('y'));
const curry = f => function c(...a) { return a.length >= f.length ? f(...a) : (...b) => c(...a, ...b); }; console.log(curry((a, b, c) => a + b + c)(1)(2)(3), curry((a, b) => a * b)(3, 4));
const compose = (...fs) => x => fs.reduceRight((v, f) => f(v), x); console.log(compose(x => x + 1, x => x * 2)(5));
