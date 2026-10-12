# JavaScript test programs for the quickjs suite.  Each entry: (slug, js).
# tests/scripts/gen_quickjs_scripts.py writes tests/suites/quickjs/js/<category>_<nn>_<slug>.js.
# Deterministic only: no Math.random, Date.now, or engine-specific stack text.

SCRIPTS = {}

SCRIPTS["lang"] = [
("var_let_const_scope", """
var a = 1; { var a = 2; } console.log(a);
let b = 1; { let b = 2; console.log(b); } console.log(b);
console.log(typeof hoisted, hoisted2()); var hoisted = 1; function hoisted2() { return 'f'; }
try { console.log(tdz); let tdz = 1; } catch (e) { console.log(e.name, e.message); }
const c = [1]; c.push(2); console.log(c); try { c = 1; } catch (e) { console.log(e.name); }
for (var i = 0; i < 3; i++) setTimeout(() => {}, 0); console.log(i);
var fs = []; for (let j = 0; j < 3; j++) fs.push(() => j); console.log(fs.map(f => f()));
"""),
("closures_counters", """
function counter() { let n = 0; return { inc: () => ++n, dec: () => --n, get value() { return n; } }; }
const c1 = counter(), c2 = counter(); c1.inc(); c1.inc(); c2.dec();
console.log(c1.value, c2.value);
const adders = [1, 2, 3].map(x => y => x + y); console.log(adders.map(f => f(10)));
function memo(f) { const m = new Map(); return n => m.has(n) ? m.get(n) : (m.set(n, f(n)), m.get(n)); }
const fib = memo(n => n < 2 ? n : fib(n - 1) + fib(n - 2)); console.log(fib(50));
"""),
("destructuring", """
const [a, , b = 5, ...rest] = [1, 2, undefined, 4, 5]; console.log(a, b, rest);
const { x, y: { z = 9, ...others } = {}, ...more } = { x: 1, y: { w: 2, v: 3 }, q: 4, r: 5 };
console.log(x, z, others, more);
function f({ p = 1, q } = {}, [r, s] = [7, 8]) { return [p, q, r, s]; } console.log(f(), f({ q: 2 }, [3]));
let m = 1, n = 2; [m, n] = [n, m]; console.log(m, n);
for (const [k, v] of Object.entries({ a: 1, b: 2 })) console.log(k, v);
const { length } = 'hello'; console.log(length);
"""),
("spread_rest", """
const a = [1, 2], b = [...a, 3, ...'hi', ...new Set([4, 4, 5])]; console.log(b);
const o = { ...{ x: 1, y: 2 }, y: 3, ...null, ...[9] }; console.log(o);
function sum(...xs) { return xs.reduce((p, c) => p + c, 0); } console.log(sum(), sum(1, 2, 3), sum(...b.filter(Number.isInteger)));
console.log(Math.max(...[3, 9, 2]), [...Array(3).keys()], Array.from({ length: 3 }, (_, i) => i * i));
"""),
("template_literals", """
const n = 5, s = 'x'; console.log(`n=${n} s=${s} ${n * 2} ${[1, 2]} ${{}} ${null} ${undefined}`);
console.log(`line1
line2`, `\\u{1F600}`.length, String.raw`a\\nb${1}`, `a${`b${`c`}`}`);
function tag(strs, ...vals) { return strs.raw.join('|') + '#' + vals.join(','); } console.log(tag`a${1}b${2}c\\n`);
"""),
("optional_chaining_nullish", """
const o = { a: { b: [1, { c: 2 }] }, f() { return this.a; }, n: null, z: 0, e: '' };
console.log(o?.a?.b?.[1]?.c, o.x?.y.z, o.n?.x, o.f?.().b.length, o.g?.(), o?.['a']?.b?.length);
console.log(o.n ?? 'd', o.z ?? 'd', o.z || 'd', o.e ?? 'd', undefined ?? null ?? 3);
let a = null; a ??= 5; let b = 0; b ||= 7; let c = 1; c &&= 9; console.log(a, b, c);
delete o?.a.b; console.log(o.a);
"""),
("operators_coercion", """
console.log(1 + '2', '3' - 1, '3' * '4', [] + [], [] + {}, 1 + null, 1 + undefined, true + true, '5' + 5 - 5);
console.log(0 == '', null == undefined, null == 0, NaN == NaN, [1] == 1, '1,2' == [1, 2], ({}) == '[object Object]');
console.log(typeof null, typeof [], typeof function () {}, typeof Symbol(), typeof 1n, typeof undefined, typeof NaN);
console.log(5 / 2 | 0, -5 >> 1, -5 >>> 28, 2 ** 10, 2 ** -1, (-2) ** 2, 7 % -3, -7 % 3, 1 / 0, -1 / 0, 0 / 0, 0 === -0, Object.is(0, -0));
console.log(~~3.7, ~5, 1 << 31, 1 << 32, 2 ** 53 + 1, 0.1 + 0.2, 9007199254740993, 1e21, 1e-7, 123456789012345680000);
"""),
("control_flow", """
outer: for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) { if (j == 2) continue outer; if (i == 3) break outer; console.log(i, j); } }
let k = 0; do { k++; } while (k < 5); console.log(k);
switch (3) { case 1: console.log('one'); case 3: console.log('three'); case 4: console.log('four'); break; default: console.log('d'); }
for (const x in { a: 1, b: 2, 3: 3, 1: 1 }) console.log(x);
for (const x of 'ab') console.log(x);
console.log([1, 2, 3].map(x => x % 2 ? 'odd' : 'even'));
l: { console.log('in'); break l; }
"""),
("exceptions", """
function f() { try { throw new TypeError('bad'); } catch (e) { return e instanceof TypeError && e.message; } finally { console.log('fin'); } }
console.log(f());
function g() { try { return 'try'; } finally { console.log('cleanup'); } } console.log(g());
function h() { for (;;) { try { break; } finally { console.log('brk'); } } return 1; } console.log(h());
try { null.x; } catch (e) { console.log(e.constructor.name, e instanceof Error); }
try { undefinedFn(); } catch (e) { console.log(e.name + ': ' + e.message); }
try { throw { code: 42 }; } catch ({ code }) { console.log(code); }
class MyErr extends Error { constructor(m) { super(m); this.name = 'MyErr'; } } try { throw new MyErr('custom'); } catch (e) { console.log(String(e), e instanceof MyErr, e instanceof Error); }
const e = new Error('outer', { cause: new Error('inner') }); console.log(e.cause.message);
try { JSON.parse('{bad'); } catch (e) { console.log(e.name); }
try { new Array(-1); } catch (e) { console.log(e.name, e.message); }
try { (function r() { r(); })(); } catch (e) { console.log(e.name); }
"""),
("functions_this_bind", """
function Show() { return this === undefined ? 'undef' : typeof this; } console.log(Show(), Show.call(5), Show.call('s'));
'use strict';
const o = { n: 1, reg() { return this.n; }, arrow: () => typeof this, nested() { return [1].map(() => this.n); } };
console.log(o.reg(), o.arrow(), o.nested(), (0, o.reg)?.call({ n: 7 }));
function add(a, b, c) { return a + b + c; } const p = add.bind(null, 1, 2); console.log(p(3), p.name, p.length, add.length);
console.log((function () { return arguments.length; })(1, 2, 3), (() => 1).name, (function foo() {}).name, new Function('a', 'b', 'return a*b')(6, 7));
console.log(add.apply(null, [1, 2, 3]), Function.prototype.call.call(add, null, 4, 5, 6), add.toString().slice(0, 20));
"""),
("getters_setters_defineProperty", """
const o = { _v: 1, get v() { return this._v * 2; }, set v(x) { this._v = x; } }; o.v = 5; console.log(o.v, Object.keys(o));
const d = {}; Object.defineProperty(d, 'hidden', { value: 42, enumerable: false }); Object.defineProperty(d, 'ro', { value: 1, writable: false, enumerable: true });
d.ro = 2; console.log(d.ro, d.hidden, Object.keys(d), JSON.stringify(d), Object.getOwnPropertyNames(d));
console.log(Object.getOwnPropertyDescriptor(o, 'v').get !== undefined, Object.getOwnPropertyDescriptors({ a: 1 }));
const f = Object.freeze({ a: 1, n: { b: 1 } }); f.a = 2; f.n.b = 2; console.log(f, Object.isFrozen(f), Object.isSealed(f));
const s = Object.seal({ a: 1 }); s.b = 2; delete s.a; s.a = 3; console.log(s);
"""),
("classes_basic", """
class A { #p = 1; static count = 0; static { A.init = 'ok'; } constructor(n) { this.n = n; A.count++; } get p() { return this.#p; } inc() { return ++this.#p; } static make() { return new A(0); } toString() { return `A(${this.n})`; } #priv() { return 'priv'; } callPriv() { return this.#priv(); } static #sp = 's'; static getSp() { return A.#sp; } }
class B extends A { constructor() { super(2); } inc() { return super.inc() * 10; } get [Symbol.toStringTag]() { return 'BB'; } }
const b = new B(); console.log(b.inc(), b.p, A.count, String(b), Object.prototype.toString.call(b), A.init, b.callPriv(), A.getSp(), A.make().n);
console.log(b instanceof A, Object.getPrototypeOf(B) === A, typeof A, A.name, B.length);
try { A(); } catch (e) { console.log(e.name); }
const C = class Named { who() { return Named.name; } }; console.log(new C().who(), C.name);
"""),
("classes_inheritance_builtins", """
class Stack extends Array { peek() { return this[this.length - 1]; } } const s = Stack.from([1, 2, 3]); s.push(4);
console.log(s.peek(), s.length, s instanceof Array, Array.isArray(s), s.map(x => x * 2) instanceof Stack);
class MyMap extends Map { getOr(k, d) { return this.has(k) ? this.get(k) : d; } } const m = new MyMap([[1, 'a']]); console.log(m.getOr(1), m.getOr(2, 'z'), m.size);
class E2 extends RangeError {} console.log(new E2('x') instanceof RangeError, new E2('x').name);
class P extends Promise {} console.log(P.resolve(1) instanceof P);
function Old(x) { this.x = x; } Old.prototype.get = function () { return this.x; }; class New extends Old { get2() { return this.get() * 2; } } console.log(new New(21).get2());
"""),
("symbols_iterators", """
const it = { from: 1, to: 4, [Symbol.iterator]() { let c = this.from, t = this.to; return { next: () => c <= t ? { value: c++, done: false } : { value: undefined, done: true }, return() { console.log('closed'); return {}; } }; } };
console.log([...it], Array.from(it), Math.max(...it));
for (const x of it) { if (x > 2) break; }
const [first] = it; console.log(first);
const s1 = Symbol('d'), s2 = Symbol.for('g'); console.log(s1.toString(), s1.description, Symbol.keyFor(s2), Symbol.for('g') === s2, typeof Symbol.iterator);
const o = { [s1]: 1, a: 2 }; console.log(Object.keys(o), Object.getOwnPropertySymbols(o).length, JSON.stringify(o));
console.log(Object(1n) instanceof BigInt, `${{ [Symbol.toPrimitive](h) { return h; } }}`, +{ [Symbol.toPrimitive](h) { return h == 'number' ? 42 : 0; } });
"""),
("generators", """
function* g() { const x = yield 1; console.log('got', x); try { yield 2; } finally { console.log('fin'); } return 3; }
const it = g(); console.log(it.next('ignored'), it.next('A'), it.return(9), it.next());
function* fibs() { let [a, b] = [0, 1]; for (;;) { yield a; [a, b] = [b, a + b]; } }
const out = []; for (const v of fibs()) { if (v > 100) break; out.push(v); } console.log(out.join());
function* inner() { yield 'a'; return 'r'; } function* outer() { const r = yield* inner(); yield r; yield* [1, 2]; } console.log([...outer()]);
const t = g(); t.next(); try { t.throw(new Error('boom')); } catch (e) { console.log(e.message); }
console.log(Object.prototype.toString.call(g()), typeof g().next, g.constructor.name);
"""),
("async_ordering", """
console.log('1 sync');
setTimeout(() => console.log('8 timeout'), 0);
Promise.resolve().then(() => console.log('4 micro1')).then(() => console.log('6 micro2'));
queueMicrotask?.(() => console.log('5 qm'));
(async () => { console.log('2 async start'); await null; console.log('5b after await'); await new Promise(r => setTimeout(r, 1)); console.log('9 after timer'); })();
new Promise(r => { console.log('3 executor'); r(); }).then(() => console.log('5c then'));
console.log('3b sync end');
"""),
("promises_combinators", """
const d = (v, ms, fail) => new Promise((res, rej) => setTimeout(() => fail ? rej(new Error(v)) : res(v), ms));
(async () => {
  console.log(await Promise.all([d('a', 3), d('b', 1), 5]));
  console.log(await Promise.race([d('slow', 5), d('fast', 1)]));
  console.log(await Promise.allSettled([d('ok', 1), d('bad', 1, true)]).then(r => r.map(x => x.status + ':' + (x.value || x.reason.message))));
  console.log(await Promise.any([d('x', 1, true), d('y', 2)]));
  try { await Promise.any([d('x', 1, true)]); } catch (e) { console.log(e.constructor.name, e.errors.length); }
  try { await Promise.all([d('z', 1, true), d('w', 2)]); } catch (e) { console.log('caught', e.message); }
  const { promise, resolve } = Promise.withResolvers ? Promise.withResolvers() : { promise: Promise.resolve('wr'), resolve() {} }; resolve('wr'); console.log(await promise);
})();
"""),
("async_generators_for_await", """
async function* ag() { for (let i = 0; i < 3; i++) { await new Promise(r => setTimeout(r, 1)); yield i; } }
(async () => {
  for await (const v of ag()) console.log('v', v);
  for await (const v of [Promise.resolve('p1'), 'plain']) console.log(v);
  const it = ag(); console.log(await it.next(), await it.return('early'), await it.next());
  async function thrower() { throw new RangeError('async fail'); } try { await thrower(); } catch (e) { console.log(e.name); }
  const r = await Promise.resolve(1).finally(() => console.log('finally')); console.log(r);
})();
"""),
("proxy_reflect", """
const log = []; const p = new Proxy({ a: 1 }, { get(t, k, r) { log.push('get ' + String(k)); return Reflect.get(t, k, r); }, set(t, k, v) { log.push('set ' + k); t[k] = v * 2; return true; }, has(t, k) { return k === 'magic' || k in t; }, deleteProperty(t, k) { log.push('del ' + k); return delete t[k]; }, ownKeys(t) { return [...Reflect.ownKeys(t), 'virt']; }, getOwnPropertyDescriptor(t, k) { return k === 'virt' ? { value: 'v', enumerable: true, configurable: true } : Reflect.getOwnPropertyDescriptor(t, k); } });
p.b = 5; console.log(p.a, p.b, 'magic' in p, 'zz' in p, Object.keys(p)); delete p.a; console.log(log);
const fp = new Proxy(function (x) { return x + 1; }, { apply(t, th, args) { return t(...args) * 10; }, construct(t, args) { return { made: args }; } }); console.log(fp(1), new fp(1, 2));
console.log(Reflect.ownKeys({ b: 1, a: 2, 1: 3, [Symbol.iterator]: 0 }), Reflect.has({ x: 1 }, 'x'), Reflect.getPrototypeOf([]) === Array.prototype, Reflect.apply(Math.max, null, [1, 5, 2]), Reflect.construct(Date, [0]).getTime());
const { proxy, revoke } = Proxy.revocable({}, {}); revoke(); try { proxy.x; } catch (e) { console.log(e.name); }
"""),
("tagged_labels_misc", """
var x = 10; var f = function () { return typeof x; }; console.log(f());
console.log(void 0, (1, 2, 3), !!'0', !!'', +[], +{}, +'0x1f', +'1e3', +' 12 ', +'1_0', parseInt('08'), parseInt('0x1f'), parseFloat('3.14abc'), Number(''), Number(null), Number(undefined), Number([5]), Number('Infinity'));
console.log([10, 9, 1].sort(), ['b', 'a'].sort().reverse(), [3, 1, 2].sort((a, b) => b - a), [, 1, , 2].length, 0 in [, 1], Object.keys([, 1, , 2]));
console.log(new Date(0).toISOString(), isNaN('abc'), isFinite('12'), Number.isNaN('abc'), Number.isInteger(5.0), Number.isSafeInteger(2 ** 53), Number.EPSILON > 0, Number.MAX_SAFE_INTEGER);
console.log(encodeURIComponent('a b&c/é'), decodeURIComponent('%E2%82%AC'), encodeURI('http://x/a b?q=é'), escape('a b'), unescape('%u20AC'));
"""),
("eval_with_function_ctor", """
console.log(eval('1 + 2'), eval('var ev = 5; ev * 2'), typeof ev, (0, eval)('typeof ev'));
const g = new Function('return this')(); console.log(typeof g, g === globalThis);
globalThis.gv = 7; console.log(gv, typeof window, typeof globalThis.console, typeof require, typeof std);
console.log(eval('({a:1})').a, eval('[1,2]').length, eval('(function(){ return arguments.length })(1,2,3)'));
try { eval('let let = 1'); } catch (e) { console.log(e.name); }
console.log(Function.prototype.toString.call(function a(b) { return b; }), (class Z {}).toString());
"""),
]

SCRIPTS["builtin"] = [
("array_methods", """
const a = [5, 1, 4, 2, 3];
console.log(a.slice(1, -1), a.splice(1, 2, 'x', 'y', 'z'), a, a.indexOf('y'), a.includes(NaN), [NaN].includes(NaN), [NaN].indexOf(NaN));
console.log([1, [2, [3, [4]]]].flat(), [1, [2, [3, [4]]]].flat(Infinity), [1, 2].flatMap(x => [x, x * 2]), Array.of(7, 8), Array(3).fill(0), [1, 2, 3].at(-1));
console.log([3, 1, 2].toSorted(), [1, 2, 3].toReversed(), [1, 2, 3].with(1, 9), [1, 2, 3, 4].toSpliced(1, 2), [1, 2, 3].findLast(x => x < 3), [1, 2, 3].findLastIndex(x => x > 5));
console.log([1, 2, 3, 4].reduce((s, x) => s + x), [1, 2, 3].reduceRight((s, x) => s + x, ''), [1, 2, 3].every(x => x > 0), [1, 2, 3].some(x => x > 2), [[1, 2], [3]].join(';'), [null, undefined, 1].join('-'));
console.log(['a', 'b'].entries().next(), [...['a', 'b'].keys()], Array.from('abc', c => c.toUpperCase()), Array.from(new Map([[1, 2]])), [1, 2, 3].copyWithin(0, 1), [1, 2, 3].lastIndexOf(3));
const objs = [{ k: 'b', i: 0 }, { k: 'a', i: 1 }, { k: 'b', i: 2 }, { k: 'a', i: 3 }]; console.log(objs.sort((x, y) => x.k < y.k ? -1 : x.k > y.k ? 1 : 0).map(o => o.i).join());
console.log(Object.groupBy ? Object.keys(Object.groupBy([1, 2, 3, 4], x => x % 2 ? 'odd' : 'even')) : 'nogroupby');
"""),
("array_edge_cases", """
const a = [1, 2, 3]; a.length = 1; console.log(a, a.length); a[5] = 6; console.log(a, a.length, Object.keys(a)); delete a[0]; console.log(a, 0 in a);
console.log([].concat(1, [2], [[3]]), [1, 2, 3].reverse(), Array.isArray(Array.prototype), [10, 1, 5].sort().join(), ['a', 'B', 'c'].sort().join(), [true, false].sort().join());
console.log(new Array(3).join('-'), new Array('3').length, Array.apply(null, { length: 2 }), [...new Array(2)], Array.from({ length: 2, 0: 'x', 1: 'y' }));
const big = []; big[2 ** 32 - 2] = 1; console.log(big.length); try { big.length = -1; } catch (e) { console.log(e.name); }
console.log([1, 2, 3].map(parseInt), ['1', '2', '3'].map(Number), [0, 1, 2].filter(Boolean), [1, 2, 3].indexOf(2, -1), String([1, [2, 3]]), [] == false, [0] == false);
"""),
("string_methods", """
const s = 'Hello, World';
console.log(s.length, s.charAt(4), s.charCodeAt(1), s.codePointAt(0), s[0], s.at(-1), s.indexOf('o'), s.lastIndexOf('o'), s.includes('World'), s.startsWith('Hell'), s.endsWith('d'));
console.log(s.slice(-5), s.substring(7, 3), s.substr(-5, 3), s.toUpperCase(), s.toLowerCase(), s.split(', '), s.split(''), 'a,b,,c'.split(',', 3), s.split(/(,)/));
console.log('  pad '.trim() + '|', 'ab'.padStart(5, 'xy'), 'ab'.padEnd(5), 'ab'.repeat(3), 'a-b-c'.replace('-', '+'), 'a-b-c'.replaceAll('-', '+'), 'abc'.replace(/b/, '[$&]'), 'x'.concat(1, 2));
console.log('a1b22c333'.replace(/\\d+/g, m => `<${m.length}>`), 'John Smith'.replace(/(\\w+)\\s(\\w+)/, '$2, $1'), 'abc'.localeCompare('abd'), 'b'.localeCompare('a'), 'a'.localeCompare('a'));
console.log('\\u00e9'.normalize('NFD').length, 'é'.normalize('NFD') === 'e\\u0301', '😀'.length, [...'😀'].length, '😀'.codePointAt(0), String.fromCodePoint(128512), String.fromCharCode(72, 105), 'ß'.toUpperCase(), 'İ'.toLowerCase().length);
console.log('abc'.isWellFormed?.(), 'a\\ud800'.isWellFormed?.(), 'a\\ud800'.toWellFormed?.().charCodeAt(1), 'aXbX'.split('X'), 'abc'.split('b').length, '%'.codePointAt(), 'z'.localeCompare('Z'));
"""),
("object_methods", """
const o = { b: 2, a: 1, 2: 'two', 1: 'one' };
console.log(Object.keys(o), Object.values(o), Object.entries(o), Object.assign({}, o, { c: 3 }), Object.fromEntries([['x', 1], ['y', 2]]), Object.fromEntries(new Map([[1, 2]])));
console.log(Object.getPrototypeOf(Object.create(null)), Object.create({ inherited: 1 }, { own: { value: 2, enumerable: true } }).inherited, 'inherited' in Object.create({ inherited: 1 }), Object.hasOwn({ a: 1 }, 'a'), ({}).hasOwnProperty.call({ a: 1 }, 'b'));
const proto = { greet() { return 'hi ' + this.name; } }; const x = Object.setPrototypeOf({ name: 'x' }, proto); console.log(x.greet(), proto.isPrototypeOf(x), Object.getOwnPropertyNames(Object.getPrototypeOf(x)));
console.log(Object.entries('ab'), Object.keys([5, 6]), Object.is(NaN, NaN), Object.isExtensible(Object.preventExtensions({})), JSON.stringify(Object.entries({ a: [1] })), typeof structuredClone);
console.log(String({}), String(Object.create(null, { toString: { value: () => 'custom' } })), `${[1, [2, 3]]}`, ({}).toString.call(null), ({}).toString.call([]), ({}).toString.call(new Date(0)), ({}).toString.call(/x/), ({}).toString.call(function () {}));
"""),
("json_roundtrip", """
const o = { s: 'str\\n"q"', n: 1.5, i: 1e21, neg: -0, nan: NaN, inf: Infinity, u: undefined, f() {}, sym: Symbol('x'), d: new Date(0), nested: { a: [1, { b: null }], t: true }, big: [], e: {} };
console.log(JSON.stringify(o)); console.log(JSON.stringify(o, null, 2)); console.log(JSON.stringify(o, ['s', 'n']), JSON.stringify([undefined, function () {}, Symbol()]), JSON.stringify('\\u2028\\ud800'));
console.log(JSON.stringify({ a: 1, b: [1, 2] }, (k, v) => typeof v === 'number' ? v * 2 : v, '\\t'), JSON.stringify({ toJSON() { return 'tj'; } }), JSON.stringify(null), JSON.stringify(undefined));
console.log(JSON.parse('{"a":[1,2,{"b":"c"}],"d":1e3,"e":-0.5,"f":"\\\\u00e9"}'), JSON.parse('[1,2]', (k, v) => Array.isArray(v) ? v : v + 1), JSON.parse(' "x" '), JSON.parse('1E2'));
for (const bad of ['{a:1}', "{'a':1}", '[1,]', '01', '', '{"a":1', 'undefined', '"\\n"']) { try { JSON.parse(bad); console.log('ok?', bad); } catch (e) { console.log(e.name); } }
const cyc = {}; cyc.self = cyc; try { JSON.stringify(cyc); } catch (e) { console.log(e.name); }
try { JSON.stringify(1n); } catch (e) { console.log(e.name); }
"""),
("math_functions", """
const r = x => Math.round(x * 1e10) / 1e10;
console.log(Math.abs(-3), Math.sign(-2), Math.floor(-1.5), Math.ceil(-1.5), Math.round(-1.5), Math.round(2.5), Math.trunc(-1.9), Math.max(), Math.min(), Math.max(1, NaN), Math.hypot(3, 4), Math.cbrt(27));
console.log(r(Math.sin(1)), r(Math.cos(1)), r(Math.tan(1)), r(Math.atan2(1, 2)), r(Math.exp(1)), r(Math.log(10)), r(Math.log2(10)), r(Math.log10(5)), r(Math.sinh(1)), r(Math.expm1(1e-5)), r(Math.log1p(1e-5)), r(Math.asinh(1)));
console.log(Math.pow(2, 0.5) === Math.SQRT2, Math.PI, Math.E, Math.LN2, Math.fround(5.5), Math.fround(5.05), Math.clz32(1), Math.imul(0xffffffff, 5), Math.sqrt(-1), Math.pow(0, 0), Math.pow(NaN, 0), 0.1 * 3, 4.35 * 100);
console.log((25).toString(2), (255).toString(16), (-255).toString(36), (0.5).toString(2), (1 / 3).toString(3), 1e21.toString(7).length > 0, (123.456).toFixed(1), (0.000001234).toFixed(8), (1e21).toFixed(2), (1.005).toFixed(2), (-1.5).toFixed(0));
console.log((1234.5678).toPrecision(6), (0.00001234).toPrecision(2), (1234.5678).toExponential(2), (0).toExponential(), 255..toString(16).toUpperCase(), Number('12px'), (12).toString().padStart(4, '0'), 5e-324, 1.7976931348623157e308 * 10, (-0).toString(), String(-0), `${-0}`, JSON.stringify(-0));
"""),
("number_formatting", """
for (const n of [0, 1, -1, 0.1, 1 / 3, 100, 1e6, 1e20, 1e21, 1.5e300, 1e-6, 1e-7, 123456789.123456789, 2 ** 31, 2 ** 53, -(2 ** 63), 0.000001, 4.35, 0.1 + 0.7, 5e-324, 1.7976931348623157e308, 255.255, 1.0000000000000002]) console.log(String(n), n.toFixed(2), n.toPrecision(4), n.toExponential(3));
console.log(parseFloat('1e1000'), parseFloat('-.5e-2'), parseFloat('0x10'), parseInt('  -42abc'), parseInt('z', 36), parseInt('101', 2), parseInt('9007199254740993'), parseInt(''), parseInt('1e3'), parseInt(0.0000005), (1e21 + 1).toLocaleString === undefined);
"""),
("bigint", """
const a = 2n ** 100n, b = 12345678901234567890n; console.log(a, a * b, a / 7n, a % 1000007n, -a, a - a, b >> 5n, b << 3n, b & 0xffffn, b | 1n, b ^ b, ~b);
console.log(typeof a, 1n < 2, 2n == 2, 2n === 2, 10n > 9.5, BigInt(123), BigInt('0x1f'), BigInt(Number.MAX_SAFE_INTEGER) + 2n, Number(2n ** 64n), parseInt('9'.repeat(30)), BigInt.asUintN(8, 257n), BigInt.asIntN(8, 255n), (123n).toString(16), (-5n).toString(2));
let f = 1n; for (let i = 1n; i <= 30n; i++) f *= i; console.log(f, f.toString().length, String(f).split('').reduce((s, d) => s + +d, 0));
try { 1n + 1; } catch (e) { console.log(e.name); } try { BigInt(1.5); } catch (e) { console.log(e.name); } try { 1n / 0n; } catch (e) { console.log(e.name); } try { Math.max(1n); } catch (e) { console.log(e.name); }
console.log(5n / 2n, -5n / 2n, -5n % 3n, 0n === -0n, JSON.stringify({ v: 1 }), [3n, 1n, 2n].sort().join(), [3n, 1n, 10n].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0)).join());
"""),
("map_set_weak", """
const m = new Map([[1, 'a'], ['1', 'b'], [NaN, 'n'], [{}, 'o']]); console.log(m.size, m.get(1), m.get('1'), m.get(NaN), m.has({}), [...m.keys()].length, m.delete(1), m.size);
m.set(0, 'z').set(-0, 'zz'); console.log(m.get(0), [...m.entries()].slice(-1)); m.forEach((v, k, mm) => { if (k === '1') console.log(v, mm === m); });
const s = new Set([3, 1, 3, 2, 1]); console.log([...s], s.size, s.has(2), [...new Set('hello')].join(''), new Set([1, 2]).union?.(new Set([2, 3])).size, new Set([1, 2]).isSubsetOf?.(new Set([1, 2, 3])));
const it = s.values(); s.add(9); console.log([...it]); const d = new Set([1, 2, 3]); for (const x of d) { if (x < 3) d.add(x + 10); } console.log([...d]);
const wm = new WeakMap(), k = {}; wm.set(k, 1); console.log(wm.get(k), wm.has({}), wm.delete(k), wm.has(k)); const ws = new WeakSet([k]); console.log(ws.has(k));
try { wm.set(1, 1); } catch (e) { console.log(e.name); } console.log(new WeakRef(k).deref() === k, typeof FinalizationRegistry);
console.log(Object.fromEntries(new Map([['a', 1]])), new Map(Object.entries({ x: 1 })).get('x'), Map.groupBy ? Map.groupBy([1, 2, 3], x => x % 2).get(1) : 'n/a');
"""),
("typed_arrays", """
const buf = new ArrayBuffer(16); const u8 = new Uint8Array(buf), i32 = new Int32Array(buf, 4, 2), dv = new DataView(buf);
i32[0] = -2; i32[1] = 0x01020304; console.log(u8.slice(4, 12), dv.getInt32(4, true), dv.getUint32(8, false), dv.getFloat32(4, true), buf.byteLength, i32.byteOffset, i32.length);
dv.setFloat64(8, Math.PI); console.log(dv.getFloat64(8), [...u8.slice(8)], dv.getBigInt64(8) > 0n, dv.getUint16(8), dv.getInt8(8));
console.log(new Uint8Array([255, 256, -1, 1.9]), new Int8Array([127, 128, 255]), new Uint8ClampedArray([300, -5, 1.5, 2.5]), new Float32Array([1.1])[0], new Float64Array(2), Uint16Array.of(1, 2), Uint8Array.from('123'));
const t = new Int16Array([5, 1, 4]); t.sort(); console.log(t, t.map(x => x * 2), t.reduce((a, b) => a + b), Array.from(t), t.subarray(1).length, t.fill(7, 1), t.reverse(), t.indexOf(7), t.join('|'), t.BYTES_PER_ELEMENT, Object.prototype.toString.call(t));
console.log(new BigInt64Array(2).fill(-1n), new BigUint64Array([2n ** 64n - 1n])[0], new Uint32Array(new Float32Array([1.5, 2.5, -1])), new Float32Array([16777217])[0]);
const r = new ArrayBuffer(8, { maxByteLength: 16 }); console.log(r.resizable, r.maxByteLength); r.resize?.(12); console.log(r.byteLength, buf.slice(4, 8).byteLength, ArrayBuffer.isView(u8), new Uint8Array(buf.slice(0, 2)).length);
try { new Int32Array(buf, 1); } catch (e) { console.log(e.name); }
"""),
("regexp_features", """
console.log(/(\\d{4})-(\\d{2})-(\\d{2})/.exec('on 2024-03-15!'), 'x'.match(/y/), 'aAbB'.match(/[a-z]/gi), 'abc'.search(/c/), /^\\s*$/.test('   '), /a.c/s.test('a\\nc'), /a.c/.test('a\\nc'));
const re = /(?<year>\\d{4})-(?<mon>\\d{2})/u; const m = re.exec('2024-03'); console.log(m.groups.year, m.groups.mon, '2024-03'.replace(re, '$<mon>/$<year>'));
console.log('aXbxc'.split(/x/i), [...'a1b2c3'.matchAll(/[a-z](\\d)/g)].map(m => m[1] + '@' + m.index), 'abcabc'.replace(/(?<=a)b/g, 'B'), 'abcabc'.replace(/b(?!c)/g, 'X'), 'price: $10, $20'.match(/(?<=\\$)\\d+/g));
const g = /o/g; console.log(g.test('foo'), g.lastIndex, g.test('foo'), g.lastIndex, g.test('foo'), g.lastIndex); const y = /a/y; y.lastIndex = 1; console.log(y.test('ba'), y.lastIndex);
console.log(/\\u{1F600}/u.test('😀'), /^.$/u.test('😀'), /^.$/.test('😀'), /\\p{L}+/u.exec('héllo wörld')[0], /\\p{Script=Greek}/u.test('α'), /[\\p{Lu}]/u.test('É'), /(a)|(b)/.exec('b'), /(?:x)(y)?/.exec('x'));
console.log(/a{2,3}/.exec('aaaa')[0], /a{2,3}?/.exec('aaaa')[0], /(a+)+b/.test('aaaaaaaaaaaaaaaaaaaac'), /\\bfoo\\b/.test('a foo b'), /(\\w+)\\s\\1/.test('hello hello'), /[^]/.test('\\n'), new RegExp('a+', 'gi').flags, /x/dgimsuy.flags);
"""),
("regexp_errors_misc", """
for (const p of ['(', '[a-', 'a{2,1}', '(?<n>a)(?<n>b)', '\\\\', '*a', 'a**', '(?<=a)+', '\\\\k<x>', '[b-a]']) { try { new RegExp(p, 'u'); console.log('ok', p); } catch (e) { console.log(e.name, p); } }
try { new RegExp('a', 'zz'); } catch (e) { console.log(e.name); }
console.log(String(/a\\/b/), new RegExp('a/b').source, new RegExp('\\n').source, /[/]/.source, RegExp.prototype.toString.call(/x/g), RegExp('x', 'g') + '', /a/ instanceof RegExp);
console.log('aaa'.lastIndexOf('a'), 'test'.replace(/t/g, '$$'), 'test'.replace(/(t)/g, "$1$1$`"), 'abc'.replace(/(?<x>b)/, (...a) => JSON.stringify(a.slice(-1))), 'a.b'.split('.'), 'a1b'.split(/\\d/), 'abc'.split(/(?:)/u));
"""),
("date_utc", """
const d = new Date(Date.UTC(2024, 1, 29, 13, 45, 30, 123)); console.log(d.toISOString(), d.getTime(), d.getUTCDay(), d.getUTCDate(), d.getUTCFullYear(), d.valueOf() === +d, d.toJSON(), JSON.stringify({ d }));
console.log(new Date('2024-03-15T10:20:30Z').getTime(), new Date('2024-03-15').getTime(), new Date(2024, 0, 31).getMonth(), Date.UTC(2000, 0), new Date(NaN).getTime(), String(new Date(NaN)), new Date(8.64e15 + 1).getTime(), new Date(0).toUTCString());
const e = new Date(Date.UTC(2024, 0, 31)); e.setUTCMonth(1); console.log(e.toISOString()); e.setUTCDate(0); console.log(e.toISOString()); e.setUTCHours(25, 61, 61); console.log(e.toISOString());
console.log(new Date('2024-02-30T00:00:00Z').getTime(), Date.parse('Thu, 01 Jan 1970 00:00:00 GMT'), Date.parse('2000-01-01T00:00:00.5+02:00'), Date.parse('garbage'), new Date(2020, 5).getDate(), new Date(99, 0).getFullYear(), new Date(Date.UTC(-1, 0)).toISOString(), new Date(Date.UTC(10000, 0)).toISOString());
console.log(typeof Date.now(), Date.length, new Date(2024, 0, 1, 0, 0, 0).getTimezoneOffset(), new Date(1e12).toISOString(), Object.prototype.toString.call(new Date()), new Date(2024, 11, 25) - new Date(2024, 11, 24));
"""),
("global_functions_misc", """
console.log(typeof globalThis, typeof console.log, typeof setTimeout, typeof clearTimeout, typeof setInterval, typeof queueMicrotask, typeof structuredClone, typeof TextEncoder, typeof URL, typeof atob, typeof performance);
let n = 0; const id = setInterval(() => { n++; if (n === 3) { clearInterval(id); console.log('interval done', n); } }, 1);
const t = setTimeout(() => console.log('never'), 5); clearTimeout(t); setTimeout((a, b) => console.log('args', a, b), 2, 'x', 'y');
console.log(Array.prototype.concat.call(1, 2).length, Array.prototype.slice.call('abc'), Array.prototype.map.call('ab', c => c + c), [].concat.apply([], [[1], [2]]), Math.max.apply(null, [1, 2]), Object.prototype.toString.call(arguments = 1));
console.log(Number.prototype.toString.call(5), String.prototype.trim.call(' x '), typeof Function.prototype, Function.prototype(), Object.getOwnPropertyNames(globalThis).includes('Array'));
"""),
("string_unicode_edges", """
const s = 'a😀b'; console.log(s.length, s.split(''), [...s], s.slice(1, 2).length, s.charAt(1).charCodeAt(0), s.codePointAt(1), s.codePointAt(2), s.indexOf('b'), encodeURIComponent(s), s.normalize() === s);
console.log('Ω'.toLowerCase(), 'ǆ'.toUpperCase(), 'ŉ'.toUpperCase(), 'ﬃ'.toUpperCase(), 'Σ'.toLowerCase(), 'ΑΣ'.toLowerCase(), 'i'.toLocaleUpperCase(), '\\u0130'.toLowerCase().length, 'ǅ'.toLowerCase());
console.log([...'ñ'.normalize('NFD')].map(c => c.charCodeAt(0).toString(16)), 'Å'.normalize('NFKC').charCodeAt(0).toString(16), '\\ufb01'.normalize('NFKD'), '한글'.normalize('NFD').length, 'ａｂｃ'.normalize('NFKC'));
console.log('a\\u0000b'.length, 'tab\\there'.split('\\t'), '\\x41\\u0042\\u{43}', 'a\\
b', escape('é€'), JSON.stringify('\\u007f\\u0080\\u2028'), String.raw`\\u0041`, 'abc'.localeCompare('ABC'), ['é', 'e', 'z', 'a'].sort(), ['é', 'e', 'z', 'a'].sort((a, b) => a.localeCompare(b)));
"""),
("sorting_stability_large", """
const n = 2000; const a = Array.from({ length: n }, (_, i) => ({ k: (i * 7919) % 101, i })); a.sort((x, y) => x.k - y.k);
let ok = true; for (let i = 1; i < n; i++) if (a[i - 1].k === a[i].k && a[i - 1].i > a[i].i) ok = false; console.log('stable', ok, a[0], a[n - 1]);
const nums = Array.from({ length: 1000 }, (_, i) => (i * 2654435761) % 1000003); nums.sort((x, y) => x - y); console.log(nums.slice(0, 5), nums[999], nums.every((v, i) => i == 0 || nums[i - 1] <= v));
const strs = nums.slice(0, 50).map(String).sort(); console.log(strs.slice(0, 5), [5, 25, 100, 1].sort(), [5, 25, 100, 1].sort((a, b) => a - b), ['b', undefined, 'a', , 'c'].sort(), [3, 2, 1].sort(() => 0));
"""),
]

SCRIPTS["program"] = [
("fib_recursive", """
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
const t = []; for (let i = 0; i <= 25; i += 5) t.push(fib(i)); console.log(t.join());
"""),
("sieve_primes", """
const N = 20000, sieve = new Uint8Array(N + 1); const primes = [];
for (let i = 2; i <= N; i++) { if (!sieve[i]) { primes.push(i); for (let j = i * i; j <= N; j += i) sieve[j] = 1; } }
console.log(primes.length, primes.slice(-3), primes.reduce((a, b) => a + b, 0));
"""),
("n_queens", """
function solve(n) { let count = 0; const cols = new Array(n).fill(false), d1 = new Array(2 * n).fill(false), d2 = new Array(2 * n).fill(false);
  (function place(r) { if (r === n) { count++; return; } for (let c = 0; c < n; c++) { if (cols[c] || d1[r + c] || d2[r - c + n]) continue; cols[c] = d1[r + c] = d2[r - c + n] = true; place(r + 1); cols[c] = d1[r + c] = d2[r - c + n] = false; } })(0); return count; }
console.log([4, 5, 6, 7, 8].map(solve).join());
"""),
("matrix_multiply", """
const n = 40; const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i * j) % 7 + 1)), B = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i + j) % 5 - 2));
const C = A.map((row, i) => B[0].map((_, j) => row.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
console.log(C[0].slice(0, 5), C[n - 1][n - 1], C.flat().reduce((a, b) => a + b, 0));
"""),
("string_builder_hash", """
let h = 5381; let s = ''; for (let i = 0; i < 3000; i++) { s += String.fromCharCode(97 + (i * 31) % 26); h = ((h << 5) + h + s.charCodeAt(i)) | 0; } console.log(s.length, h, s.slice(0, 30), s.split('a').length);
const words = 'the quick brown fox jumps over the lazy dog the end'.split(' '); const freq = {}; for (const w of words) freq[w] = (freq[w] || 0) + 1; console.log(Object.entries(freq).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3));
"""),
("json_big_roundtrip", """
const data = Array.from({ length: 500 }, (_, i) => ({ id: i, name: 'item' + i, tags: ['t' + (i % 7), 't' + (i % 3)], v: i / 7, nested: { a: [i, i * 2], ok: i % 2 === 0 } }));
const s = JSON.stringify(data); const back = JSON.parse(s); console.log(s.length, back.length, back[499].nested, JSON.stringify(back) === s, back.filter(x => x.nested.ok).length);
"""),
("tiny_interpreter", """
function tokenize(src) { return src.match(/\\d+\\.?\\d*|[-+*/()^]/g); }
function parse(tokens) { let p = 0; const peek = () => tokens[p], next = () => tokens[p++];
  function prim() { const t = next(); if (t === '(') { const v = expr(); next(); return v; } if (t === '-') return -prim(); return parseFloat(t); }
  function pow() { const b = prim(); if (peek() === '^') { next(); return b ** pow(); } return b; }
  function term() { let v = pow(); while (peek() === '*' || peek() === '/') v = next() === '*' ? v * pow() : v / pow(); return v; }
  function expr() { let v = term(); while (peek() === '+' || peek() === '-') v = next() === '+' ? v + term() : v - term(); return v; } return expr(); }
for (const e of ['1+2*3', '(1+2)*3', '2^3^2', '-4+10/4', '3.5*2-1', '((2))*((3+4))']) console.log(e, '=', parse(tokenize(e)));
"""),
("event_loop_pipeline", """
const results = []; const sleep = (ms, v) => new Promise(r => setTimeout(() => r(v), ms));
async function worker(id, jobs) { for (const j of jobs) { await sleep(j % 3, null); results.push(`w${id}:${j}`); } }
(async () => { await Promise.all([worker(1, [1, 2, 3]), worker(2, [4, 5]), worker(3, [6])]); console.log(results.length, results.sort().join()); const r = await Promise.race([sleep(20, 'slow'), sleep(1, 'fast')]); console.log(r); })();
"""),
("garbage_gc_stress", """
let keep = []; for (let i = 0; i < 20000; i++) { const o = { i, a: [i, i + 1], s: 'v' + i }; o.self = o; if (i % 100 === 0) keep.push(o); }
console.log(keep.length, keep[199].s, keep[5].self === keep[5]); const big = []; for (let i = 0; i < 300; i++) big.push(new Array(1000).fill(i)); console.log(big.length, big[299][999]);
let str = ''; for (let i = 0; i < 2000; i++) str += i; console.log(str.length); const m = new Map(); for (let i = 0; i < 5000; i++) m.set('k' + i, { i }); for (let i = 0; i < 5000; i += 2) m.delete('k' + i); console.log(m.size);
"""),
("closures_in_loops_memory", """
const fns = []; for (let i = 0; i < 1000; i++) { const big = new Array(10).fill(i); fns.push(() => big[0] + i); } console.log(fns.reduce((s, f) => s + f(), 0));
function make(n) { return n === 0 ? () => 0 : (f => () => f() + 1)(make(n - 1)); } console.log(make(100)());
"""),
]

SCRIPTS["std"] = [
("std_module_basics", """
import * as std from 'std';
import * as os from 'os';
std.printf('%d %s %5.2f %x|\\n', 42, 'str', 3.14159, 255);
console.log(std.sprintf('%05d|%-5s|%c', 42, 'ab', 65), typeof std.getenv('PATH'), std.getenv('HCC_NOPE_UNSET'), typeof os.getcwd()[0], typeof os.platform);
console.log(std.parseExtJSON('{a:1, b:[1,2,], "c":0x10, d:.5}'), std.strerror(2), std.loadFile('/nonexistent'), typeof std.evalScript);
std.puts('puts\\n'); std.out.puts('out\\n'); std.out.printf('%s\\n', 'printf'); std.err.puts('');
"""),
("std_file_io", """
import * as std from 'std';
import * as os from 'os';
const name = 'qjs_test_file.txt'; const f = std.open(name, 'w'); f.puts('line1\\nline2\\n'); f.printf('%d\\n', 3); f.close();
const g = std.open(name, 'r'); console.log(g.getline(), g.readAsString().length, g.eof(), g.tell()); g.seek(0, std.SEEK_SET); console.log(g.readAsString(5), g.getByte()); g.close();
console.log(std.loadFile(name).split('\\n'), os.stat(name)[0].size, os.readdir('.')[0].includes(name)); os.remove(name);
console.log(os.stat(name)[1] !== 0, std.open(name, 'r'), std.loadFile(name));
const [err] = [os.mkdir('qd')]; console.log(err, os.readdir('qd')[0], os.rename('qd', 'qd2'), os.remove('qd2'), os.stat('.')[0].mode !== 0);
"""),
("os_timers_and_exec", """
import * as os from 'os';
import * as std from 'std';
let n = 0; const t = os.setTimeout(() => console.log('t1'), 1); os.setTimeout(() => { console.log('t2'); n++; }, 3); os.clearTimeout(t);
console.log(typeof os.now(), os.getcwd() === os.realpath('.')[0], os.isatty(0), typeof os.pid);
const r = std.popen('echo hello; echo err >&2', 'r'); console.log(r.readAsString().trim()); console.log(r.close());
console.log(os.exec(['sh', '-c', 'exit 3']), os.exec(['true']));
const fd = os.open('/dev/null', os.O_RDWR); console.log(fd >= 0, os.write(fd, new Uint8Array(4).buffer, 0, 4), os.close(fd));
"""),
("script_args_and_exit", """
import * as std from 'std';
console.log(scriptArgs.length, typeof scriptArgs[0]);
globalThis.exitSoon = () => std.exit(7); console.log('before'); setTimeout(() => { console.log('exiting'); std.exit(7); }, 1);
"""),
("uncaught_exception_exit", """
console.log('start');
setTimeout(() => { throw new RangeError('uncaught in timer'); }, 1);
"""),
("uncaught_in_promise", """
console.log('a');
Promise.reject(new TypeError('unhandled rejection'));
console.log('b');
"""),
("module_syntax_import_export", """
import * as std from 'std';
export const x = 1; export default function f() { return 2; } export { x as y };
console.log(x, typeof f);
console.log(typeof import.meta, typeof import.meta.url, import.meta.main);
"""),
]
