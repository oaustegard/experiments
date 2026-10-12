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
  console.log(JSON.stringify(await Promise.all([d('a', 3), d('b', 1), 5])));
  console.log(await Promise.race([d('slow', 5), Promise.resolve('fast')]));
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
(async () => { await Promise.all([worker(1, [1, 2, 3]), worker(2, [4, 5]), worker(3, [6])]); console.log(results.length, results.sort().join()); const r = await Promise.race([sleep(20, 'slow'), Promise.resolve('fast')]); console.log(r); })();
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

SCRIPTS["more"] = [
("with_statement_sloppy", """
var o = { a: 1, b: 2 }; with (o) { a = 10; var c = a + b; } console.log(o, c, typeof a);
function f() { with ({ x: 5 }) { return function () { return x; }; } } console.log(f()());
"""),
("arguments_object", """
function f(a, b) { arguments[0] = 'changed'; b = 'B'; return [a, arguments[1], arguments.length, typeof arguments.callee]; } console.log(f(1, 2), f(1));
function g(a, b = 2) { arguments[0] = 'x'; return [a, arguments.length]; } console.log(g(1));
function h() { 'use strict'; return [typeof arguments.callee === 'undefined' ? 'noncallee' : 'callee']; } try { console.log(h(1)); } catch (e) { console.log(e.name); }
console.log((function () { return [].slice.call(arguments, 1); })(1, 2, 3), (function () { return Array.from(arguments).length; })(...'abcd'));
"""),
("default_params_scope", """
let x = 'outer'; function f(a = x, b = () => a, c = a + '!') { var x = 'inner'; return [a, b(), c, x]; } console.log(f(), f(1, undefined, 3));
function g(a, b = a * 2, c = b + 1) { return [a, b, c]; } console.log(g(1), g(1, 5), g.length);
try { (function (a = b, b = 1) {})(); } catch (e) { console.log(e.name); }
function n() { return new.target === undefined ? 'call' : 'new'; } console.log(n(), new n() instanceof n, Reflect.construct(n, []) instanceof n);
"""),
("getter_setter_inheritance", """
class A { get v() { return 'A' + this._v; } set v(x) { this._v = x; } static get s() { return 'sA'; } }
class B extends A { get v() { return 'B' + super.v; } set v(x) { super.v = x * 2; } static get s() { return 'sB' + super.s; } }
const b = new B(); b.v = 5; console.log(b.v, b._v, B.s, Object.getOwnPropertyNames(B.prototype), Object.getOwnPropertyDescriptor(B.prototype, 'v').set !== undefined);
const o = { __proto__: { get g() { return this.k; } }, k: 7 }; console.log(o.g, o.__proto__ !== undefined, '__proto__' in o);
const p = {}; p.__defineGetter__('z', () => 99); console.log(p.z, p.__lookupGetter__('z') !== undefined);
"""),
("class_fields_static", """
class C { a = 1; b = this.a + 1; static s = 'st'; static t = C.s + '!'; ['comp' + 1] = 'c'; static #cnt = 0; static inc() { return ++C.#cnt; } #priv = 'p'; static has(o) { return #priv in o; } arrow = () => this.a; }
const c = new C(); console.log(c, C.t, C.inc(), C.inc(), C.has(c), C.has({}), c.arrow.call(null), Object.keys(c));
class D extends C { a = 100; constructor() { super(); this.z = this.b; } } console.log(new D());
"""),
("generators_advanced", """
function* a() { try { yield 1; yield 2; } finally { yield 'cleanup'; console.log('after cleanup'); } }
const it = a(); console.log(it.next(), it.return('r'), it.next(), it.next());
function* b() { const x = yield* (function* () { const y = yield 'in'; return y * 2; })(); yield x; } const g = b(); console.log(g.next(), g.next(21), g.next());
const obj = { *[Symbol.iterator]() { yield* [1, 2, 3]; } }; console.log([...obj], Array.from(obj, x => x * x), new Set(obj).size);
function* inf() { let i = 0; while (true) yield i++; } const take = (n, it) => { const r = []; for (const v of it) { if (r.length >= n) break; r.push(v); } return r; }; console.log(take(5, inf()));
"""),
("iterator_helpers_if_present", """
const it = [1, 2, 3, 4].values(); console.log(typeof it.map, typeof Iterator);
if (typeof Iterator !== 'undefined' && it.map) { console.log(typeof it.take, typeof Iterator.from); }
console.log(Object.prototype.toString.call([].values()), Object.prototype.toString.call(new Map().entries()), Object.prototype.toString.call(''[Symbol.iterator]()), typeof [][Symbol.iterator]().next);
"""),
("exceptions_finally_order", """
function f() { const log = []; try { try { throw 1; } finally { log.push('f1'); } } catch (e) { log.push('c' + e); } finally { log.push('f2'); } return log; } console.log(f());
function g() { try { throw new Error('a'); } catch (e) { try { throw new Error('b'); } catch (e2) { return e.message + e2.message; } finally { console.log('inner finally'); } } } console.log(g());
function h() { l: try { return 'ret'; } finally { break l; } return 'after'; } console.log(h());
function k() { for (const x of [1, 2, 3]) { try { if (x === 2) continue; if (x === 3) return x; } finally { console.log('fin', x); } } } console.log(k());
const r = (() => { try { return 'a'; } finally { console.log('side'); } })(); console.log(r);
"""),
("error_properties", """
const e = new Error('msg'); console.log(Object.keys(e), e.hasOwnProperty('message'), e.hasOwnProperty('stack'), typeof e.stack, e.toString(), Object.prototype.toString.call(e));
console.log(new TypeError('t').name, TypeError.prototype.name, RangeError.prototype instanceof Error, new AggregateError([new Error('x')], 'agg').errors.length, Error('no new') instanceof Error, Error.prototype.message === '', Error.length);
console.log(String(new Error()), String(new Error(undefined)), String(new Error(null)), String(Object.assign(new Error('m'), { name: 'Custom' })), String(Object.assign(new Error('m'), { name: '' })));
try { null.f(); } catch (e) { console.log(e.message); } try { undefined.x = 1; } catch (e) { console.log(e.message); } try { ({}).x.y.z; } catch (e) { console.log(e.message); }
try { x = y + 1; let y; } catch (e) { console.log(e.name, e.message); } try { new (class A { constructor() { this.x; } }); class B extends Object { constructor() { this.x; } } new B(); } catch (e) { console.log(e.name); }
"""),
("coercion_edge_cases", """
const vals = [0, -0, 1, NaN, '', ' ', '0', 'a', null, undefined, [], [0], [1, 2], {}, true, false, 1n, Symbol.iterator];
for (const v of vals) { let s; try { s = [Boolean(v), typeof v === 'symbol' ? 'sym' : String(v), typeof v === 'bigint' ? Number(v) : (typeof v === 'symbol' ? 'sym' : Number(v))].join('|'); } catch (e) { s = e.name; } console.log(s); }
console.log([1] + [2], {} + 1, [] * 2, '5' * '5', null + 1, undefined + 1, true + '1', [] == ![], NaN != NaN, '' == 0, '0' == false, null >= 0, null > 0, null == 0, undefined == 0);
console.log(new Number(5) + 1, new String('a') + 'b', typeof new Boolean(false), !!new Boolean(false), new Number(1) === 1, Object(1) == 1, JSON.stringify([new Number(1), new String('s'), new Boolean(true)]));
"""),
("number_parsing_edge", """
for (const s of ['0b101', '0o17', '0xFf', '1_000', '1e', '.e1', '+.5', '-', '  \\n42\\t', '١٢٣', '1,5', 'Infinity', '-Infinity', 'infinity', '0x', '00012', '1e-400', '9e999']) console.log(JSON.stringify(s), Number(s), parseInt(s), parseFloat(s));
console.log(0.1 * 3, 1 / 3, 100 / 3, 2 ** 0.5, 10 ** 21, 10 ** -7, 123456789 * 987654321, 0xFFFFFFFF | 0, 0xFFFFFFFF >>> 0, 2 ** 32 >> 0, -(2 ** 31) | 0, 1e21 | 0, NaN | 0, Infinity | 0);
"""),
("string_replace_functions", """
console.log('abcabc'.replace('b', (m, off, str) => `[${m}@${off}/${str.length}]`), 'abc'.replace(/(?<l>b)/, (...args) => typeof args[args.length - 1]), 'aaa'.replace(/a/g, (m, off) => off), 'x-y_z'.replace(/[-_](.)/g, (_, c) => c.toUpperCase()));
console.log('a b  c'.split(' '), 'a b  c'.split(/\\s+/), 'abc'.split('', 2), ''.split(''), ''.split('x'), 'a1b2'.split(/(\\d)/), 'test'.split('t'), 'ab'.split(/(?:)/));
console.log('$1 $& $$'.replace(/(\\$)/g, '[$1]'), 'abc'.replace('b', '$`|$\\''), 'abc'.replace(/(b)/, '$01$10$2'), 'aXb'.replace(/x/i, '$&$&'), 'abc'.replaceAll(/b/g, 'X'));
try { 'abc'.replaceAll(/b/, 'X'); } catch (e) { console.log(e.name); }
console.log('abc'.at(5), 'abc'.charAt(5) === '', 'abc'[5], 'abc'.substring(2, 0), 'abc'.substr(1), 'abc'.slice(2, 1), ' \\t x'.trimStart(), 'x \\n'.trimEnd() + '|', 'abc'.padStart(2), 'abc'.padEnd(6, ''), 'ab'.padEnd(5, 'xyz'));
"""),
("array_like_generic_methods", """
const al = { 0: 'a', 1: 'b', 2: 'c', length: 3 };
console.log(Array.prototype.map.call(al, x => x.toUpperCase()), Array.prototype.join.call(al, '-'), Array.prototype.indexOf.call(al, 'b'), Array.prototype.slice.call(al, 1), Array.prototype.reverse.call({ 0: 1, 1: 2, length: 2 }), Array.prototype.push.call({ length: 1 }, 'x'));
console.log(Array.from('abc').entries().toString(), Array.from({ length: 3 }), Array.from(new Set([1, 1, 2])), Array.from([1, 2, 3], function (x) { return x * this.m; }, { m: 10 }), [].concat([1], [[2]], 3), [3, 2, 1].sort().concat([0]));
console.log(Array.prototype.filter.call('hello', c => c > 'h').join(''), Array.prototype.some.call('abc', c => c === 'b'), [].slice.call({ length: 2, 0: 'x' }), Array.prototype.fill.call({ length: 2 }, 7), Array.prototype.concat.call('a', 'b'));
"""),
("object_property_order", """
const o = { z: 1, 10: 'a', b: 2, 2: 'b', [Symbol('s')]: 3, '-1': 'neg', '01': 'lead', 4294967294: 'maxidx', 4294967295: 'notidx' };
console.log(Object.keys(o), JSON.stringify(o), Reflect.ownKeys(o).length); for (const k in o) process_k(k); function process_k(k) { console.log('key', k); }
o.a = 0; delete o.z; o.z = 9; console.log(Object.keys(o).join()); console.log(Object.entries({ b: 1, a: 2 }).sort().join(';'));
class P { constructor() { this.b = 1; this.a = 2; } } P.prototype.c = 3; console.log(Object.keys(new P()), (() => { const r = []; for (const k in new P()) r.push(k); return r; })());
"""),
("map_set_ordering_mutation", """
const m = new Map([['a', 1], ['b', 2], ['c', 3]]); m.delete('a'); m.set('a', 4); console.log([...m.keys()]);
const seen = []; for (const [k] of m) { seen.push(k); if (k === 'b') { m.delete('c'); m.set('d', 5); } } console.log(seen, [...m.keys()]);
const s = new Set([1, 2, 3]); const out = []; for (const v of s) { out.push(v); if (v < 3) { s.delete(v); s.add(v + 10); } if (out.length > 10) break; } console.log(out, [...s]);
const big = new Map(); for (let i = 0; i < 1000; i++) big.set(i, i * i); for (let i = 0; i < 1000; i += 3) big.delete(i); console.log(big.size, [...big.keys()].slice(0, 5), big.get(999), big.get(3));
console.log(new Map([[NaN, 1]]).get(NaN), new Set([0, -0]).size, [...new Set([-0])][0] === 0 && 1 / [...new Set([-0])][0], new Map([[{}, 1], [{}, 2]]).size);
"""),
("date_parsing_formats", """
const forms = ['2024-03-15', '2024-03-15T10:20:30', '2024-03-15T10:20:30.123Z', '2024-03-15T10:20:30+05:30', '2024-03', '2024', '+002024-03-15T00:00:00Z', 'Mar 15, 2024', 'March 15, 2024 10:20:30 UTC', '15 Mar 2024 10:20 GMT', 'Fri, 15 Mar 2024 10:20:30 GMT', '2024/03/15', '3/15/2024', 'not a date', '2024-13-01', '2024-02-30', '2024-03-15T25:00:00Z', ''];
for (const f of forms) { const t = Date.parse(f); console.log(JSON.stringify(f), Number.isNaN(t) ? 'NaN' : new Date(t).toISOString()); }
"""),
("date_setters_arithmetic", """
const d = new Date(Date.UTC(2024, 0, 31, 12)); const ops = [['setUTCMonth', 1], ['setUTCDate', 31], ['setUTCDate', 0], ['setUTCFullYear', 2023, 1, 29], ['setUTCHours', 24], ['setUTCMinutes', -1], ['setUTCSeconds', 3600], ['setUTCMilliseconds', 1500], ['setTime', 0]];
for (const [fn, ...a] of ops) { const r = d[fn](...a); console.log(fn, a.join(), r, d.toISOString()); }
const x = new Date(Date.UTC(2024, 2, 10)); console.log(x.getUTCDay(), new Date(x.getTime() + 864e5 * 365).toISOString(), Math.round((Date.UTC(2025, 0, 1) - Date.UTC(2024, 0, 1)) / 864e5), new Date(2024, 0, 1, 0, 0, 0, 0) < new Date(2024, 0, 1, 0, 0, 0, 1));
console.log(new Date(2024, 0).getTime() === new Date('2024-01-01T00:00:00').getTime(), new Date(0, 0).getFullYear(), new Date(100, 0).getFullYear(), new Date(-1).toISOString(), new Date(1e15).toISOString(), new Date(2 ** 31 * 1000).toISOString());
"""),
("regexp_backtracking_unicode", """
console.log(/^(a+)+$/.test('aaaaaaaaaaaaaaaaaaaa'), /(x+x+)+y/.test('xxxxxxxxxxxxxxxxxx'), 'aaa'.match(/a*?/)[0].length, 'aaa'.match(/a*?$/)[0], /(a|ab)(c|bcd)(d*)/.exec('abcd'), /(?=(a+))a*b\\1/.exec('baaabac'), /(?!a)\\w+/.exec('abc'));
console.log('The Quick Brown'.match(/\\b\\w/g), 'ÀÉÎ'.match(/\\p{Lu}/gu), 'a\\u0301'.match(/./gu).length, '😀😁'.match(/./gu).length, '😀😁'.match(/./g).length, /[\\u{1F600}-\\u{1F64F}]/u.test('😁'), /\\w/u.test('é'), /\\w/iu.test('ſ'), /[\\w-]/.test('-'), /\\cJ/.test('\\n'), /[\\b]/.test('\\b'));
console.log('2024-03-15'.match(/(?<y>\\d+)-(?<m>\\d+)-(?<d>\\d+)/).groups, /(?<a>.)\\k<a>/.test('xx'), 'aBc'.replace(/b/i, m => m.toLowerCase()), /a(?<!b)/.test('a'), /^[^\\d\\s]+$/.test('ab c'), /\\d{3,}/.exec('12 123 1234')[0], 'x'.repeat(100).replace(/x{10}/g, 'T'));
"""),
("typed_array_conversions", """
const f64 = new Float64Array([1.5, -0, NaN, Infinity, 1e300]); const u8 = new Uint8Array(f64.buffer); console.log(u8.length, [...u8.slice(0, 8)], new Float64Array(u8.buffer)[1], Object.is(new Float64Array(u8.buffer)[1], -0));
console.log(Array.from(new Uint8Array([1, 2, 3]).map(x => x * 100)), new Int8Array([1, 2, 3]).map(x => x * 100), new Uint16Array([1, 2, 3]).filter(x => x > 1), new Uint8Array(5).fill(7, 1, 3), new Int32Array([5, 1, 4]).sort((a, b) => b - a), new Float32Array([3, 1, 2]).sort().reverse());
const dv = new DataView(new ArrayBuffer(8)); dv.setUint16(0, 0x1234); dv.setUint16(2, 0x1234, true); console.log([...new Uint8Array(dv.buffer, 0, 4)].map(x => x.toString(16)), dv.getUint32(0), dv.getUint32(0, true), dv.getInt16(2, true));
console.log(Uint8Array.from({ length: 3 }, (_, i) => i * 100), new Uint8Array([1, 2, 3]).toString(), Array.prototype.slice.call(new Uint8Array([9, 8])), JSON.stringify(new Uint8Array([1, 2])), new Uint8Array([1, 2, 3]).subarray(1, 2).byteOffset, new Uint8Array(new ArrayBuffer(4), 1, 2).byteLength);
console.log(Atomics.add(new Int32Array(new SharedArrayBuffer(8)), 0, 5), typeof Atomics.wait, new Int32Array(new SharedArrayBuffer(8)).length);
"""),
("symbol_well_known", """
class Temp { constructor(c) { this.c = c; } [Symbol.toPrimitive](hint) { return hint === 'string' ? `${this.c}C` : hint === 'number' ? this.c : `default:${this.c}`; } get [Symbol.toStringTag]() { return 'Temp'; } static [Symbol.hasInstance](x) { return 'c' in x; } }
const t = new Temp(20); console.log(`${t}`, +t, t + '', t * 2, String(t), Object.prototype.toString.call(t), ({ c: 1 }) instanceof Temp, [] instanceof Temp);
const arr = [1, 2, 3]; arr[Symbol.isConcatSpreadable] = false; console.log([0].concat(arr).length, [0].concat({ length: 2, 0: 'a', 1: 'b', [Symbol.isConcatSpreadable]: true }));
class MyArr extends Array { static get [Symbol.species]() { return Array; } } const ma = new MyArr(1, 2, 3); console.log(ma.map(x => x) instanceof MyArr, ma instanceof MyArr, ma.filter(Boolean).constructor.name);
const re = { [Symbol.match](s) { return ['custom:' + s]; }, [Symbol.replace](s, r) { return s + '->' + r; }, [Symbol.split](s) { return ['sp', s]; }, [Symbol.search]() { return 42; } }; console.log('x'.match(re), 'x'.replace(re, 'y'), 'x'.split(re), 'x'.search(re));
console.log(Symbol.asyncIterator.toString(), Symbol.unscopables.description, Object.getOwnPropertySymbols(Array.prototype).length > 0, Array.prototype[Symbol.unscopables].flat);
"""),
("weakref_finalization_gc", """
let o = { big: new Array(1000).fill(1) }; const wr = new WeakRef(o); console.log(wr.deref() === o, typeof FinalizationRegistry, typeof gc);
const reg = new FinalizationRegistry(() => {}); reg.register(o, 'tok', o); console.log(reg.unregister(o), reg.unregister({}));
const cache = new Map(); function mk(i) { return { i, payload: 'x'.repeat(100) }; } for (let i = 0; i < 5000; i++) cache.set(i % 50, mk(i)); console.log(cache.size, cache.get(7).i);
let s = ''; for (let i = 0; i < 200; i++) { const a = []; for (let j = 0; j < 100; j++) a.push({ j, a }); s = String(a.length); } console.log(s);
"""),
("proxy_invariants_traps", """
const t = {}; Object.defineProperty(t, 'fixed', { value: 1, configurable: false }); const p = new Proxy(t, { get: () => 2, has: () => false, ownKeys: () => [], deleteProperty: () => true });
for (const f of [() => p.fixed, () => 'fixed' in p, () => Object.keys(p), () => delete p.fixed, () => Object.getOwnPropertyDescriptor(p, 'fixed')]) { try { console.log('ok', f()); } catch (e) { console.log(e.name); } }
const calls = []; const q = new Proxy([], { get(t, k, r) { calls.push(String(k)); return Reflect.get(t, k, r); }, set(t, k, v, r) { calls.push('set ' + String(k)); return Reflect.set(t, k, v, r); } }); q.push(1, 2); console.log(calls.join());
console.log(Array.isArray(new Proxy([], {})), typeof new Proxy(function () {}, {}), typeof new Proxy({}, {}), Object.prototype.toString.call(new Proxy([], {})), JSON.stringify(new Proxy({ a: [1] }, {})), new Proxy({}, { getPrototypeOf: () => Array.prototype }) instanceof Array);
"""),
("module_like_patterns", """
const mod = (function () { let priv = 0; const api = { inc: () => ++priv, get count() { return priv; } }; return Object.freeze(api); })(); mod.inc(); mod.inc(); console.log(mod.count, Object.isFrozen(mod));
const EventEmitter = class { #h = {}; on(e, f) { (this.#h[e] ||= []).push(f); return this; } emit(e, ...a) { (this.#h[e] || []).forEach(f => f(...a)); return !!this.#h[e]; } };
const em = new EventEmitter(); em.on('x', (a, b) => console.log('x1', a, b)).on('x', a => console.log('x2', a)); console.log(em.emit('x', 1, 2), em.emit('y'));
const curry = f => function c(...a) { return a.length >= f.length ? f(...a) : (...b) => c(...a, ...b); }; console.log(curry((a, b, c) => a + b + c)(1)(2)(3), curry((a, b) => a * b)(3, 4));
const compose = (...fs) => x => fs.reduceRight((v, f) => f(v), x); console.log(compose(x => x + 1, x => x * 2)(5));
"""),
("lru_and_trie_programs", """
class LRU { constructor(n) { this.n = n; this.m = new Map(); } get(k) { if (!this.m.has(k)) return -1; const v = this.m.get(k); this.m.delete(k); this.m.set(k, v); return v; } put(k, v) { this.m.delete(k); this.m.set(k, v); if (this.m.size > this.n) this.m.delete(this.m.keys().next().value); } }
const c = new LRU(2); c.put(1, 1); c.put(2, 2); console.log(c.get(1)); c.put(3, 3); console.log(c.get(2), c.get(3)); c.put(4, 4); console.log(c.get(1), c.get(3), c.get(4));
const trie = {}; for (const w of ['car', 'card', 'care', 'cat', 'dog', 'do']) { let n = trie; for (const ch of w) n = n[ch] ||= {}; n.$ = true; }
const words = (n, p = '') => Object.entries(n).flatMap(([k, v]) => k === '$' ? [p] : words(v, p + k)); console.log(words(trie).sort().join(), words(trie.c.a).length);
"""),
("dijkstra_graph", """
const g = { A: { B: 7, C: 9, F: 14 }, B: { A: 7, C: 10, D: 15 }, C: { A: 9, B: 10, D: 11, F: 2 }, D: { B: 15, C: 11, E: 6 }, E: { D: 6, F: 9 }, F: { A: 14, C: 2, E: 9 } };
function dijk(s) { const d = Object.fromEntries(Object.keys(g).map(k => [k, Infinity])); d[s] = 0; const q = new Set(Object.keys(g)); while (q.size) { let u = null; for (const v of q) if (u === null || d[v] < d[u]) u = v; q.delete(u); for (const [v, w] of Object.entries(g[u])) if (d[u] + w < d[v]) d[v] = d[u] + w; } return d; } console.log(dijk('A'));
"""),
("json_pretty_deep_and_unicode", """
let deep = {}; let cur = deep; for (let i = 0; i < 50; i++) { cur.n = { i }; cur = cur.n; } const s = JSON.stringify(deep); console.log(s.length, JSON.parse(s).n.n.n.i);
console.log(JSON.stringify({ 'ключ': 'значение', e: '😀', ctl: '\\u0000\\u001f\\u007f', quote: '"\\\\/', big: 1e300, small: 1e-300, arr: [[]], o: {} }, null, '--'));
console.log(JSON.stringify('\\ud83d'), JSON.stringify('\\ud83d\\ude00'), JSON.parse('"\\\\ud83d\\\\ude00"'), JSON.parse('{"__proto__": 1}').__proto__, Object.keys(JSON.parse('{"__proto__": {"x": 1}}')));
console.log(JSON.parse('[1, 2, 3]', function (k, v) { return Array.isArray(v) ? v.length : v * 2; }), JSON.stringify({ a: 1, b: 2, c: 3 }, ['c', 'a']), JSON.stringify([1, [2, [3]]], null, 1).split('\\n').length);
"""),
("labeled_blocks_switch_edge", """
function f(x) { switch (typeof x) { case 'number': if (x > 0) { return 'pos'; } case 'string': return 'num-or-str'; default: return 'other'; case 'boolean': return 'bool'; } } console.log(f(1), f(-1), f('s'), f(true), f(null));
function g(x) { switch (x) { case 1: case 2: return 'low'; case NaN: return 'nan'; case '1': return 'str1'; default: return 'dflt'; } } console.log(g(1), g(2), g('1'), g(NaN), g(3));
a: for (var i = 0; i < 3; i++) { b: for (var j = 0; j < 3; j++) { if (j === 1) continue a; if (i === 2) break a; console.log(i, j); } } console.log(i, j);
var r = []; x: { r.push(1); y: { r.push(2); break x; } r.push(3); } console.log(r);
let n = 0; while (true) { if (++n > 5) break; else continue; } do n--; while (n > 3); console.log(n);
for (var k = 0, m = 10; k < m; k += 3, m -= 3); console.log(k, m);
"""),
("recursion_depth_tail", """
function sum(n) { return n === 0 ? 0 : n + sum(n - 1); } console.log(sum(50), sum(100));
let depth = 0; function probe() { depth++; probe(); } try { probe(); } catch (e) { console.log(e.name); }
function even(n) { return n === 0 ? true : odd(n - 1); } function odd(n) { return n === 0 ? false : even(n - 1); } console.log(even(100), odd(77));
const fact = n => n <= 1n ? 1n : n * fact(n - 1n); console.log(String(fact(60n)).length);
"""),
("string_building_large", """
let s = ''; for (let i = 0; i < 30000; i++) s += i % 10; console.log(s.length, s.slice(-5)); const parts = []; for (let i = 0; i < 30000; i++) parts.push(i % 7); console.log(parts.join('').length);
console.log('ab'.repeat(50000).length, 'x'.repeat(1e6).length, ('a'.repeat(1000) + 'b').lastIndexOf('b'), 'abc'.repeat(3).split('c').length);
let w = ''; for (let i = 0; i < 2000; i++) w = 'é' + w + '😀'; console.log(w.length, w.codePointAt(0), w.at(-1).charCodeAt(0));
const rope = []; for (let i = 0; i < 100; i++) rope.push('x'.repeat(i)); console.log(rope.join('|').length, rope.join('').length, rope.reduce((a, b) => a + b.length, 0));
"""),
("numeric_loops_float_accumulation", """
let s = 0; for (let i = 1; i <= 100000; i++) s += 1 / (i * i); console.log(s, Math.sqrt(s * 6));
let a = 0; for (let i = 0; i < 10; i++) a += 0.1; console.log(a, a === 1, Math.abs(a - 1) < Number.EPSILON);
let h = 0; for (let i = 0; i < 100000; i++) h = (h * 31 + i) >>> 0; console.log(h); let x = 1; for (let i = 0; i < 100; i++) x = x * 1.1 % 7; console.log(x);
const pi = (n) => { let p = 0; for (let k = 0; k < n; k++) p += (k % 2 ? -1 : 1) / (2 * k + 1); return 4 * p; }; console.log(pi(100000), pi(10).toFixed(6));
console.log(Array.from({ length: 10 }, (_, i) => (i * 0.1).toFixed(1)).join(), (1e21 + 1).toString(), 2 ** 1023 * 2, -(2 ** 1023) * 2, 5e-324 / 2, (0.1).toFixed(20), (1.45).toFixed(1), (8.345).toFixed(2));
"""),
]
