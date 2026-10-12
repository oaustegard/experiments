const o = { b: 2, a: 1, 2: 'two', 1: 'one' };
console.log(Object.keys(o), Object.values(o), Object.entries(o), Object.assign({}, o, { c: 3 }), Object.fromEntries([['x', 1], ['y', 2]]), Object.fromEntries(new Map([[1, 2]])));
console.log(Object.getPrototypeOf(Object.create(null)), Object.create({ inherited: 1 }, { own: { value: 2, enumerable: true } }).inherited, 'inherited' in Object.create({ inherited: 1 }), Object.hasOwn({ a: 1 }, 'a'), ({}).hasOwnProperty.call({ a: 1 }, 'b'));
const proto = { greet() { return 'hi ' + this.name; } }; const x = Object.setPrototypeOf({ name: 'x' }, proto); console.log(x.greet(), proto.isPrototypeOf(x), Object.getOwnPropertyNames(Object.getPrototypeOf(x)));
console.log(Object.entries('ab'), Object.keys([5, 6]), Object.is(NaN, NaN), Object.isExtensible(Object.preventExtensions({})), JSON.stringify(Object.entries({ a: [1] })), typeof structuredClone);
console.log(String({}), String(Object.create(null, { toString: { value: () => 'custom' } })), `${[1, [2, 3]]}`, ({}).toString.call(null), ({}).toString.call([]), ({}).toString.call(new Date(0)), ({}).toString.call(/x/), ({}).toString.call(function () {}));
