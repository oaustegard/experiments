const o = { s: 'str\n"q"', n: 1.5, i: 1e21, neg: -0, nan: NaN, inf: Infinity, u: undefined, f() {}, sym: Symbol('x'), d: new Date(0), nested: { a: [1, { b: null }], t: true }, big: [], e: {} };
console.log(JSON.stringify(o)); console.log(JSON.stringify(o, null, 2)); console.log(JSON.stringify(o, ['s', 'n']), JSON.stringify([undefined, function () {}, Symbol()]), JSON.stringify('\u2028\ud800'));
console.log(JSON.stringify({ a: 1, b: [1, 2] }, (k, v) => typeof v === 'number' ? v * 2 : v, '\t'), JSON.stringify({ toJSON() { return 'tj'; } }), JSON.stringify(null), JSON.stringify(undefined));
console.log(JSON.parse('{"a":[1,2,{"b":"c"}],"d":1e3,"e":-0.5,"f":"\\u00e9"}'), JSON.parse('[1,2]', (k, v) => Array.isArray(v) ? v : v + 1), JSON.parse(' "x" '), JSON.parse('1E2'));
for (const bad of ['{a:1}', "{'a':1}", '[1,]', '01', '', '{"a":1', 'undefined', '"\n"']) { try { JSON.parse(bad); console.log('ok?', bad); } catch (e) { console.log(e.name); } }
const cyc = {}; cyc.self = cyc; try { JSON.stringify(cyc); } catch (e) { console.log(e.name); }
try { JSON.stringify(1n); } catch (e) { console.log(e.name); }
