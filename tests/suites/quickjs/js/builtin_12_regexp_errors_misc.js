for (const p of ['(', '[a-', 'a{2,1}', '(?<n>a)(?<n>b)', '\\', '*a', 'a**', '(?<=a)+', '\\k<x>', '[b-a]']) { try { new RegExp(p, 'u'); console.log('ok', p); } catch (e) { console.log(e.name, p); } }
try { new RegExp('a', 'zz'); } catch (e) { console.log(e.name); }
console.log(String(/a\/b/), new RegExp('a/b').source, new RegExp('\n').source, /[/]/.source, RegExp.prototype.toString.call(/x/g), RegExp('x', 'g') + '', /a/ instanceof RegExp);
console.log('aaa'.lastIndexOf('a'), 'test'.replace(/t/g, '$$'), 'test'.replace(/(t)/g, "$1$1$`"), 'abc'.replace(/(?<x>b)/, (...a) => JSON.stringify(a.slice(-1))), 'a.b'.split('.'), 'a1b'.split(/\d/), 'abc'.split(/(?:)/u));
