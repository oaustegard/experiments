console.log('abcabc'.replace('b', (m, off, str) => `[${m}@${off}/${str.length}]`), 'abc'.replace(/(?<l>b)/, (...args) => typeof args[args.length - 1]), 'aaa'.replace(/a/g, (m, off) => off), 'x-y_z'.replace(/[-_](.)/g, (_, c) => c.toUpperCase()));
console.log('a b  c'.split(' '), 'a b  c'.split(/\s+/), 'abc'.split('', 2), ''.split(''), ''.split('x'), 'a1b2'.split(/(\d)/), 'test'.split('t'), 'ab'.split(/(?:)/));
console.log('$1 $& $$'.replace(/(\$)/g, '[$1]'), 'abc'.replace('b', '$`|$\''), 'abc'.replace(/(b)/, '$01$10$2'), 'aXb'.replace(/x/i, '$&$&'), 'abc'.replaceAll(/b/g, 'X'));
try { 'abc'.replaceAll(/b/, 'X'); } catch (e) { console.log(e.name); }
console.log('abc'.at(5), 'abc'.charAt(5) === '', 'abc'[5], 'abc'.substring(2, 0), 'abc'.substr(1), 'abc'.slice(2, 1), ' \t x'.trimStart(), 'x \n'.trimEnd() + '|', 'abc'.padStart(2), 'abc'.padEnd(6, ''), 'ab'.padEnd(5, 'xyz'));
