console.log(/(\d{4})-(\d{2})-(\d{2})/.exec('on 2024-03-15!'), 'x'.match(/y/), 'aAbB'.match(/[a-z]/gi), 'abc'.search(/c/), /^\s*$/.test('   '), /a.c/s.test('a\nc'), /a.c/.test('a\nc'));
const re = /(?<year>\d{4})-(?<mon>\d{2})/u; const m = re.exec('2024-03'); console.log(m.groups.year, m.groups.mon, '2024-03'.replace(re, '$<mon>/$<year>'));
console.log('aXbxc'.split(/x/i), [...'a1b2c3'.matchAll(/[a-z](\d)/g)].map(m => m[1] + '@' + m.index), 'abcabc'.replace(/(?<=a)b/g, 'B'), 'abcabc'.replace(/b(?!c)/g, 'X'), 'price: $10, $20'.match(/(?<=\$)\d+/g));
const g = /o/g; console.log(g.test('foo'), g.lastIndex, g.test('foo'), g.lastIndex, g.test('foo'), g.lastIndex); const y = /a/y; y.lastIndex = 1; console.log(y.test('ba'), y.lastIndex);
console.log(/\u{1F600}/u.test('😀'), /^.$/u.test('😀'), /^.$/.test('😀'), /\p{L}+/u.exec('héllo wörld')[0], /\p{Script=Greek}/u.test('α'), /[\p{Lu}]/u.test('É'), /(a)|(b)/.exec('b'), /(?:x)(y)?/.exec('x'));
console.log(/a{2,3}/.exec('aaaa')[0], /a{2,3}?/.exec('aaaa')[0], /(a+)+b/.test('aaaaaaaaaaaaaaaaaaaac'), /\bfoo\b/.test('a foo b'), /(\w+)\s\1/.test('hello hello'), /[^]/.test('\n'), new RegExp('a+', 'gi').flags, /x/dgimsuy.flags);
