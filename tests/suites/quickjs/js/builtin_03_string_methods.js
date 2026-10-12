const s = 'Hello, World';
console.log(s.length, s.charAt(4), s.charCodeAt(1), s.codePointAt(0), s[0], s.at(-1), s.indexOf('o'), s.lastIndexOf('o'), s.includes('World'), s.startsWith('Hell'), s.endsWith('d'));
console.log(s.slice(-5), s.substring(7, 3), s.substr(-5, 3), s.toUpperCase(), s.toLowerCase(), s.split(', '), s.split(''), 'a,b,,c'.split(',', 3), s.split(/(,)/));
console.log('  pad '.trim() + '|', 'ab'.padStart(5, 'xy'), 'ab'.padEnd(5), 'ab'.repeat(3), 'a-b-c'.replace('-', '+'), 'a-b-c'.replaceAll('-', '+'), 'abc'.replace(/b/, '[$&]'), 'x'.concat(1, 2));
console.log('a1b22c333'.replace(/\d+/g, m => `<${m.length}>`), 'John Smith'.replace(/(\w+)\s(\w+)/, '$2, $1'), 'abc'.localeCompare('abd'), 'b'.localeCompare('a'), 'a'.localeCompare('a'));
console.log('\u00e9'.normalize('NFD').length, 'é'.normalize('NFD') === 'e\u0301', '😀'.length, [...'😀'].length, '😀'.codePointAt(0), String.fromCodePoint(128512), String.fromCharCode(72, 105), 'ß'.toUpperCase(), 'İ'.toLowerCase().length);
console.log('abc'.isWellFormed?.(), 'a\ud800'.isWellFormed?.(), 'a\ud800'.toWellFormed?.().charCodeAt(1), 'aXbX'.split('X'), 'abc'.split('b').length, '%'.codePointAt(), 'z'.localeCompare('Z'));
