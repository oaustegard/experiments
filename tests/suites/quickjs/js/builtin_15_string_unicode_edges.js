const s = 'a😀b'; console.log(s.length, s.split(''), [...s], s.slice(1, 2).length, s.charAt(1).charCodeAt(0), s.codePointAt(1), s.codePointAt(2), s.indexOf('b'), encodeURIComponent(s), s.normalize() === s);
console.log('Ω'.toLowerCase(), 'ǆ'.toUpperCase(), 'ŉ'.toUpperCase(), 'ﬃ'.toUpperCase(), 'Σ'.toLowerCase(), 'ΑΣ'.toLowerCase(), 'i'.toLocaleUpperCase(), '\u0130'.toLowerCase().length, 'ǅ'.toLowerCase());
console.log([...'ñ'.normalize('NFD')].map(c => c.charCodeAt(0).toString(16)), 'Å'.normalize('NFKC').charCodeAt(0).toString(16), '\ufb01'.normalize('NFKD'), '한글'.normalize('NFD').length, 'ａｂｃ'.normalize('NFKC'));
console.log('a\u0000b'.length, 'tab\there'.split('\t'), '\x41\u0042\u{43}', 'a\
b', escape('é€'), JSON.stringify('\u007f\u0080\u2028'), String.raw`\u0041`, 'abc'.localeCompare('ABC'), ['é', 'e', 'z', 'a'].sort(), ['é', 'e', 'z', 'a'].sort((a, b) => a.localeCompare(b)));
