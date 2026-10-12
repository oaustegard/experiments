const n = 5, s = 'x'; console.log(`n=${n} s=${s} ${n * 2} ${[1, 2]} ${{}} ${null} ${undefined}`);
console.log(`line1
line2`, `\u{1F600}`.length, String.raw`a\nb${1}`, `a${`b${`c`}`}`);
function tag(strs, ...vals) { return strs.raw.join('|') + '#' + vals.join(','); } console.log(tag`a${1}b${2}c\n`);
