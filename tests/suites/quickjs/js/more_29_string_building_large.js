let s = ''; for (let i = 0; i < 30000; i++) s += i % 10; console.log(s.length, s.slice(-5)); const parts = []; for (let i = 0; i < 30000; i++) parts.push(i % 7); console.log(parts.join('').length);
console.log('ab'.repeat(50000).length, 'x'.repeat(1e6).length, ('a'.repeat(1000) + 'b').lastIndexOf('b'), 'abc'.repeat(3).split('c').length);
let w = ''; for (let i = 0; i < 2000; i++) w = 'é' + w + '😀'; console.log(w.length, w.codePointAt(0), w.at(-1).charCodeAt(0));
const rope = []; for (let i = 0; i < 100; i++) rope.push('x'.repeat(i)); console.log(rope.join('|').length, rope.join('').length, rope.reduce((a, b) => a + b.length, 0));
