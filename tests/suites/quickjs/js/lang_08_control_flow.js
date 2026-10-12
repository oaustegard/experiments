outer: for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) { if (j == 2) continue outer; if (i == 3) break outer; console.log(i, j); } }
let k = 0; do { k++; } while (k < 5); console.log(k);
switch (3) { case 1: console.log('one'); case 3: console.log('three'); case 4: console.log('four'); break; default: console.log('d'); }
for (const x in { a: 1, b: 2, 3: 3, 1: 1 }) console.log(x);
for (const x of 'ab') console.log(x);
console.log([1, 2, 3].map(x => x % 2 ? 'odd' : 'even'));
l: { console.log('in'); break l; }
