function sum(n) { return n === 0 ? 0 : n + sum(n - 1); } console.log(sum(50), sum(100));
let depth = 0; function probe() { depth++; probe(); } try { probe(); } catch (e) { console.log(e.name); }
function even(n) { return n === 0 ? true : odd(n - 1); } function odd(n) { return n === 0 ? false : even(n - 1); } console.log(even(100), odd(77));
const fact = n => n <= 1n ? 1n : n * fact(n - 1n); console.log(String(fact(60n)).length);
