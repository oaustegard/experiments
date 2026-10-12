const fns = []; for (let i = 0; i < 1000; i++) { const big = new Array(10).fill(i); fns.push(() => big[0] + i); } console.log(fns.reduce((s, f) => s + f(), 0));
function make(n) { return n === 0 ? () => 0 : (f => () => f() + 1)(make(n - 1)); } console.log(make(100)());
