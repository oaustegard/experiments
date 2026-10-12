function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
const t = []; for (let i = 0; i <= 25; i += 5) t.push(fib(i)); console.log(t.join());
