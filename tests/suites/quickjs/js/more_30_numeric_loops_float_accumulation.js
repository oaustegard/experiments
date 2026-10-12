let s = 0; for (let i = 1; i <= 100000; i++) s += 1 / (i * i); console.log(s, Math.sqrt(s * 6));
let a = 0; for (let i = 0; i < 10; i++) a += 0.1; console.log(a, a === 1, Math.abs(a - 1) < Number.EPSILON);
let h = 0; for (let i = 0; i < 100000; i++) h = (h * 31 + i) >>> 0; console.log(h); let x = 1; for (let i = 0; i < 100; i++) x = x * 1.1 % 7; console.log(x);
const pi = (n) => { let p = 0; for (let k = 0; k < n; k++) p += (k % 2 ? -1 : 1) / (2 * k + 1); return 4 * p; }; console.log(pi(100000), pi(10).toFixed(6));
console.log(Array.from({ length: 10 }, (_, i) => (i * 0.1).toFixed(1)).join(), (1e21 + 1).toString(), 2 ** 1023 * 2, -(2 ** 1023) * 2, 5e-324 / 2, (0.1).toFixed(20), (1.45).toFixed(1), (8.345).toFixed(2));
