const n = 40; const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i * j) % 7 + 1)), B = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i + j) % 5 - 2));
const C = A.map((row, i) => B[0].map((_, j) => row.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
console.log(C[0].slice(0, 5), C[n - 1][n - 1], C.flat().reduce((a, b) => a + b, 0));
