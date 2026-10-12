function solve(n) { let count = 0; const cols = new Array(n).fill(false), d1 = new Array(2 * n).fill(false), d2 = new Array(2 * n).fill(false);
  (function place(r) { if (r === n) { count++; return; } for (let c = 0; c < n; c++) { if (cols[c] || d1[r + c] || d2[r - c + n]) continue; cols[c] = d1[r + c] = d2[r - c + n] = true; place(r + 1); cols[c] = d1[r + c] = d2[r - c + n] = false; } })(0); return count; }
console.log([4, 5, 6, 7, 8].map(solve).join());
