const a = 2n ** 100n, b = 12345678901234567890n; console.log(a, a * b, a / 7n, a % 1000007n, -a, a - a, b >> 5n, b << 3n, b & 0xffffn, b | 1n, b ^ b, ~b);
console.log(typeof a, 1n < 2, 2n == 2, 2n === 2, 10n > 9.5, BigInt(123), BigInt('0x1f'), BigInt(Number.MAX_SAFE_INTEGER) + 2n, Number(2n ** 64n), parseInt('9'.repeat(30)), BigInt.asUintN(8, 257n), BigInt.asIntN(8, 255n), (123n).toString(16), (-5n).toString(2));
let f = 1n; for (let i = 1n; i <= 30n; i++) f *= i; console.log(f, f.toString().length, String(f).split('').reduce((s, d) => s + +d, 0));
try { 1n + 1; } catch (e) { console.log(e.name); } try { BigInt(1.5); } catch (e) { console.log(e.name); } try { 1n / 0n; } catch (e) { console.log(e.name); } try { Math.max(1n); } catch (e) { console.log(e.name); }
console.log(5n / 2n, -5n / 2n, -5n % 3n, 0n === -0n, JSON.stringify({ v: 1 }), [3n, 1n, 2n].sort().join(), [3n, 1n, 10n].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0)).join());
