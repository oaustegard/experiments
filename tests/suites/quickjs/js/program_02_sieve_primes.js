const N = 20000, sieve = new Uint8Array(N + 1); const primes = [];
for (let i = 2; i <= N; i++) { if (!sieve[i]) { primes.push(i); for (let j = i * i; j <= N; j += i) sieve[j] = 1; } }
console.log(primes.length, primes.slice(-3), primes.reduce((a, b) => a + b, 0));
