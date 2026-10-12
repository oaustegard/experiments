// EXPECT-ERROR: [sema]
// A label difference is a 32/64-bit reloc: not a load-time constant for __int128.
void f(void) { a: ; b: ; static __int128_t d = (long) &&a - (long) &&b; }
