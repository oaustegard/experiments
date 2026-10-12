// EXPECT-EXIT: 0
// A declaration with an empty parameter list replaces the hidden prototype
// that a __builtin_<libm>(...) call implicitly created.
__typeof (__builtin_ceil (0)) ceil (); 
long ceil1 (void) { return ceil (); }
int main (void) { return 0; }
