// EXPECT-FLAGS: -std=c23 -pedantic-errors
// EXPECT-ERROR: subscripting 'register' array
// Subscripting a `register` array is only valid from C2Y on; before that,
// -pedantic-errors makes it an error.
int main(void) { register int a[2]; a[0] = 1; return a[0]; }
