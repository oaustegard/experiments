// EXPECT-ERROR: 'static' may not be used without an array size
// C99 6.7.5.2: `static` in an array parameter declarator requires a size.
void f(char buf[static]);
