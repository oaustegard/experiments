// EXPECT-ERROR: conflicting types for 'f'
// An unprototyped declaration is incompatible with a variadic prototype (C11 6.7.6.3p15).
void f();
void f(int, ...);
