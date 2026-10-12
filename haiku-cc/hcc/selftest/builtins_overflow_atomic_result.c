// EXPECT-ERROR: [parse]
// The result pointer of __builtin_add_overflow may not point to an _Atomic object.
_Atomic int a = 1, b = 2, c = 3;
void f(void) { __builtin_add_overflow(a, b, &c); }
