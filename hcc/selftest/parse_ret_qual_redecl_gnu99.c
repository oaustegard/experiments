// EXPECT-FLAGS: -std=gnu99
// EXPECT-ERROR: conflicting types
// C89/C99: a top-level const on a non-void return type is part of the function type (GCC
// -std=gnu99 rejects this; under -std=gnu11 and later it is accepted, see sema_func_ret_qual_redecl.c).
int f(void);
const int f(void) { return 4; }
int main(void) { return f(); }
