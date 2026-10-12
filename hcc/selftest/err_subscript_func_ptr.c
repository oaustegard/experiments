// EXPECT-ERROR: [sema] subscripted value is pointer to function
void (*f)(void);
void g(void) { f[0]; }
