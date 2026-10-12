// EXPECT-ERROR: [parse] redefinition of 'f2'
static __attribute__((weakref("bar"))) void f2(void);
static void f2(void) { }
void *use_f2(void) { return f2; }
