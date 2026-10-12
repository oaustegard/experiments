// EXPECT-ERROR: initializer element is not constant
// A static initializer may not take the address of a thread-local object.
extern __thread int tl;
int *p = &tl;
