// EXPECT-ERROR: wrong number of arguments
// nonnull_if_nonzero takes 2 or 3 arguments; one argument is an error.
extern void f (char *) __attribute__((nonnull_if_nonzero(1)));
int main (void) { return 0; }
