// EXPECT-OK
// A bare nonnull attribute is fine on a function with a prototype.
extern void f (char *) __attribute__((nonnull));
int main(void) { return 0; }
