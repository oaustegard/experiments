// EXPECT-ERROR: [parse] 'nonnull' attribute without arguments on a non-prototype
// A bare nonnull attribute needs a prototype to say which parameters are non-null.
extern void f () __attribute__((nonnull));
