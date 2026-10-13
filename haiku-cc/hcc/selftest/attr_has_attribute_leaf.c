// EXPECT-EXIT: 3
// __builtin_has_attribute sees leaf and returns_nonnull; copy(f) does not inherit leaf.
__attribute__((leaf, returns_nonnull)) void *fn(void);
__attribute__((copy(fn), weakref("fn"))) static void *fn_copy(void);   // weakref needs static linkage (GCC)
_Static_assert(__builtin_has_attribute(fn, leaf), "leaf");
_Static_assert(__builtin_has_attribute(fn, returns_nonnull), "returns_nonnull");
_Static_assert(!__builtin_has_attribute(fn_copy, leaf), "no leaf via copy");
_Static_assert(__builtin_has_attribute(fn_copy, returns_nonnull), "nonnull via copy");
int main(void) { return 3; }
