// EXPECT-ERROR: conflicting types
// A transparent_union parameter is not the untagged-union extension (which only warns):
// a redeclaration whose parameter type differs from the union is an error.
typedef union { int *i; long *l; } U __attribute__((transparent_union));
extern void f0(U);
extern void f0(void *);
int main(void) { return 0; }
